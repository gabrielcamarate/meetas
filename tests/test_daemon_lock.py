"""Single-instance lock, PID file and stale-lock handling."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.daemon import (
    AlreadyRunningError, acquire_lock, graceful_stop, read_pid, release_lock,
    write_pid,
)


def test_acquire_holds_our_pid(config):
    acquire_lock(config)
    assert config.lock_file.exists()
    assert int(config.lock_file.read_text().strip()) == os.getpid()
    release_lock(config)
    assert not config.lock_file.exists()


_HOLD_LOCK = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from core.config import Config\n"
    "from core.daemon import acquire_lock\n"
    "acquire_lock(Config(base_dir=Path(sys.argv[1])))\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def test_conflicting_live_holder_raises(config):
    # Another core process really holds the lock.
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(config.base_dir)],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(AlreadyRunningError) as exc:
            acquire_lock(config)
        assert exc.value.pid == holder.pid
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_lock_naming_reused_pid_is_cleared(config):
    # gabrielcamarate/Gabriel-OS#7: after a reboot the PID in a stale lock
    # belonged to an unrelated live process (proton.vpn.daemon) and the core
    # refused to start forever.
    other = subprocess.Popen(["sleep", "60"])
    try:
        config.lock_file.parent.mkdir(parents=True, exist_ok=True)
        config.lock_file.write_text(str(other.pid))
        acquire_lock(config)
        assert int(config.lock_file.read_text().strip()) == os.getpid()
        release_lock(config)
    finally:
        other.kill()
        other.wait(timeout=10)


def test_stale_lock_is_cleared_and_reacquired(config):
    p = subprocess.Popen(["true"])
    p.wait()  # pid is now dead
    config.lock_file.parent.mkdir(parents=True, exist_ok=True)
    config.lock_file.write_text(str(p.pid))
    acquire_lock(config)  # must clear the stale lock
    assert int(config.lock_file.read_text().strip()) == os.getpid()
    release_lock(config)


def _foreign_pid_files(config, pid):
    # Leftover core.lock/core.pid from a dead core whose PID now belongs to
    # an unrelated live process (gabrielcamarate/Gabriel-OS#8).
    config.lock_file.parent.mkdir(parents=True, exist_ok=True)
    config.lock_file.write_text(str(pid))
    write_pid(config, pid)


def test_stop_does_not_signal_reused_pid(config):
    other = subprocess.Popen(["sleep", "60"])
    try:
        _foreign_pid_files(config, other.pid)
        assert graceful_stop(config, timeout=1) is True
        assert other.poll() is None  # still alive: never signalled
        assert not config.lock_file.exists()
        assert not config.pid_file.exists()
    finally:
        other.kill()
        other.wait(timeout=10)


def test_status_and_daemon_ignore_reused_pid(config, monkeypatch):
    import core.cli as cli

    class Daemonized(Exception):
        pass

    def fake_daemonize(*a, **kw):
        raise Daemonized

    monkeypatch.setattr(cli, "_print_health", lambda config: None)
    monkeypatch.setattr(cli, "daemonize", fake_daemonize)
    other = subprocess.Popen(["sleep", "60"])
    try:
        _foreign_pid_files(config, other.pid)
        assert cli.cmd_status(None) == 1
        with pytest.raises(Daemonized):  # went on to start, not "Already running"
            cli.cmd_daemon(cli.build_parser().parse_args(["daemon"]))
    finally:
        other.kill()
        other.wait(timeout=10)


def test_stop_terminates_real_holder(config):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(config.base_dir)],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        assert graceful_stop(config, timeout=10) is True
        assert holder.wait(timeout=10) is not None
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_pid_write_read_roundtrip(config):
    write_pid(config, 4242)
    assert read_pid(config) == 4242
    release_lock(config)
    assert read_pid(config) is None


def test_restart_subcommand_has_no_migrations_flag():
    # Regression: `restart` parses an args namespace without
    # `no_migrations`; cmd_daemon used to crash *after* the double-fork
    # (AttributeError, stderr on /dev/null), so the new daemon never started.
    from core.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["restart"])
    assert getattr(args, "no_migrations", False) is False
    args = parser.parse_args(["restart", "--no-migrations"])
    assert args.no_migrations is True
    # daemon keeps its own flag as well
    assert parser.parse_args(["daemon", "--no-migrations"]).no_migrations is True


def test_daemon_and_restart_share_no_migrations_flag():
    # Regression: `restart` used to parse an args namespace without
    # `no_migrations`; cmd_daemon then crashed *after* the double-fork
    # (silent, stderr on /dev/null) and the new daemon never started.
    from core.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["daemon", "--no-migrations"]).no_migrations is True
    assert parser.parse_args(["restart", "--no-migrations"]).no_migrations is True
    assert getattr(parser.parse_args(["restart"]), "no_migrations", False) is False


def test_daemon_parent_verify_detects_live_and_dead(tmp_path):
    # L7: the first parent must not report a clean start when the grandchild
    # crashed.  ``_parent_verify_daemon`` returns True only when the daemon has
    # written a PID file whose process is still alive.
    import types
    from core.daemon import _parent_verify_daemon

    # no config -> legacy fire-and-forget behavior -> assume ok
    assert _parent_verify_daemon(None, 0.1) is True

    # live PID (our own) -> daemon is up
    live = tmp_path / "live.pid"
    live.write_text(str(os.getpid()))
    assert _parent_verify_daemon(types.SimpleNamespace(pid_file=live), 1.0) is True

    # dead PID -> the daemon crashed before/after writing its PID
    p = subprocess.Popen(["true"])
    p.wait()
    dead = tmp_path / "dead.pid"
    dead.write_text(str(p.pid))
    assert _parent_verify_daemon(types.SimpleNamespace(pid_file=dead), 0.3) is False

    # missing PID file (crashed before writing it) -> not up
    assert _parent_verify_daemon(
        types.SimpleNamespace(pid_file=tmp_path / "missing.pid"), 0.3) is False
