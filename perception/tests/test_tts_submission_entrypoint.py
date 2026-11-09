"""Exercise environment overrides through the real perception entrypoint."""
from types import SimpleNamespace
import sys

import pytest

import main


@pytest.mark.parametrize("environment,cfg,expected", [
    ({"MCP_PORT": "21720", "WS_PORT": "21721"}, {"mcp_port": 15720, "ws_port": 15721}, (21720, 21721)),
    ({"MCP_PORT": "", "WS_PORT": ""}, {"mcp_port": 21820, "ws_port": 21821}, (21820, 21821)),
    ({}, {}, (15720, 15721)),
])
def test_main_binds_environment_ports_before_yaml_defaults(monkeypatch, environment, cfg, expected):
    for key in ("MCP_PORT", "WS_PORT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    observed = {}
    class Server:
        def __init__(self, address, handler):
            observed["mcp"] = address[1]
        def serve_forever(self):
            return
    class Thread:
        def __init__(self, target=None, args=(), **kwargs):
            if target is main._start_ws_thread:
                observed["ws"] = args[0]
        def start(self):
            return
    executor = SimpleNamespace(spin=lambda: None, shutdown=lambda: None)
    monkeypatch.setattr(main, "_load_config", lambda: cfg)
    monkeypatch.setattr(main.rclpy, "init", lambda: None)
    monkeypatch.setattr(main.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(main.rclpy.executors, "MultiThreadedExecutor", lambda: executor)
    monkeypatch.setattr(main, "PerceptionBundle", lambda *a: SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(main, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(main.threading, "Thread", Thread)
    monkeypatch.setattr(main, "_start_registration", lambda *a: None)
    monkeypatch.setattr(main.signal, "signal", lambda *a: None)
    main.main()
    assert (observed["mcp"], observed["ws"]) == expected


@pytest.mark.parametrize("selected", [None, "tts_xiaoyou04"])
def test_plugin_selection_keeps_default_and_uses_the_selected_module(monkeypatch, selected):
    class Plugin:
        PREFIX = "tts"
        def __init__(self, cfg, executor):
            self.cfg = cfg
    import plugins.tts
    monkeypatch.setattr(plugins.tts, "TTSPlugin", Plugin)
    monkeypatch.setitem(sys.modules, "plugins.tts_xiaoyou04", SimpleNamespace(TTSPlugin=Plugin))
    if selected is None:
        monkeypatch.delenv("TTS_PLUGIN", raising=False)
    else:
        monkeypatch.setenv("TTS_PLUGIN", selected)
    cfg = {"plugins": {"tts": {"enabled": True}, "echo_cancellation": {"enabled": False},
                       "audio_mixer": {"enabled": False}}}
    bundle = main.PerceptionBundle(cfg, None)
    assert len(bundle._plugins) == 1 and isinstance(bundle._plugins[0], Plugin)


def test_unavailable_selected_plugin_does_not_substitute_a_different_voice(monkeypatch):
    monkeypatch.setenv("TTS_PLUGIN", "missing_tts_plugin")
    cfg = {"plugins": {"tts": {"enabled": True}, "echo_cancellation": {"enabled": False},
                       "audio_mixer": {"enabled": False}}}
    bundle = main.PerceptionBundle(cfg, None)
    assert not bundle._plugins
