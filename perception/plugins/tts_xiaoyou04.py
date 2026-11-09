"""Selected by TTS_PLUGIN=tts_xiaoyou04; retains the public `tts` tool contract."""
from __future__ import annotations

from copy import deepcopy
import threading

from plugins.tts import SherpaOnnxTTSPlugin, TOOLS
from plugins.xiaoyou04.runtime import Xiaoyou04Runtime


def _adapter(cfg, on_status=None):
    return Xiaoyou04Runtime(cfg, on_status=on_status)


class TTSPlugin(SherpaOnnxTTSPlugin):
    # The evaluator's TTS_PLUGIN is a tool name, while the product uses `tts`.
    # Bundle routing canonicalizes this alias before dispatch.
    ALIASES = ("tts_xiaoyou04",)

    def __init__(self, plugin_cfg, executor):
        self._lifecycle_lock = threading.RLock()
        # The default config.yaml names the public VITS2 model directory. A
        # custom plugin must not download its own release over that directory.
        cfg = dict(plugin_cfg)
        cfg["model_dir"] = "/models/xiaoyou04"
        cfg["device"] = "gpu"
        cfg["speaker_id"] = 0
        cfg["engine"] = "kokoro-xiaoyou04-zh-en"
        super().__init__(cfg, executor, adapter_factory=_adapter)

    def get_tools(self):
        tools = deepcopy(TOOLS)
        schema = tools[0]["configSchema"]
        # One voice and one engine; keep the operator's panel focused.
        schema["properties"] = {"speed": dict(schema["properties"]["speed"], minimum=.5, maximum=2)}
        inputs = tools[0]["inputSchema"]
        inputs["properties"].pop("reading_language", None)
        inputs["x-action-params"]["speak"] = {"params": ["text"], "required": ["text"],
                                                   "description": "Say this text in the fixed Xiaoyou 04 voice"}
        return tools

    def dispatch(self, name, args):
        if name in self.ALIASES:
            name = self.PREFIX
        action = args.get("action") if name == "tts" else name
        if action == "support_language":
            return {"engine": "kokoro-xiaoyou04-zh-en", "languages": ["zh", "en", "mixed"]}
        if action == "config":
            args = dict(args)
            # Darvin sends empty cloud-TTS fields even for an on-device model.
            # Empty placeholders are a no-op; nonempty cloud settings are errors.
            for key in ("api_key", "model", "voice", "url"):
                if key in args and args[key] in ("", None):
                    args.pop(key)
            unsupported = set(args) - {"action", "instance_id", "speed"}
            if unsupported:
                raise ValueError("Xiaoyou 04 supports only speed configuration; unexpected: "
                                 + ", ".join(sorted(unsupported)))
            if "speed" in args:
                value = float(args["speed"])
                if not .5 <= value <= 2:
                    raise ValueError("Xiaoyou 04 speed must be between 0.5 and 2")
        if action == "info":
            result = super().dispatch(name, args)
            result.update(engine="kokoro-xiaoyou04-zh-en", voice="04")
            adapter = self._adapter
            if adapter is not None:
                result["providers"] = adapter.providers
                result["release_checkpoint_sha256"] = adapter.release["source_checkpoint_sha256"]
                result["diagnostic_trial"] = bool(adapter.release.get("diagnostic_trial", False))
                result["release_model_sha256"] = adapter.release.get("files", {}).get("model.onnx", {}).get("sha256")
            return result
        with self._lifecycle_lock:
            return super().dispatch(name, args)

    def shutdown(self):
        with self._lifecycle_lock:
            super().dispatch("tts", {"action": "stop"})
            if self._adapter is not None:
                self._adapter.close()
                self._adapter = None
