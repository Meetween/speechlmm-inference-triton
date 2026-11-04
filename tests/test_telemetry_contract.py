#!/usr/bin/env python3
"""Static contract: a model shim and its config.pbtxt opt into telemetry together."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MODELS = _REPO / "models"
_REQUIRED_OUTPUTS = ("GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO", "METRICS")


def _shim_instruments(model_dir: Path) -> bool:
    source = (model_dir / "1" / "model.py").read_text()
    return "instrument(" in source


def _config_text(model_dir: Path) -> str:
    return (model_dir / "config.pbtxt").read_text()


def _declares_output(config: str, name: str) -> bool:
    return re.search(rf'name:\s*"{re.escape(name)}"', config) is not None


class TelemetryContractTests(unittest.TestCase):
    def test_shim_and_config_agree(self) -> None:
        for model_dir in sorted(path for path in _MODELS.iterdir() if path.is_dir()):
            shim = _shim_instruments(model_dir)
            config = _declares_output(_config_text(model_dir), "METRICS")
            self.assertEqual(
                shim,
                config,
                f"{model_dir.name}: shim instrument()={shim} METRICS output={config}",
            )

    def test_codec_frame_rate_is_numeric_when_present(self) -> None:
        for model_dir in sorted(path for path in _MODELS.iterdir() if path.is_dir()):
            config = _config_text(model_dir)
            match = re.search(
                r'key:\s*"codec_frame_rate"\s*value:\s*\{\s*string_value:\s*"([^"]+)"',
                config,
            )
            if match is None:
                continue
            float(match.group(1))

    def test_outputs_superset_of_contract(self) -> None:
        for model_dir in sorted(path for path in _MODELS.iterdir() if path.is_dir()):
            if not _shim_instruments(model_dir):
                continue
            config = _config_text(model_dir)
            for name in _REQUIRED_OUTPUTS:
                self.assertTrue(
                    _declares_output(config, name),
                    f"{model_dir.name} missing output {name}",
                )

    def test_shared_backends_subclass_base(self) -> None:
        for name in ("speechlmm_backend.py", "qwen_omni_backend.py"):
            source = (_REPO / "backends" / name).read_text()
            self.assertIn("class TritonPythonModel(BaseBackend)", source)

    def test_speech_models_declare_codec_rate(self) -> None:
        for name, rate in (
            ("qwen3_omni", "12.5"),
            ("speechlmm_v2_xl", "12.5"),
            ("qwen25_omni", "50"),
            ("speechlmm_v2_l", "50"),
        ):
            config = _config_text(_MODELS / name)
            self.assertIn('key: "codec_frame_rate"', config)
            self.assertIn(f'string_value: "{rate}"', config)
        for name in ("voxtral_mini_3b_2507",):
            config = _config_text(_MODELS / name)
            self.assertNotIn('key: "codec_frame_rate"', config)


if __name__ == "__main__":
    unittest.main()
