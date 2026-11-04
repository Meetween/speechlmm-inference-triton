#!/usr/bin/env python3
"""serve_singularity.sh validation, with a fake singularity on PATH."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "serve_singularity.sh"

_SINGULARITY_SHIM = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SINGULARITY_ARGV_LOG}"
for arg in "$@"; do
  if [[ "${arg}" == "test" ]]; then
    exit 0
  fi
  if [[ "${arg}" == "tritonserver" || "${arg}" == "python3.12" ]]; then
    sleep 60
    exit 0
  fi
done
exit 0
"""

_CURL_SHIM = """#!/usr/bin/env bash
exit 0
"""


class ServeSingularityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.sif = self.root / "fake.sif"
        self.sif.write_text("sif")
        self.log = self.root / "singularity.log"
        bindir = self.root / "bin"
        bindir.mkdir()
        for name, body in (("singularity", _SINGULARITY_SHIM), ("curl", _CURL_SHIM)):
            path = bindir / name
            path.write_text(body)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{bindir}:{self.env['PATH']}"
        self.env["SINGULARITY_ARGV_LOG"] = str(self.log)
        self.env["SIF"] = str(self.sif)
        for name in list(self.env):
            if (
                name.endswith(
                    ("_MODEL_PATH", "_DEVICE_MAP", "_DISABLE_TALKER", "_DISABLE_CODE2WAV", "_CONTEXT_OVERFLOW")
                )
                or name in {
                    "DEVICE_MAP", "SPEECHLMM_SPEAKER", "TALKER_MAX_NEW_TOKENS", "TALKER_DO_SAMPLE",
                    "CONTEXT_OVERFLOW",
                }
            ):
                self.env.pop(name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run(self, extra: dict[str, str], timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
        env = {**self.env, **extra}
        return subprocess.run(
            ["bash", str(_SCRIPT)],
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _start(self, extra: dict[str, str]) -> subprocess.Popen[str]:
        env = {**self.env, **extra}
        return subprocess.Popen(
            ["bash", str(_SCRIPT)],
            cwd=str(_REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def _wait_log(self, needle: str, timeout: float = 5.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.log.exists():
                text = self.log.read_text()
                if needle in text:
                    return text
            time.sleep(0.05)
        self.fail(f"{needle!r} not in singularity log:\n{self.log.read_text() if self.log.exists() else ''}")

    def test_http_input_size_is_configurable(self) -> None:
        for value in (None, "1073741824"):
            extra = {"LOAD_MODEL": "qwen25_omni", "QWEN25_OMNI_MODEL_PATH": "Qwen/model"}
            if value is not None:
                extra["TRITON_HTTP_MAX_INPUT_SIZE"] = value
            proc = self._start(extra)
            try:
                log = self._wait_log("--load-model qwen25_omni")
                self.assertIn("--http-max-input-size " + (value or "268435456"), log)
            finally:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
                proc.stdout.close()
            self.log.unlink()

    def test_each_model_demands_its_own_variable(self) -> None:
        for model, variable in (
            ("qwen3_omni", "QWEN3_OMNI_MODEL_PATH"),
            ("speechlmm_v2_xl", "SPEECHLMM_V2_XL_MODEL_PATH"),
        ):
            with self.subTest(model=model):
                result = self._run({"LOAD_MODEL": model})
                self.assertEqual(result.returncode, 1)
                self.assertIn(variable, result.stderr)

    def test_one_model_does_not_require_the_others(self) -> None:
        proc = self._start(
            {"LOAD_MODEL": "qwen3_omni", "QWEN3_OMNI_MODEL_PATH": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}
        )
        try:
            log = self._wait_log("--load-model qwen3_omni")
            self.assertIn("--env QWEN3_OMNI_MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct", log)
            self.assertNotIn("test -d Qwen/", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_a_model_the_script_has_never_heard_of_is_forwarded(self) -> None:
        proc = self._start(
            {"LOAD_MODEL": "brand_new_omni", "BRAND_NEW_OMNI_MODEL_PATH": "/ckpt/new"}
        )
        try:
            log = self._wait_log("--load-model brand_new_omni")
            self.assertIn("--env BRAND_NEW_OMNI_MODEL_PATH=/ckpt/new", log)
            self.assertIn("test -d /ckpt/new", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_per_model_device_map_wins_over_the_global_one(self) -> None:
        proc = self._start(
            {
                "LOAD_MODEL": "speechlmm_v2_l qwen3_omni",
                "SPEECHLMM_V2_L_MODEL_PATH": "/ckpt/l",
                "QWEN3_OMNI_MODEL_PATH": "/ckpt/qwen3",
                "SPEECHLMM_V2_L_DEVICE_MAP": "balanced",
                "DEVICE_MAP": "auto",
            }
        )
        try:
            log = self._wait_log("--load-model qwen3_omni")
            self.assertIn("--env SPEECHLMM_V2_L_DEVICE_MAP=balanced", log)
            self.assertIn("--env QWEN3_OMNI_DEVICE_MAP=auto", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_context_overflow_defaults_to_warn_and_forwards_the_model_override(self) -> None:
        proc = self._start(
            {
                "LOAD_MODEL": "speechlmm_v2_l qwen3_omni",
                "SPEECHLMM_V2_L_MODEL_PATH": "/ckpt/l",
                "QWEN3_OMNI_MODEL_PATH": "/ckpt/qwen3",
                "SPEECHLMM_V2_L_CONTEXT_OVERFLOW": "reject",
            }
        )
        try:
            log = self._wait_log("--load-model qwen3_omni")
            self.assertIn("--env CONTEXT_OVERFLOW=warn", log)
            self.assertIn("--env SPEECHLMM_V2_L_CONTEXT_OVERFLOW=reject", log)
            self.assertNotIn("QWEN3_OMNI_CONTEXT_OVERFLOW", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_no_device_map_leaves_the_config_default(self) -> None:
        proc = self._start(
            {"LOAD_MODEL": "qwen3_omni", "QWEN3_OMNI_MODEL_PATH": "/ckpt/qwen3"}
        )
        try:
            log = self._wait_log("--load-model qwen3_omni")
            self.assertNotIn("QWEN3_OMNI_DEVICE_MAP", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_two_models_load_together(self) -> None:
        proc = self._start(
            {
                "LOAD_MODEL": "speechlmm_v2_xl qwen3_omni",
                "SPEECHLMM_V2_XL_MODEL_PATH": "/ckpt/xl",
                "QWEN3_OMNI_MODEL_PATH": "/ckpt/qwen3",
            }
        )
        try:
            log = self._wait_log("--load-model qwen3_omni")
            self.assertIn("--load-model speechlmm_v2_xl", log)
            self.assertIn("test -d /ckpt/qwen3", log)
            self.assertIn("test -d /ckpt/xl", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_sidecar_and_speaker_env_is_forwarded(self) -> None:
        proc = self._start(
            {
                "LOAD_MODEL": "qwen3_tts",
                "QWEN3_TTS_MODEL_PATH": "/ckpt/tts",
                "QWEN3_TTS_SITE_PACKAGES": "/deps/qwen3_tts_site",
                "QWEN3_TTS_SPEAKER": "ryan",
            }
        )
        try:
            log = self._wait_log("--load-model qwen3_tts")
            self.assertIn("--env QWEN3_TTS_SITE_PACKAGES=/deps/qwen3_tts_site", log)
            self.assertIn("--env QWEN3_TTS_SPEAKER=ryan", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_speech_env_is_forwarded(self) -> None:
        proc = self._start(
            {
                "LOAD_MODEL": "speechlmm_v2_l",
                "SPEECHLMM_V2_L_MODEL_PATH": "/ckpt/l",
                "SPEECHLMM_V2_L_DISABLE_TALKER": "0",
                "SPEECHLMM_V2_L_DISABLE_CODE2WAV": "0",
                "SPEECHLMM_SPEAKER": "chelsie",
            }
        )
        try:
            log = self._wait_log("--load-model speechlmm_v2_l")
            self.assertIn("--env SPEECHLMM_V2_L_DISABLE_TALKER=0", log)
            self.assertIn("--env SPEECHLMM_V2_L_DISABLE_CODE2WAV=0", log)
            self.assertIn("--env SPEECHLMM_SPEAKER=chelsie", log)
            self.assertIn("--env TALKER_MAX_NEW_TOKENS=4096", log)
            self.assertIn("--env TALKER_DO_SAMPLE=1", log)
        finally:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
