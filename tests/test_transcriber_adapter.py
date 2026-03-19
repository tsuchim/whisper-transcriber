import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


class TestTranscriberAdapter(TestCase):
    def test_transcribe_file_delegates_to_whisper_module(self) -> None:
        adapter_path = Path(__file__).resolve().parent.parent / "transcriber_adapter.py"
        spec = importlib.util.spec_from_file_location("transcriber_adapter_test", adapter_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        fake_whisper = type(
            "FakeWhisper",
            (),
            {
                "transcribe_long_audio": staticmethod(lambda **kwargs: Path(kwargs["output_path"]).write_text("ok", encoding="utf-8")),
                "offload_model": staticmethod(lambda: None),
                "restore_model": staticmethod(lambda: None),
                "set_logger": staticmethod(lambda logger: None),
            },
        )()

        with patch.object(module, "_load_whisper_module", return_value=fake_whisper):
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "out.txt"
                result = module.transcribe_file(
                    audio_file="input.wav",
                    output_path=str(output_path),
                    prompt="p",
                    header="h",
                    vad_profile="fast",
                )
                self.assertEqual(result, str(output_path))
                self.assertEqual(output_path.read_text(encoding="utf-8"), "ok")
