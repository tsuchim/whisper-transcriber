import unittest

import numpy as np

import vad_utils


def _fake_webrtcvad_module(speech_frames: set[int]):
    class _Vad:
        def __init__(self, aggressiveness: int):
            self._i = 0
            self._aggr = aggressiveness

        def is_speech(self, frame: bytes, sr: int) -> bool:
            i = self._i
            self._i += 1
            return i in speech_frames

    class _Module:
        Vad = _Vad

    return _Module()


class TestVadUtils(unittest.TestCase):
    def test_webrtcvad_ranges_ms_invalid_sr(self) -> None:
        audio = np.zeros(1600, dtype=np.float32)
        fake = _fake_webrtcvad_module(set())
        with self.assertRaises(ValueError):
            vad_utils.webrtcvad_ranges_ms(audio, sr=44100, webrtcvad_module=fake)

    def test_webrtcvad_ranges_ms_invalid_frame_ms(self) -> None:
        audio = np.zeros(1600, dtype=np.float32)
        fake = _fake_webrtcvad_module(set())
        with self.assertRaises(ValueError):
            vad_utils.webrtcvad_ranges_ms(audio, sr=16000, webrtcvad_module=fake, frame_ms=25)

    def test_webrtcvad_ranges_ms_empty_audio(self) -> None:
        fake = _fake_webrtcvad_module(set())
        out = vad_utils.webrtcvad_ranges_ms(np.array([], dtype=np.float32), sr=16000, webrtcvad_module=fake)
        self.assertEqual(out, [])

    def test_webrtcvad_ranges_ms_detects_single_segment(self) -> None:
        # Use strict thresholds for deterministic behavior:
        # padding_frames=3 => start when 3/3 speech, end when 3/3 non-speech
        sr = 16000
        frame_ms = 10
        padding_ms = 30  # => 3 frames

        # 10 frames of audio => 100ms (content doesn't matter for this fake VAD)
        n_frames = 10
        n_samples = int(sr * (n_frames * frame_ms) / 1000)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Speech for frames [0..6], then silence for [7..9]
        fake = _fake_webrtcvad_module(set(range(0, 7)))
        out = vad_utils.webrtcvad_ranges_ms(
            audio,
            sr=sr,
            webrtcvad_module=fake,
            aggressiveness=2,
            frame_ms=frame_ms,
            padding_ms=padding_ms,
            start_ratio=1.0,
            end_ratio=1.0,
        )
        self.assertEqual(out, [(0, 70)])

    def test_merge_ranges_ms(self) -> None:
        self.assertEqual(vad_utils.merge_ranges_ms([(0, 100), (150, 200)], gap_ms=10), [(0, 100), (150, 200)])
        self.assertEqual(vad_utils.merge_ranges_ms([(0, 100), (105, 200)], gap_ms=10), [(0, 200)])


if __name__ == "__main__":
    unittest.main()

