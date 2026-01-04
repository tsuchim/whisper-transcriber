from __future__ import annotations

from typing import Optional

import numpy as np


def merge_ranges_ms(ranges: list[tuple[int, int]], gap_ms: int = 100) -> list[tuple[int, int]]:
    """(start_ms, end_ms) の範囲リストをソートしてマージする。"""
    if not ranges:
        return []

    ranges_sorted = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = ranges_sorted[0]
    for s, e in ranges_sorted[1:]:
        if s <= cur_e + gap_ms:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def ranges_total_ms(ranges: list[tuple[int, int]]) -> int:
    """(start_ms, end_ms) の総長(ms)を計算する（範囲は重ならない前提）。"""
    return sum(max(0, int(e) - int(s)) for s, e in ranges)


def ranges_intersection_ms(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """2つの (start_ms,end_ms) 範囲の交差長(ms)を計算する。"""
    if not a or not b:
        return 0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    i = 0
    j = 0
    total = 0
    while i < len(a_sorted) and j < len(b_sorted):
        a_s, a_e = a_sorted[i]
        b_s, b_e = b_sorted[j]
        s = max(a_s, b_s)
        e = min(a_e, b_e)
        if e > s:
            total += e - s
        if a_e <= b_e:
            i += 1
        else:
            j += 1
    return total


def webrtcvad_ranges_ms(
    audio: np.ndarray,
    sr: int,
    webrtcvad_module,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    start_ratio: float = 0.9,
    end_ratio: float = 0.9,
) -> list[tuple[int, int]]:
    """WebRTC VAD で音声区間を検出（軽量・高速）。

    Returns:
        (start_ms, end_ms) のローカル範囲（chunk先頭=0）
    """
    if sr not in {8000, 16000, 32000, 48000}:
        raise ValueError(f"webrtcvad unsupported sample rate: {sr}")
    if frame_ms not in {10, 20, 30}:
        raise ValueError(f"webrtcvad unsupported frame_ms: {frame_ms}")

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    # float32 [-1,1] → int16 PCM
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    bytes_per_frame = int(sr * frame_ms / 1000) * 2
    if bytes_per_frame <= 0:
        return []

    pcm_bytes = pcm.tobytes()
    n_frames = len(pcm_bytes) // bytes_per_frame
    if n_frames <= 0:
        return []

    vad = webrtcvad_module.Vad(int(aggressiveness))
    padding_frames = max(1, int(round(padding_ms / frame_ms)))
    start_trigger = int(np.ceil(padding_frames * start_ratio))
    end_trigger = int(np.ceil(padding_frames * end_ratio))

    ring: list[bool] = []
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start_frame = 0

    for i in range(n_frames):
        frame = pcm_bytes[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        is_speech = bool(vad.is_speech(frame, sr))

        ring.append(is_speech)
        if len(ring) > padding_frames:
            ring.pop(0)

        if not in_segment:
            if len(ring) == padding_frames and sum(ring) >= start_trigger:
                in_segment = True
                seg_start_frame = max(0, i - padding_frames + 1)
                ring.clear()
        else:
            if len(ring) == padding_frames and (padding_frames - sum(ring)) >= end_trigger:
                seg_end_frame = max(seg_start_frame, i - padding_frames + 1)
                start_ms = int(seg_start_frame * frame_ms)
                end_ms = int(seg_end_frame * frame_ms)
                if end_ms > start_ms:
                    segments.append((start_ms, end_ms))
                in_segment = False
                ring.clear()

    if in_segment:
        start_ms = int(seg_start_frame * frame_ms)
        end_ms = int(n_frames * frame_ms)
        if end_ms > start_ms:
            segments.append((start_ms, end_ms))

    return segments


def webrtcvad_calibrate_aggressiveness(
    audio: np.ndarray,
    sr: int,
    webrtcvad_module,
    cap_sec: int = 120,
) -> int:
    """WebRTC VAD aggressiveness を簡易的に自動推定する。

    1/2/3 の結果の整合性（IoU）から最も安定していそうな値を選ぶ。
    """
    max_samples = int(sr * max(1, cap_sec))
    sample = audio[: min(int(audio.shape[0]), max_samples)]
    if sample.size <= 0:
        return 2

    results: dict[int, list[tuple[int, int]]] = {}
    for aggr in (1, 2, 3):
        results[aggr] = webrtcvad_ranges_ms(sample, sr=sr, webrtcvad_module=webrtcvad_module, aggressiveness=aggr)

    scores: dict[int, float] = {}
    for aggr in (1, 2, 3):
        ratios: list[float] = []
        for other in (1, 2, 3):
            if other == aggr:
                continue
            inter = ranges_intersection_ms(results[aggr], results[other])
            union = ranges_total_ms(results[aggr]) + ranges_total_ms(results[other]) - inter
            ratios.append(1.0 if union <= 0 else inter / union)
        scores[aggr] = sum(ratios) / max(1, len(ratios))

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    # タイは中庸(2)を優先
    if scores.get(2, 0.0) == scores.get(best, 0.0):
        return 2
    return int(best)


def webrtcvad_in_chunks(
    audio: np.ndarray,
    sr: int,
    webrtcvad_module,
    chunk_count: int = 5,
    overlap_ms: int = 300,
    aggressiveness: Optional[int] = None,
) -> list[tuple[int, int]]:
    """WebRTC VAD を分割実行し、範囲を全体msに正規化して返す。"""
    chosen_aggr = aggressiveness if aggressiveness is not None else webrtcvad_calibrate_aggressiveness(audio, sr=sr, webrtcvad_module=webrtcvad_module)

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    if chunk_count <= 1 or total_samples < sr * 60:
        chunk_count = 1

    overlap_samples = int(overlap_ms * sr / 1000)
    chunk_len = int(np.ceil(total_samples / chunk_count))

    all_ranges: list[tuple[int, int]] = []
    for i in range(chunk_count):
        base_start = i * chunk_len
        base_end = min(total_samples, (i + 1) * chunk_len)
        if base_start >= base_end:
            break

        chunk_start = max(0, base_start - overlap_samples)
        chunk_end = min(total_samples, base_end + overlap_samples)
        chunk_audio = audio[chunk_start:chunk_end]

        local = webrtcvad_ranges_ms(
            chunk_audio,
            sr=sr,
            webrtcvad_module=webrtcvad_module,
            aggressiveness=chosen_aggr,
            frame_ms=30,
            padding_ms=300,
        )
        offset_ms = int(chunk_start * 1000 / sr)
        for s_ms, e_ms in local:
            start_ms = int(s_ms) + offset_ms
            end_ms = int(e_ms) + offset_ms
            if end_ms > start_ms:
                all_ranges.append((start_ms, end_ms))

    return merge_ranges_ms(all_ranges, gap_ms=150)

