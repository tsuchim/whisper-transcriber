#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2025 tsuchim

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
from pathlib import Path
from typing import cast

import librosa
import noisereduce as nr  # type: ignore
import numpy as np

try:
    import cupy as cp  # type: ignore

    has_cupy = True  # type: ignore
    print("CuPy (GPU対応numpy) が利用可能です")
except ImportError:
    cp = np  # type: ignore
    has_cupy = False  # type: ignore
    print("CuPy が見つかりません。CPUでnumpyを使用します")
import torch
from pydub import AudioSegment  # type: ignore
from pydub.effects import compress_dynamic_range, normalize  # type: ignore
from pydub.silence import detect_nonsilent  # type: ignore
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "kotoba-tech/kotoba-whisper-v2.2"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True).to(device)  # type: ignore
processor = AutoProcessor.from_pretrained(model_id)  # type: ignore

print(f"GPU利用可能: {torch.cuda.is_available()}")
print(f"transformers device: {device}")
if has_cupy and torch.cuda.is_available():
    print(f"CuPy GPU acceleration: 有効")
else:
    print(f"CuPy GPU acceleration: 無効 (CPU処理)")

if len(sys.argv) < 2:
    print("使い方: python transcribe.py audio1.mp3 audio2.wav ...")
    sys.exit(1)


def transcribe_long_audio(audio_file: str):
    """長い音声を分割して処理し、結果を連結する"""
    print(f"\n=== {audio_file} ===")

    # 音声を読み込み
    audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore

    # 会議音声向け前処理パイプライン（動的レベル調整対応）
    print("音声を前処理中...")

    # 1. 適応的ノイズ除去（正規化前に実行）
    audio = nr.reduce_noise(y=audio, sr=16000, stationary=False, prop_decrease=0.3)  # pyright: ignore
    audio = cast(np.ndarray, audio)  # pyright: ignore

    # 2. 会議音声向け動的正規化（RMS正規化で音量差を保持）
    # L2正規化の代わりにRMS正規化を使用して、相対的な音量差を保持
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        # 目標RMSレベル: -20dBFS相当
        target_rms = 0.1  # 約-20dBFS
        audio = audio * (target_rms / rms)
        # クリッピング防止
        audio = np.clip(audio, -1.0, 1.0)
    audio = cast(np.ndarray, audio)  # pyright: ignore

    # 3. pydubでの処理のためにAudioSegmentに変換（GPU対応）
    if has_cupy and torch.cuda.is_available():
        # GPU処理
        audio_gpu = cp.asarray(audio)  # type: ignore
        audio_int16 = (audio_gpu * 32767).astype(cp.int16)  # type: ignore
        audio_int16_cpu = cp.asnumpy(audio_int16)  # type: ignore
    else:
        # CPU処理
        audio_int16_cpu = (audio * 32767).astype(np.int16)  # type: ignore

    audio_segment = AudioSegment(audio_int16_cpu.tobytes(), frame_rate=16000, sample_width=2, channels=1)  # type: ignore

    # 4. 会議音声向け適応的音量分析
    print("音声を分析中...")

    # 短時間窓での音量分析（500ms窓で細かく分析）
    window_size = 500  # 500ms窓
    audio_rms_chunks = []
    for i in range(0, len(audio_segment), window_size):
        chunk = audio_segment[i : i + window_size]
        if len(chunk) > 100 and chunk.dBFS > -float('inf'):  # type: ignore
            audio_rms_chunks.append(chunk.dBFS)  # type: ignore

    if audio_rms_chunks:
        min_db = min(audio_rms_chunks)
        max_db = max(audio_rms_chunks)
        avg_db = sum(audio_rms_chunks) / len(audio_rms_chunks)

        print(f"音声分析結果: 最小={min_db:.1f}dB, 最大={max_db:.1f}dB, 平均={avg_db:.1f}dB, 音量差={max_db-min_db:.1f}dB")

        # 会議音声向け適応的閾値：実際の無音部分を正確に検出
        volume_range = max_db - min_db
        if volume_range > 20:  # 音量差が20dB以上の場合（会議音声の特徴）
            # より厳格な無音検出：統計的アプローチで真の無音を検出
            # 音量の下位5%を無音候補として、そこから閾値を設定
            sorted_chunks = sorted(audio_rms_chunks)
            percentile_5 = sorted_chunks[max(0, len(sorted_chunks) // 20)]  # 下位5%
            percentile_15 = sorted_chunks[max(0, len(sorted_chunks) // 7)]  # 下位15%

            # より積極的な分割のため、閾値を緩める
            optimal_silence_thresh = max(-70, min(percentile_5 + 8, percentile_15))
            print(f"会議音声モード: 音量差{volume_range:.1f}dB、積極的分割({percentile_5:.1f}dB基準)")
            print(f"下位5%値: {percentile_5:.1f}dB, 下位15%値: {percentile_15:.1f}dB")
        else:
            # 通常の2σ法
            std_db = (sum([(x - avg_db) ** 2 for x in audio_rms_chunks]) / len(audio_rms_chunks)) ** 0.5
            optimal_silence_thresh = max(-60, avg_db - 2 * std_db)

        print(f"最適な無音閾値: {optimal_silence_thresh:.1f}dB")
    else:
        optimal_silence_thresh = -50  # デフォルト値
        print("音声分析失敗、デフォルト値-50dBを使用")

    print("無音検出で音声を分割中...")

    # 会議音声向け無音検出（積極的分割で長時間チャンク回避）
    if audio_rms_chunks:
        volume_range = max(audio_rms_chunks) - min(audio_rms_chunks)
        # より短い無音でも分割して、確実に処理可能なサイズにする
        min_silence_duration = 600 if volume_range > 20 else 400  # 600ms/400msの無音で分割
    else:
        min_silence_duration = 400

    # seek_stepを小さくして、より細かく無音を検出
    nonsilent_ranges = detect_nonsilent(audio_segment, min_silence_len=min_silence_duration, silence_thresh=optimal_silence_thresh, seek_step=25)  # type: ignore    print(f"検出された音声セグメント数: {len(nonsilent_ranges)}")  # type: ignore

    # 会議音声向け適応的音量調整
    if nonsilent_ranges:  # 音声セグメントが存在する場合のみ
        # 各セグメントの音量を分析
        segment_volumes = []
        for start_ms, end_ms in nonsilent_ranges:  # type: ignore
            segment = audio_segment[start_ms:end_ms]  # type: ignore
            if len(segment) > 100:  # type: ignore
                segment_volumes.append(segment.dBFS)  # type: ignore

        if segment_volumes:
            min_segment_db = min(segment_volumes)
            max_segment_db = max(segment_volumes)
            avg_segment_db = sum(segment_volumes) / len(segment_volumes)

            # 小さな音声セグメントがある場合の保護処理
            if min_segment_db < -35:  # 小さな音声が検出された場合
                # 軽微な圧縮を適用して小さな音声を持ち上げる
                audio_segment = compress_dynamic_range(audio_segment, threshold=-35.0, ratio=3.0, attack=5.0, release=50.0)  # type: ignore
                print(f"小さな音声検出: 軽微な圧縮を適用（{min_segment_db:.1f}dB → 改善）")

            # 全体的な音量調整（控えめに）
            current_db = audio_segment.dBFS  # type: ignore
            target_db = -18  # 会議音声向けやや高めの目標レベル
            if current_db < -25:  # 全体的に小さい場合のみ
                adjustment = min(4, target_db - current_db)  # 最大4dBまで
                audio_segment = audio_segment + adjustment  # type: ignore
                print(f"全体音量調整: {adjustment:.1f}dB 上昇（{current_db:.1f}dB → {audio_segment.dBFS:.1f}dB）")  # type: ignore

    # 無音でない部分を適切な長さにまとめる（積極的分割で処理性能向上）
    max_chunk_ms = 20000  # 20秒に短縮（より積極的に分割）
    chunks = []  # type: ignore

    # 冒頭の取りこぼし対策：音声開始から最初の無音区間前までを追加
    if nonsilent_ranges and nonsilent_ranges[0][0] > 1000:  # 最初のセグメントが1秒以降から始まる場合
        print(f"冒頭音声検出: 0s-{nonsilent_ranges[0][0]/1000:.1f}sを追加")
        processed_ranges = [(0, nonsilent_ranges[0][0])]  # 冒頭部分を追加
    else:
        processed_ranges = []

    # 20秒を超える単一セグメントは強制分割
    for start_ms, end_ms in nonsilent_ranges:  # type: ignore
        duration = end_ms - start_ms
        if duration > 20000:  # 20秒を超える場合
            print(f"長時間セグメント検出({duration/1000:.1f}s): 強制分割します")
            # 15秒ずつに分割（オーバーラップ有り）
            current_start = start_ms
            while current_start < end_ms:
                current_end = min(current_start + 15000, end_ms)
                if current_end - current_start >= 1000:  # 1秒以上のセグメントのみ
                    processed_ranges.append((current_start, current_end))
                current_start += 12000  # 3秒オーバーラップ
        else:
            processed_ranges.append((start_ms, end_ms))

    # 末尾の取りこぼし対策：最後のセグメント後から音声終了までを追加
    audio_length_ms = len(audio_segment)
    if nonsilent_ranges and nonsilent_ranges[-1][1] < audio_length_ms - 1000:  # 最後のセグメントと音声終了に1秒以上の差がある場合
        print(f"末尾音声検出: {nonsilent_ranges[-1][1]/1000:.1f}s-{audio_length_ms/1000:.1f}sを追加")
        processed_ranges.append((nonsilent_ranges[-1][1], audio_length_ms))

    # 分割されたセグメントをチャンクにまとめる
    current_chunk_start = None  # type: ignore
    current_chunk_end = None  # type: ignore

    for start_ms, end_ms in processed_ranges:  # type: ignore
        if current_chunk_start is None:
            current_chunk_start = start_ms  # type: ignore
            current_chunk_end = end_ms  # type: ignore
        else:
            # 現在のチャンクに追加できるかチェック
            if (end_ms - current_chunk_start) <= max_chunk_ms:  # type: ignore
                current_chunk_end = end_ms  # type: ignore
            else:
                # 現在のチャンクを完了して新しいチャンクを開始
                chunks.append((current_chunk_start, current_chunk_end))  # type: ignore
                current_chunk_start = start_ms  # type: ignore
                current_chunk_end = end_ms  # type: ignore

    # 最後のチャンクを追加
    if current_chunk_start is not None:
        chunks.append((current_chunk_start, current_chunk_end))  # type: ignore

    print(f"処理用チャンク数: {len(chunks)}")  # type: ignore

    transcriptions = []  # type: ignore

    for i, (start_ms, end_ms) in enumerate(chunks):  # type: ignore
        print(f"チャンク {i+1}/{len(chunks)} 処理中 ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)")  # type: ignore

        # チャンクを抽出
        chunk_segment = audio_segment[start_ms:end_ms]  # type: ignore

        # 短すぎるチャンクは処理しない（より短いチャンクも処理）
        if len(chunk_segment) < 200:  # type: ignore
            continue

        # numpy配列に変換（GPU対応）
        if has_cupy and torch.cuda.is_available():
            # GPU処理
            chunk_samples = cp.array(chunk_segment.get_array_of_samples(), dtype=cp.float32) / 32767.0  # type: ignore
            chunk_data = cp.asnumpy(chunk_samples)  # type: ignore
        else:
            # CPU処理
            chunk_data = np.array(chunk_segment.get_array_of_samples(), dtype=np.float32) / 32767.0  # type: ignore

        # 音声を処理
        inputs = processor(chunk_data, sampling_rate=16000, return_tensors="pt")  # type: ignore
        input_features = inputs.input_features.to(device).to(torch_dtype)  # type: ignore

        with torch.no_grad():
            generated_tokens = model.generate(  # pyright: ignore
                input_features,
                language="ja",
                task="transcribe",
                do_sample=False,
                temperature=0.0,
                no_repeat_ngram_size=0,
                use_cache=True,
                suppress_tokens=None,  # 抑制トークンを無効化
                num_beams=1,  # ビームサーチを無効化して感度向上
                repetition_penalty=1.0,  # 繰り返しペナルティを無効化
            )
            generated_tokens = cast(torch.Tensor, generated_tokens)  # pyright: ignore

        transcription: str = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]  # type: ignore

        if transcription.strip():  # type: ignore  # 空でない場合のみ追加
            transcriptions.append(transcription.strip())  # type: ignore
            print(f"結果: {transcription.strip()}")  # type: ignore

    # 全てのチャンクの結果を連結（改行区切り）
    full_transcription = "\n".join(transcriptions)  # type: ignore

    # テキストファイルに出力
    audio_path = Path(audio_file)
    output_file = audio_path.with_suffix('.txt')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_transcription)
        print(f"\n=== テキストファイルに保存完了 ===")
        print(f"出力先: {output_file}")
    except Exception as e:
        print(f"ファイル保存エラー: {e}")

    print(f"\n=== 完全な転写結果 ===")  # type: ignore
    print(full_transcription)  # type: ignore

    return full_transcription  # type: ignore


# 各音声ファイルを処理
for audio_file in sys.argv[1:]:
    try:
        transcribe_long_audio(audio_file)
    except Exception as e:  # type: ignore
        print(f"エラー: {audio_file} の処理中にエラーが発生しました: {e}")

"""
会議音声向け前処理パイプラインの実装根拠：

1. RMS正規化（音量差保持）：
   - L2正規化の代わりにRMS正規化を使用
   - 発話者間の相対的な音量差を保持しつつ、全体レベルを統一
   - 会議音声の特性（マイク距離、声の大きさの違い）に対応

2. 適応的無音検出閾値：
   - 音量差20dB以上で会議音声モードを自動判別
   - 小さな音声保護のため、最小音量から15%上を閾値に設定
   - 短い無音（300ms）も検出して発話の切れ目を正確に把握

3. 動的音量調整：
   - セグメント毎の音量分析で小さな発話を検出
   - -35dB以下の小さな音声に軽微な圧縮を適用
   - 全体調整は控えめ（最大4dB）で音声の自然さを保持

4. 会議音声の特徴に対応：
   - 500ms窓での細かい音量分析
   - 発話者交代、マイク距離変化、声量差への対応
   - 小さな相槌や質問も確実に認識

参考：
- 会議音声処理: IEEE音響・音声・信号処理学会標準
- 動的レンジ処理: AES（Audio Engineering Society）推奨事項
- Whisper論文: https://arxiv.org/abs/2212.04356
"""
