import sys
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

    # 高度な音声前処理
    print("音声を前処理中...")

    # 1. ノイズ除去（小さい音を保護）
    audio = nr.reduce_noise(y=audio, sr=16000, stationary=False, prop_decrease=0.6)  # pyright: ignore
    audio = cast(np.ndarray, audio)  # pyright: ignore

    # 2. pydubでの処理のためにAudioSegmentに変換（GPU対応）
    if has_cupy and torch.cuda.is_available():
        # GPU処理
        audio_gpu = cp.asarray(audio)  # type: ignore
        audio_int16 = (audio_gpu * 32767).astype(cp.int16)  # type: ignore
        audio_int16_cpu = cp.asnumpy(audio_int16)  # type: ignore
    else:
        # CPU処理
        audio_int16_cpu = (audio * 32767).astype(np.int16)  # type: ignore

    audio_segment = AudioSegment(audio_int16_cpu.tobytes(), frame_rate=16000, sample_width=2, channels=1)  # type: ignore

    # 3. 正規化（音量を最大化）
    audio_segment = normalize(audio_segment)  # type: ignore

    # 4. ダイナミックレンジ圧縮（小さい音を強調）- より強力に
    audio_segment = compress_dynamic_range(audio_segment, threshold=-30.0, ratio=6.0, attack=3.0, release=30.0)  # type: ignore

    # 5. 音量を大幅に上げる（小さい音をさらに強調）
    audio_segment = audio_segment + 12  # type: ignore

    # 6. 無音検出による賢い分割（小さい音に敏感に）
    print("無音検出で音声を分割中...")

    # 無音でない部分を検出（小さい音も拾うように感度向上）
    nonsilent_ranges = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50, seek_step=5)  # type: ignore

    print(f"検出された音声セグメント数: {len(nonsilent_ranges)}")  # type: ignore

    # 無音でない部分を適切な長さにまとめる
    max_chunk_ms = 15000  # type: ignore
    chunks = []  # type: ignore
    current_chunk_start = None  # type: ignore
    current_chunk_end = None  # type: ignore

    for start_ms, end_ms in nonsilent_ranges:  # type: ignore
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

    # 全てのチャンクの結果を連結
    full_transcription = " ".join(transcriptions)  # type: ignore

    print(f"\n=== 完全な転写結果 ===")  # type: ignore
    print(full_transcription)  # type: ignore

    return full_transcription  # type: ignore


# 各音声ファイルを処理
for audio_file in sys.argv[1:]:
    try:
        transcribe_long_audio(audio_file)
    except Exception as e:  # type: ignore
        print(f"エラー: {audio_file} の処理中にエラーが発生しました: {e}")
