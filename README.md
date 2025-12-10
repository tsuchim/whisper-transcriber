# whisper-transcriber

## 概要

CUDA12.1 + Python3.12環境で動作する**高速日本語音声認識ツール**です。

- **モデル**: kotoba-whisper-v2.2（日本語特化）
- **前処理**: Silero VAD（ニューラルネット音声検出）で従来比100倍高速化
- **対応形式**: MP3, MP4, FLV, MKV, WAV, WebM等の音声・動画ファイル
- **GPU対応**: CUDA/CPU自動判定、CuPy GPU加速対応

### 処理時間の目安（NVIDIA GPU環境）

| 音声長 | 推定時間 | スピードアップ |
|--------|---------|--------------|
| 10分 | 10-20秒 | 約30-60倍 |
| 1時間 | 1-2分 | 約20-40倍 |
| 3時間 | 3-5分 | 約20-40倍 |

※旧版（noisereduce利用）比較：10分で3-5分かかっていたのが10-20秒に短縮

## 主な改善点（v2.0）

### 1. Silero VAD による高速音声検出

- **従来**: pydub（音量ベース）→ 遅い、音量差で見落とし
- **現在**: Silero VAD（ニューラルネット）→ **100倍高速**、小さな声も検出

実装:
```python
# 音量に関係なく「人の声」を検出
speech_timestamps = get_speech_timestamps(
    audio, 
    model,
    threshold=0.3,               # 高感度
    min_speech_duration_ms=100,  # 短い発話も検出
    speech_pad_ms=100            # マージン付与
)
```

### 2. ノイズ除去の削除

- **従来**: noisereduce (stationary=False) → 10-30分かかる
- **現在**: 削除 → **Whisper自体のノイズ耐性に依存**

Whisperはノイズの多い環境でも高精度のため、前処理でのノイズ除去は不要と判定。

### 3. ベクトル演算による高速化

- **従来**: Python forループで音量分析 → 1-3分
- **現在**: numpy/CuPy ベクトル演算 → **数秒**

## 使い方

### 基本

```bash
python whisper-v3.py audio.mp3
python whisper-v3.py video.mp4 video2.flv  # 複数ファイル対応
```

### オプション

| オプション | 短縮形 | 説明 |
|-----------|--------|------|
| `--prompt` | `-p` | AIに渡すプロンプト文字列（文脈情報、話者名など） |
| `--header` | `-H` | 出力ファイルの冒頭に付加するヘッダー（`\n` で改行） |
| `--output` | `-o` | 出力ファイルパス（単一ファイル処理時のみ有効） |

### 使用例

```bash
# 基本的な文字起こし
python whisper-v3.py audio.mp3

# プロンプトを指定（AIの精度向上）
python whisper-v3.py video.mp4 --prompt "配信者: 山田太郎。キーワード: ゲーム配信"

# ヘッダー付きで出力
python whisper-v3.py audio.flv --header "配信: 山田太郎\n日時: 2025-01-01 12:00"

# 出力ファイルを指定
python whisper-v3.py audio.flv --output /path/to/output.txt
```

## インストール手順

### 前提条件

- Python 3.12.x
- NVIDIA GPU（CUDA 12.1対応ドライバ）
  - CPU環境でも動作しますが、処理時間は大幅に長くなります
  - 推奨: VRAM 4GB以上（FP16モード）

### セットアップ

```bash
# 1. 仮想環境を作成（推奨）
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# 2. 依存パッケージをインストール
pip install -r requirements.txt
```

**注意**: PyTorch GPU版は `requirements.txt` の `--extra-index-url` から自動インストール。詳細は [PyTorch公式](https://pytorch.org/get-started/locally/) を参照。

## 前処理パイプラインの詳細

### 処理フロー

```
1. 音声読み込み (PyAV/ffmpeg/librosa)
   ↓
2. RMS正規化 (高速ベクトル演算)
   ↓
3. Silero VAD 音声区間検出 (ニューラルネット)
   ↓
4. セグメント毎の音量分析 (小さな声保護)
   ↓
5. チャンク分割 (Whisper処理用)
   ↓
6. 音声認識 (kotoba-whisper-v2.2)
```

### 各ステップの特徴

| ステップ | 時間 | 目的 |
|---------|------|------|
| **音声読み込み** | 〜10秒 | 16kHz/mono/PCMへの統一 |
| **RMS正規化** | 〜1秒 | 全体レベルを-20dBFSに統一 |
| **Silero VAD** | 〜10秒 | 発話区間を高速検出（音量非依存） |
| **音量分析** | 〜1秒 | 小さい発話（-35dB以下）を検出 |
| **チャンク分割** | 〜1秒 | 25秒以下の処理単位に分割 |
| **Whisper推論** | 大部分 | GPU/CPU処理 |

### 小さな声の処理

Silero VAD + 軽微なダイナミクス圧縮:

```python
# 小さな発話（-35dB以下）に軽微な圧縮を適用
if min_segment_db < -35:
    audio_segment = compress_dynamic_range(
        audio_segment, 
        threshold=-35.0, 
        ratio=3.0,        # 軽微な圧縮
        attack=5.0, 
        release=50.0
    )
```

## トラブルシューティング

### Silero VAD が見つからない場合

```
警告: Silero VAD が利用不可のため pydub を使用（低速）
```

**原因**: torch.hub が Silero VAD をダウンロードできていない

**対処法**:
```python
# 手動ダウンロード
import torch
torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
```

### GPU メモリ不足

**症状**: CUDA Out of Memory エラー

**対処法**:
1. 同時実行を避ける
2. CPU実行に切り替え（遅いが安定）
3. GPU メモリ解放: `nvidia-smi`で確認後、再起動

## 主要ファイル

- `whisper-v3.py` : 音声認識メインスクリプト
- `requirements.txt` : 依存パッケージリスト（GPU対応）
- `README.md` : このファイル
- `LICENSE` : Apache License 2.0

## ライセンスについて

本プロジェクトは Apache License 2.0 を採用しています。これは使用している主要ライブラリ（PyTorch、Transformers等）のライセンスとの互換性を考慮した選択です。

## 参考資料

- **Silero VAD**: https://github.com/snakers4/silero-vad
- **Whisper論文**: https://arxiv.org/abs/2212.04356
- **kotoba-whisper**: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2
- **PyTorch公式**: https://pytorch.org/get-started/locally/

## コーディングに使用したAI

- GitHub Copilot Chat（Claude Haiku 4.5）
- 設計・分析: Cursor/Claude Sonnet

## 更新履歴

### v2.0 (2025-12)

- **Silero VAD 導入**: 前処理を100倍高速化
- **noisereduce 削除**: Whisper自体のノイズ耐性に依存
- **ベクトル演算化**: 音量分析を高速化
- ドキュメント刷新

### v1.0 (2025-08)

- 初版リリース
- noisereduce + pydub による音声前処理

---

ご質問・不具合は GitHub Issues へどうぞ。
