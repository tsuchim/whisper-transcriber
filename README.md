# whisper-transcriber

`kotoba-whisper-v2.2` を **公式 pipeline ベース** で使うための文字起こしラッパです。

このリポジトリは plain ASR 専用ではありません。`v2.2` の価値である以下を有効にします。

- speaker diarization
- punctuation postprocessing

`whisper-v3.py` は後方互換の CLI 名として残し、実体は `kotoba_transcriber/` に分離しています。`live-scraping` など外部アプリケーションからはトップレベルの `transcriber_adapter.py` を参照してください。

## 動作方針

- モデル: `kotoba-tech/kotoba-whisper-v2.2`
- 話者分離: `pyannote/speaker-diarization-3.1`
- 日本語話者分離補助: `diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn`
- 句読点: `punctuators`
- 推論経路: Hugging Face `pipeline(task="kotoba-whisper", trust_remote_code=True)`

つまり、`AutoModelForSpeechSeq2Seq + generate()` を直接回す構成ではありません。

## 出力

単一入力 `sample.wav` に対して、以下を生成します。

- `sample.txt`
  - 話者ラベル付きの時系列テキスト
- `sample.transcript.json`
  - 正規化済みメタデータ
  - `text`
  - `segments`
  - `language`
  - `engine`
  - `model`
  - `speakers`

`sample.txt` の例:

```text
SPEAKER_00: 今日はよろしくお願いします。
SPEAKER_01: はい、お願いします。
SPEAKER_00: それでは始めます。
```

## セットアップ

前提:

- Python 3.12
- FFmpeg が PATH にあること
- Hugging Face にログインできること
- `pyannote/speaker-diarization-3.1` の利用規約を事前承諾していること

セットアップ:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
hf auth login
```

## 使い方

基本:

```bash
python3 whisper-v3.py audio.mp3
python3 whisper-v3.py video.mp4 another.flv
```

プロンプトや話者数制約を付ける例:

```bash
python3 whisper-v3.py meeting.mp3 \
  --prompt "配信者: 山田太郎。会議テーマ: 月次レビュー。" \
  --min-speakers 2 \
  --max-speakers 4
```

句読点を無効化する例:

```bash
python3 whisper-v3.py interview.wav --no-punctuation
```

単一ファイルの出力先を明示する例:

```bash
python3 whisper-v3.py audio.flv --output /path/to/output.txt
```

## オプション

- `--prompt`, `-p`: Whisper prompt
- `--header`, `-H`: テキスト先頭ヘッダー
- `--output`, `-o`: 出力 `.txt` パス
- `--chunk-length-s`: ASR chunk 長
- `--stride-length-s`: ASR stride 長
- `--batch-size`: pipeline batch size
- `--num-speakers`: 話者数固定
- `--min-speakers`: 最小話者数
- `--max-speakers`: 最大話者数
- `--add-silence-start`: 各話者区間の前に足す無音秒数
- `--add-silence-end`: 各話者区間の後に足す無音秒数
- `--no-punctuation`: 句読点後処理を無効化

## 外部アプリ連携

外部アプリケーションからは `transcriber_adapter.py` を読み込みます。

公開 API:

- `transcribe_file(audio_file, output_path, prompt=None, header=None, vad_profile=None, **kwargs)`
- `restore_model()`
- `offload_model()`
- `set_logger(logger)`

`vad_profile` は後方互換のため残していますが、この実装では使いません。

## 環境変数

必要に応じて以下で既定値を上書きできます。

- `KOTOBA_DEVICE`
- `KOTOBA_DEVICE_PYANNOTE`
- `KOTOBA_TORCH_DTYPE`
- `KOTOBA_ATTN_IMPLEMENTATION`
- `KOTOBA_BATCH_SIZE`
- `KOTOBA_CHUNK_LENGTH_S`
- `KOTOBA_STRIDE_LENGTH_S`
- `KOTOBA_ADD_PUNCTUATION`
- `KOTOBA_NUM_SPEAKERS`
- `KOTOBA_MIN_SPEAKERS`
- `KOTOBA_MAX_SPEAKERS`
- `KOTOBA_ADD_SILENCE_START`
- `KOTOBA_ADD_SILENCE_END`
- `KOTOBA_MODEL_PYANNOTE`
- `KOTOBA_MODEL_DIARIZERS`
- `HUGGINGFACE_HUB_TOKEN`

## 依存バージョン方針

- `transformers==4.48.0`、`huggingface-hub==0.27.1`、`pyannote.audio==3.3.2`、`diarizers` の固定コミットは、このリポジトリで検証した既知の動作組み合わせを再現するために固定しています。
- `huggingface-hub==0.27.1` は、このリポジトリ側の互換性固定です。`kotoba-whisper-v2.2` 自体の上流要件としてこの exact version を要求していることまでは確認していません。
- これらを更新する場合は、pipeline 初期化、Hugging Face 認証、pyannote の話者分離、句読点後処理までまとめて再確認してください。

### 認証 CLI に関する運用メモ

- 2026-03-21 時点で、利用環境によっては `huggingface-cli login` 実行時に `Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.` が表示されることを確認しています。
- 2026-03-21 の変更では、`huggingface-hub==0.27.1` を含む依存固定を導入しました。
- 同日、この警告への対応要望があったにもかかわらず、その時点では README と案内文の修正を行わず、警告が出る状態を残していました。
- 現在の README と案内文は `hf auth login` に更新しています。将来の CLI 仕様変更時は、README の手順、エラーメッセージ、および依存バージョン方針をまとめて見直してください。

## 運用上の注意

- 初回起動時はモデルダウンロードに時間がかかります。
- `pyannote` 系モデルは Hugging Face の認証と利用規約承諾が必要です。
- `v2.2` の話者分離と句読点を使うため、plain ASR より依存は重いです。
- モデル退避は GPU からの完全解放を優先し、`offload_model()` では pipeline オブジェクトを破棄します。

## 開発

```bash
python3 -m unittest discover -s tests
```

## 参考資料

- kotoba-whisper v2.2: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2
- pyannote speaker diarization: https://huggingface.co/pyannote/speaker-diarization-3.1
- PyTorch: https://pytorch.org/get-started/locally/
