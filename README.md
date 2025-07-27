# whisper-transcriber

## 概要
CUDA12.1 + Python3.12環境で動作する高速音声認識ツールです。
Whisper v3ベースでGPU/CPU自動判定、ノイズ除去、音声前処理に対応。

## インストール手順
1. Python 3.12.x環境を用意
2. CUDA 12.1対応GPUドライバをインストール
3. 仮想環境を作成（推奨: python -m venv .venv）
4. requirements.txtで依存パッケージを一括インストール
   ```
   pip install -r requirements.txt
   ```
   ※PyTorch GPU版は公式extra-index-url記法で自動インストール
   ※公式: https://pytorch.org/get-started/locally/

## 主要ファイル
- whisper-v3.py : 音声認識メインスクリプト
- requirements.txt : 依存パッケージリスト（GPU対応）
- .github/copilot-instructions.md : Copilot用カスタム指示
- LICENSE : Apache License 2.0

## ライセンスについて
本プロジェクトはApache License 2.0を採用しています。これは使用している主要ライブラリ（PyTorch、Transformers等）のライセンスとの互換性を考慮した選択です。
ライセンス上の問題やより適切な選択肢があれば、こっそり教えてください。

## 推奨設定
- .gitignore : Python/VSCode/OS向け公式推奨設定
- .editorconfig : 文字コード・改行コード統一（UTF-8 LF推奨）

## コーディングに使用したAI
- GitHub Copilot Chat: Agent
  - Claude Sonnet 4
  - GPT-4.1

## 参考
- PyTorch公式: https://pytorch.org/get-started/locally/
- GitHub公式: https://docs.github.com/en/get-started/quickstart/create-a-repo

---

ご質問・不具合はGitHub Issuesへどうぞ。
