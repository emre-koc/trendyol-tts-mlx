#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${MODEL_ID:-Trendyol/Trendyol-TTS}"
LOCAL_DIR="${LOCAL_DIR:-models/Trendyol-TTS}"
mkdir -p "$LOCAL_DIR"
hf download "$MODEL_ID" --local-dir "$LOCAL_DIR"
