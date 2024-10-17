#!/bin/bash
OUTPUT_DIR="$(dirname "$0")/../output"
sudo rm -rf "$OUTPUT_DIR"/*
echo "🧹 🧼 Cleaned output directory at: $OUTPUT_DIR"
WANDB_LOGS_DIR="$(dirname "$0")/../morphs/wandb"
sudo rm -rf "$WANDB_LOGS_DIR"
echo "🧹 🧼 Removed wandb logs directory at: $WANDB_LOGS_DIR"