#!/bin/bash
# One chat request to the one-node server. The model reasons before it answers; with the server's --reasoning-parser
# muse the reasoning lands in reasoning_content and content holds the answer. reasoning_strength low keeps the
# reasoning short (the chat template defaults to high).
set -euo pipefail

URL=${URL:-http://127.0.0.1:30000}
MODEL_PATH=${MODEL_PATH:-/models/Muse-Glimmer-30B}

curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 256,
  "chat_template_kwargs": {"reasoning_strength": "low"},
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
