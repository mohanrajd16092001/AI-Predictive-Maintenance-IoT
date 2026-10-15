#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Setup complete. Next: .venv/bin/python main.py run'
