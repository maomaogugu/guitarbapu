#!/usr/bin/env sh

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir" || exit 1

if [ -x "$script_dir/.venv/bin/python" ]; then
  app_python="$script_dir/.venv/bin/python"
else
  app_python="python3"
fi

exec "$app_python" -m src.gui.app
