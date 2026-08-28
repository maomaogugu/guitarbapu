#!/bin/zsh

cd "$(dirname "$0")" || exit 1
if [[ -x "$PWD/.venv/bin/python" ]]; then
  app_python="$PWD/.venv/bin/python"
else
  app_python="python3"
fi

"$app_python" -m src.gui.app
app_status=$?

if [[ $app_status -ne 0 ]]; then
  echo
  echo "GuitarBapu 启动失败，请确认已在 .venv 或当前 Python 中安装 requirements.txt"
  read -r "?按回车键关闭窗口…"
fi

exit $app_status
