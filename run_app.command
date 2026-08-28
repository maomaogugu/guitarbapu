#!/bin/zsh

cd "$(dirname "$0")" || exit 1
python3 -m src.gui.app
app_status=$?

if [[ $app_status -ne 0 ]]; then
  echo
  echo "GuitarBapu 启动失败，请确认已执行：python3 -m pip install -r requirements.txt"
  read -r "?按回车键关闭窗口…"
fi

exit $app_status
