#!/usr/bin/env bash
#
# ailoop.sh — 让 opencode 自主循环完成任务（自动批准权限，无需盯对话）
#
# 用法:
#   ./ailoop.sh "目标描述" [最大轮数(默认30)]
#
set -u

GOAL="${1:-}"
MAX_ROUNDS="${2:-30}"
LOG_DIR="loop-logs"
DONE_MARK="<GOAL_COMPLETE>"

if [ -z "$GOAL" ]; then
  echo "用法: $0 \"目标描述\" [最大轮数(默认30)]" >&2
  exit 1
fi

if ! command -v opencode >/dev/null 2>&1; then
  echo "错误: 未找到 opencode 命令" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

FIRST_PROMPT="你的目标：$GOAL

工作规则：
1. 自主规划、自主推进，不要向用户提问或等待确认。
2. 每轮完成一小步可验证的工作，确保代码始终可运行。
3. 每轮结束时评估：目标是否已完全达成？没有则说明还差什么。
4. 当且仅当目标完全达成时，在你的回复结尾输出 $DONE_MARK 。"

CONTINUE_PROMPT="继续朝目标推进。自主决定下一步并执行，不要等待确认。若目标已完全达成，请在回复结尾输出 $DONE_MARK 。"

list_sessions() {
  opencode session list 2>/dev/null | awk '{print $1}' | sort
}

commit_round() {
  local round="$1"
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    git add -A
    git commit -m "ailoop: round $round" >/dev/null
    echo "[ailoop] 已提交: round $round ($(git rev-parse --short HEAD))"
  else
    echo "[ailoop] 第 $round 轮无变更，跳过提交"
  fi
}

echo "=========================================="
echo " 目标: $GOAL"
echo " 最大轮数: $MAX_ROUNDS  |  日志: $LOG_DIR"
echo "=========================================="

SESSION_ID=""
BEFORE_SESSIONS="$(list_sessions)"

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  log="$LOG_DIR/round-$round.log"
  echo
  echo "[ailoop] ---- 第 $round / $MAX_ROUNDS 轮 ----"

  if [ "$round" -eq 1 ]; then
    opencode run --auto "$FIRST_PROMPT" 2>&1 | tee "$log"
    # 找出本轮新建的会话 ID，之后用 -s 精确续接，避免受其他会话干扰
    SESSION_ID="$(comm -13 <(echo "$BEFORE_SESSIONS") <(list_sessions) | grep '^ses_' | head -1)"
    [ -n "$SESSION_ID" ] && echo "[ailoop] 会话 ID: $SESSION_ID"
  elif [ -n "$SESSION_ID" ]; then
    opencode run --auto -s "$SESSION_ID" "$CONTINUE_PROMPT" 2>&1 | tee "$log"
  else
    opencode run --auto --continue "$CONTINUE_PROMPT" 2>&1 | tee "$log"
  fi

  commit_round "$round"

  if grep -qF "$DONE_MARK" "$log"; then
    echo
    echo "[ailoop] ✅ 检测到 $DONE_MARK ，目标已达成（共 $round 轮）"
    exit 0
  fi
done

echo
echo "[ailoop] ⚠️ 达到最大轮数 $MAX_ROUNDS ，强制停止。请检查 $LOG_DIR 与 git log。"
exit 1
