# Sourced by the launchers: repository root as cwd, variables from .env, $PY, absolute $EXP_ROOT.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; [ -f .env ] && . ./.env; set +a
PY="${PYTHON:-python}"
EXP_ROOT="${EXP_ROOT:-outputs}"
case "$EXP_ROOT" in /*) ;; *) EXP_ROOT="$PWD/$EXP_ROOT" ;; esac
export EXP_ROOT
mkdir -p "$EXP_ROOT/logs"
backbone_of() { echo "$1" | grep -o "qwen8b\|llama8b\|ministral8b\|olmo7b\|gptoss20b\|r1qwen7b" | head -1; }
