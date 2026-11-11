# Shared launcher initialization. Parse .env through rewind.config, exactly as Python does.
REWIND_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REWIND_REPO_ROOT" || return 1
# Bootstrap interpreter can be supplied by the caller; PYTHON from .env applies afterwards.
REWIND_EXPORTS="$("${PYTHON:-python}" -m rewind.config --shell)" || return 1
eval "$REWIND_EXPORTS"
unset REWIND_EXPORTS
PY="${PYTHON:-python}"
mkdir -p "$EXP_ROOT/logs"
