# Shared launcher initialization. Parse .env through trac.config, exactly as Python does.
TRAC_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAC_REPO_ROOT" || return 1
# Bootstrap interpreter can be supplied by the caller; PYTHON from .env applies afterwards.
TRAC_EXPORTS="$("${PYTHON:-python}" -m trac.config --shell)" || return 1
eval "$TRAC_EXPORTS"
unset TRAC_EXPORTS
PY="${PYTHON:-python}"
mkdir -p "$EXP_ROOT/logs"
