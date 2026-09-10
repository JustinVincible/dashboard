#!/usr/bin/env bash
# Installs jobdash as a macOS LaunchAgent that starts at login.
# Usage: ./install-service.sh [port]      (default port 8787)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8787}"
LABEL="com.jobdash"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ "$(uname)" = "Darwin" ] || { echo "launchd is macOS-only; run 'python app.py' directly."; exit 1; }
[ -x "$PROJECT_DIR/.venv/bin/python" ] || { echo "no .venv here — create it first (see README)."; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__PORT__|$PORT|g" \
    "$PROJECT_DIR/com.jobdash.plist.template" > "$PLIST"

# bootout first so a re-run replaces the running agent instead of erroring.
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST"

echo "installed $PLIST"
launchctl print "gui/$UID/$LABEL" | grep -E "state|pid|last exit" || true
echo "dashboard: http://localhost:$PORT"
