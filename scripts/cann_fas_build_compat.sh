#!/usr/bin/env bash
# Source inside the build subshell only. The pinned v9.0.1 build.sh uses
# gawk solely to prefix log lines with timestamps; raw logs are sufficient.
if ! command -v gawk >/dev/null 2>&1; then
  echo "gawk not found: keeping official build logs without timestamp prefixes." >&2
  gawk() {
    # This is a narrow logging fallback, not a general GNU awk replacement.
    if [[ $# -ne 1 || "$1" != '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}' ]]; then
      echo "Unsupported gawk invocation in CANN build; logging fallback accepts only the pinned timestamp filter." >&2
      return 64
    fi
    command cat
  }
  # build.sh runs as a child Bash and invokes the filter in its pipeline.
  export -f gawk
fi
