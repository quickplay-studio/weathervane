#!/usr/bin/env bash
# Commit the given paths and push to the current branch, rebasing and retrying when another run pushed first.
# usage: scripts/commit_push.sh "commit message" path [path ...]
set -euo pipefail
msg="$1"
shift
git config user.name "weathervane-bot"
git config user.email "weathervane-bot@users.noreply.github.com"
for p in "$@"; do
  [ -e "$p" ] && git add -- "$p"
done
if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi
git commit -q -m "$msg"
branch="${GITHUB_REF_NAME:-main}"
for attempt in 1 2 3; do
  if git pull --rebase origin "$branch" && git push origin "HEAD:${branch}"; then
    echo "pushed on attempt ${attempt}"
    exit 0
  fi
  echo "push attempt ${attempt} failed, retrying"
  git rebase --abort 2>/dev/null || true
  sleep $((attempt * 7))
done
echo "could not push after 3 attempts" >&2
exit 1
