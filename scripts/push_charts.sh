#!/usr/bin/env bash
# Force-push the files in out/charts to the orphan branch "charts" as its only commit. Charts never go to main.
# Needs GITHUB_TOKEN and GITHUB_REPOSITORY (both present in Actions). Does nothing when there are no charts.
set -euo pipefail
src="${1:-out/charts}"
if [ ! -d "$src" ] || [ -z "$(ls -A "$src" 2>/dev/null)" ]; then
  echo "no charts to push"
  exit 0
fi
: "${GITHUB_TOKEN:?GITHUB_TOKEN is not set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
tmp="$(mktemp -d)"
cp -r "$src"/. "$tmp"/
cd "$tmp"
git init -q -b charts
git config user.name "weathervane-bot"
git config user.email "weathervane-bot@users.noreply.github.com"
git add -A
git commit -q -m "charts $(date -u +%Y-%m-%dT%H:%MZ)"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git push -q --force origin charts
echo "charts branch updated"
