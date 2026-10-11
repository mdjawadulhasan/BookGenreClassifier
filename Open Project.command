#!/bin/zsh
set -eu
project_dir="${0:A:h}"
exec "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
    --new-window \
    --extensions-dir "$project_dir/.vscode-extensions" \
    --user-data-dir "$project_dir/.vscode-user-data" \
    "$project_dir"
