#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv-mujoco/bin/python"
default_xml="$project_dir/assert/mujoco_diana7_updated/mujoco_diana7_updated/scene_tactile.xml"
xml_path="${1:-$default_xml}"

if [[ ! -x "$python_bin" ]]; then
  echo "MuJoCo environment not found: $python_bin" >&2
  exit 1
fi

if [[ ! -f "$xml_path" ]]; then
  echo "MJCF file not found: $xml_path" >&2
  exit 1
fi

exec env PYGLFW_LIBRARY_VARIANT=x11 "$python_bin" -m mujoco.viewer --mjcf="$xml_path"
