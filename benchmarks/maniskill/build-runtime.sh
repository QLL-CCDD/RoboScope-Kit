#!/usr/bin/env bash
set -Eeuo pipefail

# Build the pinned ManiSkill multi-task runtime at any chosen install prefix.
# Needs Linux x86_64, NVIDIA GPU/driver, Vulkan loader, uv 0.11.14, curl, tar and patch.
# Rebuild at a new prefix to change location; an existing Python env is not relocatable.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || {
  echo "ManiSkill runtime builds require Linux x86_64" >&2; exit 3;
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
SOURCE_COMMIT="baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
SOURCE_SHA256="db7f1fb49038a1a11612876022ba31fdb8c9095e4e3f4a408728abf8b821f49e"
SOURCE_URL="https://codeload.github.com/mani-skill/ManiSkill/tar.gz/$SOURCE_COMMIT"
SOURCE_ROOT="$INSTALL_DIR/ManiSkill"
ENV_PREFIX="$INSTALL_DIR/env"
ENV_PYTHON="$ENV_PREFIX/bin/python"
RUNTIME_HOME="$INSTALL_DIR/home"
ASSET_ROOT="$INSTALL_DIR/assets"
PHYSX_VERSION="105.1-physx-5.3.1.patch0"
PHYSX_SHA256="4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"
PHYSX_PARENT="$RUNTIME_HOME/.sapien/physx/$PHYSX_VERSION"
PHYSX_LIBRARY="$PHYSX_PARENT/libPhysXGpu_64.so"
PHYSX_URL="https://github.com/sapien-sim/physx-precompiled/releases/download/$PHYSX_VERSION/linux-so.zip"
KIT_SOURCE="$(cd -- "$SCRIPT_DIR/../../packages/roboscope-kit" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
[[ "$("$UV_BIN" --version | awk '{print $2}')" == 0.11.14 ]] || {
  echo "uv 0.11.14 is required; set UV_BIN to its executable" >&2; exit 3;
}
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_NO_PROGRESS=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${MANISKILL_ASSET_DIR:-}" ]]; then
  install -d "$MANISKILL_ASSET_DIR"
  EXTERNAL_ASSET_ROOT="$(cd -- "$MANISKILL_ASSET_DIR" && pwd -P)"
  if [[ "$EXTERNAL_ASSET_ROOT" != "$ASSET_ROOT" ]]; then
    if [[ -e "$ASSET_ROOT" || -L "$ASSET_ROOT" ]]; then
      [[ "$(readlink -f -- "$ASSET_ROOT")" == "$EXTERNAL_ASSET_ROOT" ]] || {
        echo "existing runtime assets point to another directory; use a fresh prefix" >&2; exit 4;
      }
    else
      ln -s -- "$EXTERNAL_ASSET_ROOT" "$ASSET_ROOT"
    fi
  fi
else
  install -d "$ASSET_ROOT"
fi
export MS_ASSET_DIR="$ASSET_ROOT"

STAGING="$(mktemp -d "$INSTALL_DIR/.build.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT
ARCHIVE="${MANISKILL_SOURCE_ARCHIVE:-}"
if [[ -z "$ARCHIVE" ]]; then
  install -d "$INSTALL_DIR/downloads"
  ARCHIVE="$INSTALL_DIR/downloads/maniskill-$SOURCE_COMMIT.tar.gz"
  if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 --output "$ARCHIVE.part" "$SOURCE_URL"
    mv -- "$ARCHIVE.part" "$ARCHIVE"
  fi
fi
[[ -f "$ARCHIVE" ]] || { echo "source archive not found: $ARCHIVE" >&2; exit 4; }
[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || {
  echo "ManiSkill source archive checksum mismatch" >&2; exit 4;
}
if [[ ! -d "$SOURCE_ROOT" ]]; then
  install -d "$STAGING/source"
  tar -xzf "$ARCHIVE" --strip-components=1 -C "$STAGING/source"
  for path in setup.py mani_skill/envs/tasks/tabletop/pick_cube.py \
    mani_skill/agents/robots/panda/panda.py \
    mani_skill/utils/scene_builder/table/assets/table.glb; do
    [[ -f "$STAGING/source/$path" ]] || { echo "source is missing $path" >&2; exit 4; }
  done
  printf '%s\n' "$SOURCE_SHA256" > "$STAGING/source/.roboscope-source-sha256"
  mv -- "$STAGING/source" "$SOURCE_ROOT"
else
  [[ -f "$SOURCE_ROOT/.roboscope-source-sha256" ]] &&
    [[ "$(cat "$SOURCE_ROOT/.roboscope-source-sha256")" == "$SOURCE_SHA256" ]] || {
      echo "existing source has no matching provenance; use a fresh install directory" >&2; exit 4;
    }
fi
"$UV_BIN" python install 3.11.15
PYTHON_BIN="$("$UV_BIN" python find --managed-python 3.11.15)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
if [[ ! -x "$ENV_PYTHON" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_PREFIX"
fi
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
JACOBIAN_PATCH_VERSION="jacobian-cuda-graphs-v2"
JACOBIAN_PATCH="$SCRIPT_DIR/patches/jacobian-cuda-graphs.patch"
JACOBIAN_HELPER="$SCRIPT_DIR/patches/roboscope_jacobian.py"
JACOBIAN_PATCH_SHA256="1f2314e6031b4d8e9dfb6b2d97290942eac32acbdee686b6f6595915b657152b"
JACOBIAN_HELPER_SHA256="bf0bc707f11b1d6e1a3456b13dba07740b3e605c76fc9fbc0694fe507ea2080d"
KINEMATICS_ORIGINAL_SHA256="7f7a90a7c898e27a629618cbcc2b536d25feb9c8743c9e46b4882b8a39917e96"
KINEMATICS_PATCHED_SHA256="dbf091eba7774660f8246eca6fdd33458d6fe5961d23d0d2865d0918c85df862"
KINEMATICS_PATH="$SOURCE_ROOT/mani_skill/agents/controllers/utils/kinematics.py"
INSTALLED_HELPER="$SOURCE_ROOT/mani_skill/agents/controllers/utils/roboscope_jacobian.py"
for item in "$JACOBIAN_PATCH:$JACOBIAN_PATCH_SHA256" "$JACOBIAN_HELPER:$JACOBIAN_HELPER_SHA256"; do
  [[ "$(sha256sum "${item%:*}" | awk '{print $1}')" == "${item##*:}" ]] || {
    echo "Jacobian patch input checksum mismatch: ${item%:*}" >&2; exit 4;
  }
done
KINEMATICS_CURRENT_SHA256="$(sha256sum "$KINEMATICS_PATH" | awk '{print $1}')"
if [[ "$KINEMATICS_CURRENT_SHA256" == "$KINEMATICS_ORIGINAL_SHA256" ]]; then
  patch --batch --forward --directory "$SOURCE_ROOT" -p1 < "$JACOBIAN_PATCH"
elif [[ "$KINEMATICS_CURRENT_SHA256" != "$KINEMATICS_PATCHED_SHA256" ]]; then
  echo "ManiSkill kinematics source differs from the pinned original or patch; use a fresh prefix" >&2
  exit 4
fi
[[ "$(sha256sum "$KINEMATICS_PATH" | awk '{print $1}')" == "$KINEMATICS_PATCHED_SHA256" ]] || {
  echo "Patched ManiSkill kinematics checksum mismatch" >&2; exit 4;
}
if [[ -f "$INSTALLED_HELPER" ]]; then
  [[ "$(sha256sum "$INSTALLED_HELPER" | awk '{print $1}')" == "$JACOBIAN_HELPER_SHA256" ]] || {
    echo "Existing Jacobian helper differs from this patch; use a fresh prefix" >&2; exit 4;
  }
else
  install -m 0644 "$JACOBIAN_HELPER" "$INSTALLED_HELPER"
fi
"$ENV_PYTHON" -I - "$SOURCE_ROOT" "$JACOBIAN_PATCH_VERSION" "$JACOBIAN_PATCH_SHA256" \
  "$JACOBIAN_HELPER_SHA256" "$KINEMATICS_ORIGINAL_SHA256" "$KINEMATICS_PATCHED_SHA256" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1]) / ".roboscope-jacobian-patch.json"
expected = dict(zip(("patch_version", "patch_sha256", "helper_sha256", "original_kinematics_sha256",
                     "patched_kinematics_sha256"), sys.argv[2:], strict=True))
if marker.exists() and json.loads(marker.read_text()) != expected:
    raise RuntimeError("Existing source patch provenance differs; use a fresh prefix")
marker.write_text(json.dumps(expected, indent=2) + "\n")
PY
install -d "$PHYSX_PARENT"
if [[ ! -f "$PHYSX_LIBRARY" ]]; then
  if [[ -n "${MANISKILL_PHYSX_GPU_LIBRARY:-}" ]]; then
    [[ "$(sha256sum "$MANISKILL_PHYSX_GPU_LIBRARY" | awk '{print $1}')" == "$PHYSX_SHA256" ]] || {
      echo "supplied PhysX GPU library checksum mismatch" >&2; exit 4;
    }
    install -m 0644 "$MANISKILL_PHYSX_GPU_LIBRARY" "$PHYSX_LIBRARY"
  else
    install -d "$INSTALL_DIR/downloads"
    PHYSX_ARCHIVE="$INSTALL_DIR/downloads/physx-$PHYSX_VERSION-linux-so.zip"
    if [[ ! -f "$PHYSX_ARCHIVE" ]]; then
      curl --fail --location --retry 3 --output "$PHYSX_ARCHIVE.part" "$PHYSX_URL"
      mv -- "$PHYSX_ARCHIVE.part" "$PHYSX_ARCHIVE"
    fi
    "$ENV_PYTHON" - "$PHYSX_ARCHIVE" "$PHYSX_LIBRARY" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

with zipfile.ZipFile(sys.argv[1]) as archive:
    members = [name for name in archive.namelist() if Path(name).name == "libPhysXGpu_64.so"]
    if len(members) != 1:
        raise RuntimeError("PhysX release ZIP does not contain exactly one GPU library")
    with archive.open(members[0]) as source, open(sys.argv[2], "wb") as destination:
        shutil.copyfileobj(source, destination)
PY
  fi
fi
[[ "$(sha256sum "$PHYSX_LIBRARY" | awk '{print $1}')" == "$PHYSX_SHA256" ]] || {
  echo "PhysX GPU library checksum mismatch" >&2; exit 4;
}
"$UV_BIN" pip install --python "$ENV_PYTHON" --requirement "$SCRIPT_DIR/runtime-requirements.txt"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation "$SOURCE_ROOT"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation "$KIT_SOURCE"
"$UV_BIN" pip check --python "$ENV_PYTHON"
"$UV_BIN" pip freeze --python "$ENV_PYTHON" > "$INSTALL_DIR/runtime-requirements-resolved.txt"
install -m 0644 "$SCRIPT_DIR/benchmark-manifest.json" "$INSTALL_DIR/benchmark-manifest.json"
install -m 0755 "$SCRIPT_DIR/prepare-assets.py" "$INSTALL_DIR/prepare-assets.py"
IFS=',' read -r -a ASSET_SUITES <<< "${MANISKILL_ASSET_SUITES:-baseline_state_v1}"
IFS=',' read -r -a ASSET_TASKS <<< "${MANISKILL_ASSET_TASKS:-}"
ASSET_ARGUMENTS=()
if [[ -n "${MANISKILL_ASSET_ARCHIVE_DIR:-}" ]]; then
  ASSET_ARGUMENTS+=(--archive-dir "$MANISKILL_ASSET_ARCHIVE_DIR")
fi
for suite in "${ASSET_SUITES[@]}"; do
  [[ -n "$suite" ]] && ASSET_ARGUMENTS+=(--suite "$suite")
done
for task in "${ASSET_TASKS[@]}"; do
  [[ -n "$task" ]] && ASSET_ARGUMENTS+=(--task-id "$task")
done
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I "$INSTALL_DIR/prepare-assets.py" \
  --benchmark-manifest "$INSTALL_DIR/benchmark-manifest.json" --asset-dir "$ASSET_ROOT" \
  "${ASSET_ARGUMENTS[@]}" --download --output "$INSTALL_DIR/asset-provenance.json"
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I - "$INSTALL_DIR" "$SOURCE_COMMIT" "$SOURCE_SHA256" <<'PY'
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import mani_skill
import pytorch_kinematics
import roboscope_kit
import sapien
import sapien.physx
import torch

expected = {
    "mani_skill": "3.0.0b15", "sapien": "3.0.0b1", "gymnasium": "0.29.1",
    "torch": "2.6.0", "numpy": "1.26.4",
}
versions = {name: version(name) for name in expected}
for name, wanted in expected.items():
    if versions[name].split("+")[0] != wanted:
        raise RuntimeError(f"{name} is {versions[name]}, expected {wanted}")
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime build requires torch.cuda.is_available()")
if sapien.physx.version() != "105.1-physx-5.3.1.patch0":
    raise RuntimeError(f"Unexpected PhysX version: {sapien.physx.version()}")
root = Path(sys.argv[1])
patch_record = json.loads((root / "ManiSkill/.roboscope-jacobian-patch.json").read_text())
kinematics_root = Path(mani_skill.__file__).parent / "agents/controllers/utils"
for filename, field in (("kinematics.py", "patched_kinematics_sha256"),
                        ("roboscope_jacobian.py", "helper_sha256")):
    if hashlib.sha256((kinematics_root / filename).read_bytes()).hexdigest() != patch_record[field]:
        raise RuntimeError(f"Installed runtime patch differs: {filename}")
pk_root = Path(pytorch_kinematics.__file__).parent
pk_expected = {
    "chain.py": "1f8ef93febb8abda9ba9ec042d329e8e1e783f2002ad2028b051b8869d53d11d",
    "jacobian.py": "d45b196acec719f09716d825e1051d422e6361c9a4b494a54c6c194c256615c6",
    "ik.py": "42c37d80f33f8f40bfbd8327d978cfd237757ff7a994ce114d4e4980dc14bace",
}
for filename, expected_hash in pk_expected.items():
    if hashlib.sha256((pk_root / filename).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"pytorch-kinematics 0.7.4 source differs: {filename}")
patch_record.update(pytorch_kinematics_version=version("pytorch-kinematics"),
                    pytorch_kinematics_sources=pk_expected,
                    graph_switch_raw=os.environ.get("ROBOSCOPE_MANISKILL_JACOBIAN_GRAPHS"),
                    graph_switch_default="0")
records = []
asset_root = Path(mani_skill.__file__).parent
for relative in (
    "assets/robots/panda/panda_v2.urdf",
    "assets/robots/panda/panda_stick.urdf",
    "utils/scene_builder/table/assets/table.glb",
):
    path = asset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Bundled profile asset is missing: {path}")
    records.append({
        "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
(root / "build-provenance.json").write_text(json.dumps({
    "source_commit": sys.argv[2], "source_archive_sha256": sys.argv[3],
    "python": sys.version, "versions": versions, "kit_version": version("roboscope-kit"),
    "kit_module": roboscope_kit.__file__, "maniskill_module": mani_skill.__file__,
    "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "profile_asset_checksums": records,
    "physx_gpu_library": {
        "version": sapien.physx.version(),
        "relative_path": "home/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so",
        "sha256": hashlib.sha256((root / "home/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so").read_bytes()).hexdigest(),
    },
    "manifest_sha256": hashlib.sha256((root / "benchmark-manifest.json").read_bytes()).hexdigest(),
    "asset_provenance": "asset-provenance.json",
    "runtime_patches": [patch_record],
    "scope": "Manifest-driven ManiSkill state/RGB suites; representative runtime validation",
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(versions))
PY
SELF_TEST_OUTPUT="$INSTALL_DIR/self-test-$(date -u +%Y%m%dT%H%M%SZ)-$$"
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I -m roboscope_kit.runtime.benchmark --benchmark maniskill --self-test \
  --benchmark-manifest "$INSTALL_DIR/benchmark-manifest.json" --output-dir "$SELF_TEST_OUTPUT"
echo "Built ManiSkill $SOURCE_COMMIT in $INSTALL_DIR"
echo "Self-test evidence: $SELF_TEST_OUTPUT"
