# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import vllm.envs as envs
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    claim_triton_cache_dir,
    unique_filepath,
)


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_claim_triton_cache_dir_defaults_under_vllm_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    # setenv "" rather than delenv: the function writes os.environ directly,
    # and delenv records nothing to restore for an absent variable.
    monkeypatch.setenv("TRITON_CACHE_DIR", "")
    envs.disable_envs_cache()

    claim_triton_cache_dir()

    expected = tmp_path / "triton_cache"
    assert os.environ["TRITON_CACHE_DIR"] == str(expected)
    assert expected.is_dir()


def test_claim_triton_cache_dir_honours_caller(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "mine"))
    envs.disable_envs_cache()

    claim_triton_cache_dir()

    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "mine")
