"""CPU checks for the NPU kernel's configuration, memory and backend identity.

These tests do not compile or execute an Ascend kernel.
"""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def backend(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "torch_npu", "tilelang", "tvm"):
            raise AssertionError(f"Pure configuration imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    name = "_plain_attention_npu_cpu_contract"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/tilelang_attention_npu.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(builtins, "__import__", original)
    return module


def test_default_long_shape_and_bounded_scratch(backend):
    spec = backend.AttentionSpec()
    assert (spec.batch, spec.heads, spec.queries, spec.keys, spec.dim) == (1, 28, 37741, 37741, 128)
    assert (spec.block_m, spec.block_n) == (128, 128)
    assert spec.padded_queries == spec.padded_keys == 37760
    plan = spec.memory_plan(20)
    # Scratch depends on physical cores and tile sizes, never S-by-S storage.
    assert plan["workspace_bytes"] == 20 * (128 * 128 * (4 + 2) + 128 * 128 * 4)
    assert plan["original_input_bytes"] == 3 * 1 * 28 * 37741 * 128 * 2
    twice = backend.AttentionSpec(queries=2 * 37741, keys=2 * 37741)
    assert twice.memory_plan(20)["workspace_bytes"] == plan["workspace_bytes"]
    assert twice.memory_plan(20)["original_input_bytes"] == 2 * plan["original_input_bytes"]


@pytest.mark.parametrize("queries,keys", [(1, 1), (64, 64), (65, 129), (129, 65), (37741, 37741)])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("tile", [64, 128])
def test_padding_allocation_accounts_for_only_needed_copies(backend, queries, keys, dim, tile):
    spec = backend.AttentionSpec(batch=2, heads=3, queries=queries, keys=keys, dim=dim,
                                 block_m=tile, block_n=tile)
    plan = spec.memory_plan(24)
    qp, kp = (queries + tile - 1) // tile * tile, (keys + tile - 1) // tile * tile
    qbytes = 2 * 2 * 3 * qp * dim if queries % tile else 0
    kvbytes = 2 * 2 * 2 * 3 * kp * dim if keys % tile else 0
    assert plan["input_padding_allocated_bytes"] == qbytes + kvbytes
    assert plan["padded_output_bytes"] == 2 * 2 * 3 * qp * dim
    assert plan["active_cube_cores"] <= min(24, 2 * 3 * ((queries + tile - 1) // tile))
    assert plan["prepared_allocated_bytes"] == qbytes + kvbytes + plan["padded_output_bytes"] + plan["workspace_bytes"]


@pytest.mark.parametrize("field,value", [("queries", 0), ("keys", -1), ("batch", True),
                                          ("heads", 1.5), ("dim", 80), ("block_m", 256),
                                          ("block_n", 32), ("dtype", "float32")])
def test_invalid_configuration_fails_before_any_device_import(backend, field, value):
    with pytest.raises(ValueError):
        backend.AttentionSpec(**{field: value})


@pytest.mark.parametrize("block_m,block_n", [(64, 128), (128, 64), (128, 256)])
def test_unreviewed_rectangular_tiles_are_rejected(backend, block_m, block_n):
    with pytest.raises(ValueError):
        backend.AttentionSpec(block_m=block_m, block_n=block_n)


@pytest.mark.parametrize("cores", [0, -1, True, 2.5])
def test_runtime_core_count_cannot_be_guessed_or_invalid(backend, cores):
    with pytest.raises(ValueError):
        backend.AttentionSpec().memory_plan(cores)


def test_small_shape_does_not_launch_more_cores_than_tasks(backend):
    spec = backend.AttentionSpec(heads=1, queries=1, keys=1)
    assert spec.query_tiles == 1
    assert spec.memory_plan(24)["active_cube_cores"] == 1


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("dim", [64, 128])
def test_supported_tiles_fit_local_memory_with_compiler_headroom(backend, tile, dim):
    plan = backend.AttentionSpec(dim=dim, block_m=tile, block_n=tile).local_memory_plan()
    # The pinned helper's largest reduction scratch is 4,672 bytes. Reserve
    # more space for compiler temporaries, row alignment and allocation padding.
    assert plan["ub_explicit_bytes_per_vector"] + 16 * 1024 <= plan["ub_capacity_bytes_per_vector"]
    assert plan["l1_explicit_bytes_per_cube"] <= plan["l1_capacity_bytes_per_cube"]
    assert plan["l0c_explicit_bytes_per_cube"] <= plan["l0c_capacity_bytes_per_cube"]


def fake_installed_backend(tmp_path, commit):
    package = tmp_path / "site-packages" / "tilelang"
    package.mkdir(parents=True)
    if commit is not None:
        (package / "VERSION").write_text(f"0.1.4+{commit}", encoding="utf-8")
    language = SimpleNamespace(**{name: object() for name in (
        "gemm_v0", "alloc_L1", "alloc_L0C", "alloc_ub", "Scope", "set_cross_flag", "wait_cross_flag")})
    return SimpleNamespace(__file__=str(package / "__init__.py"), __version__="test", language=language)


def test_source_built_wheel_must_carry_correct_commit(backend, tmp_path, monkeypatch):
    installed = fake_installed_backend(tmp_path, backend.SOURCE_COMMIT)
    monkeypatch.setitem(sys.modules, "tilelang", installed)
    evidence = backend._backend_provenance(installed)
    assert evidence["verified_commit"] == backend.SOURCE_COMMIT
    assert evidence["target"] == "ascendc"
    assert evidence["compiler_arch"] == "dav-2201"


@pytest.mark.parametrize("commit", [None, "0" * 40])
def test_unpinned_or_cuda_wheel_is_rejected(backend, tmp_path, commit):
    installed = fake_installed_backend(tmp_path, commit)
    with pytest.raises(RuntimeError, match="could not verify"):
        backend._backend_provenance(installed)


def test_matching_version_without_ascendc_language_is_rejected(backend, tmp_path, monkeypatch):
    installed = fake_installed_backend(tmp_path, backend.SOURCE_COMMIT)
    del installed.language.gemm_v0
    monkeypatch.setitem(sys.modules, "tilelang", installed)
    with pytest.raises(RuntimeError, match="AscendC"):
        backend._backend_provenance(installed)


def test_online_softmax_tail_math_has_independent_dense_reference(backend):
    result = backend.cpu_math_self_test()
    assert result["max_abs_error"] < 1e-14
    assert result["rows_checked"] == 26
    assert result["block_sizes"] == [64, 128]
    assert 257 in result["tail_key_lengths"]
    assert result["npu_compile_or_run_performed"] is False
