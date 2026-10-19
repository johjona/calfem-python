"""Tests for the Gmsh concrete slab example."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples import example_concrete_slab as slab


def test_slab_mesh_markers_and_boundary_conditions(tmp_path, monkeypatch):
    original_gmsh_mesh = slab.cfm.GmshMesh

    def gmsh_mesh_in_test_directory(geometry):
        mesh = original_gmsh_mesh(geometry)
        mesh.mesh_dir = str(tmp_path)
        return mesh

    monkeypatch.setattr(slab.cfm, "GmshMesh", gmsh_mesh_in_test_directory)

    coords, edof, _, bdofs = slab.build_slab_mesh()

    assert coords.shape == (221, 2)
    assert edof.shape == (192, 12)
    np.testing.assert_allclose(coords.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(coords.max(axis=0), [8.0, 5.0])
    assert sorted(bdofs) == [1, 2, 3, 4]
    assert [len(bdofs[marker]) for marker in range(1, 5)] == [51, 39, 51, 39]

    bc = np.array([], dtype=int)
    bc_val = np.array([], dtype=float)
    bc, bc_val = slab.cfu.applybc(bdofs, bc, bc_val, 1)
    bc, bc_val = slab.cfu.applybc3D(bdofs, bc, bc_val, 2, dimension=1)

    expected_bc = set(bdofs[1]) | set(bdofs[2][::3])
    assert set(bc) == expected_bc
    assert len(bc) == 63
    np.testing.assert_allclose(bc_val, 0.0)
