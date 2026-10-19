"""Regression tests for the rectangular plate element routines."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import calfem.core as cfc
import calfem.core_compat as cfc_compat


EX = [0.0, 2.0, 2.0, 0.0]
EY = [0.0, 0.0, 1.0, 1.0]
EP = [0.2]
D = cfc.hooke(1, 210e9, 0.3)
MODULES = [cfc, cfc_compat]


@pytest.mark.parametrize("module", MODULES)
def test_platre_stiffness_and_uniform_load(module):
    Ke = module.platre(EX, EY, EP, D)
    assert np.shape(Ke) == (12, 12)
    np.testing.assert_allclose(Ke, np.asarray(Ke).T, rtol=1e-13, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(Ke)[[0, 0, 0, 1, 2, 5, 11], [0, 1, 2, 1, 2, 11, 11]],
        [1504615384.615385, 649230769.2307694, -144615384.61538464,
         424615384.6153847, 160000000.00000006, 40000000.000000015,
         160000000.00000006],
        rtol=1e-13,
    )

    Keq, fe = module.platre(EX, EY, EP, D, [4.5])
    np.testing.assert_allclose(Keq, Ke)
    expected_fe = np.array(
        [2.25, 0.375, -0.75, 2.25, 0.375, 0.75,
         2.25, -0.375, 0.75, 2.25, -0.375, -0.75]
    ).reshape(12, 1)
    assert np.shape(fe) == (12, 1)
    np.testing.assert_allclose(fe, expected_fe)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "ex, ey, node_order, load_signs",
    [
        ([2, 0, 0, 2], [0, 0, 1, 1], [1, 2, 3, 0],
         [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1]),
        ([2, 0, 0, 2], [1, 1, 0, 0], [2, 3, 0, 1],
         [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1]),
        ([0, 2, 2, 0], [1, 1, 0, 0], [3, 0, 1, 2],
         [1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1]),
    ],
)
def test_platre_reversed_rectangle_node_orders(module, ex, ey, node_order, load_signs):
    Kbase = np.asarray(module.platre(EX, EY, EP, D))
    dofs = [3*node + dof for node in node_order for dof in range(3)]
    expected_K = Kbase[np.ix_(dofs, dofs)]
    Ke, fe = module.platre(ex, ey, EP, D, 4.5)
    np.testing.assert_allclose(Ke, expected_K)

    R1, R2, R3 = 2.25, 0.375, 0.75
    expected_fe = np.asarray(load_signs) * np.array(
        [R1, R2, R3, R1, R2, R3, R1, R2, R3, R1, R2, R3]
    )
    np.testing.assert_allclose(np.asarray(fe).reshape(-1), expected_fe)


@pytest.mark.parametrize("module", MODULES)
def test_platrs_returns_resultants_and_curvatures(module):
    Dplate = cfc.hooke(1, 30e9, 0.2)
    ep = [0.15]
    ed = np.array([
        0.001, 0.002, -0.001,
        0.003, -0.002, 0.004,
        -0.001, 0.005, 0.002,
        0.004, -0.003, 0.001,
    ])
    es, et = module.platrs([0, 8, 8, 0], [0, 0, 5, 5], ep, Dplate, ed)

    expected_es = np.array([[
        -2944.3359375, 1098.6328125, -3339.84375,
        1635.589599609375, -2707.03125,
    ]])
    expected_et = np.array([[0.000375, -0.0002, 0.00095]])
    assert es.shape == (1, 5)
    assert et.shape == (1, 3)
    np.testing.assert_allclose(es, expected_es, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(et, expected_et, rtol=1e-12, atol=1e-12)

    Db = ep[0]**3/12 * np.asarray(Dplate)
    np.testing.assert_allclose(et, -np.linalg.solve(Db, es[0]).reshape(1, 3))


@pytest.mark.parametrize("module", MODULES)
def test_platrs_zero_displacement_and_list_inputs(module):
    es, et = module.platrs(EX, EY, EP, D, [0.0]*12)
    assert es.shape == (1, 5)
    assert et.shape == (1, 3)
    np.testing.assert_allclose(es, 0.0)
    np.testing.assert_allclose(et, 0.0)


@pytest.mark.parametrize("module", MODULES)
def test_platrs_reversed_rectangle_node_orders(module):
    Dplate = cfc.hooke(1, 30e9, 0.2)
    ed = np.linspace(-0.01, 0.02, 12).reshape(4, 3)
    expected = module.platrs(EX, EY, EP, Dplate, ed.reshape(-1))
    cases = [
        ([2, 0, 0, 2], [0, 0, 1, 1], [1, 2, 3, 0]),
        ([2, 0, 0, 2], [1, 1, 0, 0], [2, 3, 0, 1]),
        ([0, 2, 2, 0], [1, 1, 0, 0], [3, 0, 1, 2]),
    ]
    for ex, ey, input_order in cases:
        es, et = module.platrs(ex, ey, EP, Dplate, ed[input_order].reshape(-1))
        np.testing.assert_allclose(es, expected[0])
        np.testing.assert_allclose(et, expected[1])


def test_core_and_compat_plate_outputs_match():
    ed = np.linspace(-0.01, 0.02, 12)
    core_results = cfc.platrs(EX, EY, EP, D, ed)
    compat_results = cfc_compat.platrs(EX, EY, EP, D, ed)
    np.testing.assert_allclose(core_results[0], compat_results[0])
    np.testing.assert_allclose(core_results[1], compat_results[1])
