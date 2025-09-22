import pathlib
import warnings

import numpy as np
import pytest

import xtgeo
from tests.conftest import measure_peak_memory_usage
from xtgeo.common.log import functimer

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

SFILE1 = pathlib.Path("cubes/etc/ib_synth_iainb.segy")
SFILE2 = pathlib.Path("cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
SFILE3 = pathlib.Path("cubes/etc/cube_w_deadtraces.segy")

TOP2A = pathlib.Path("surfaces/reek/2/01_topreek_rota.gri")
TOP2B = pathlib.Path("surfaces/reek/2/04_basereek_rota.gri")

# ======================================================================================
# This is a a set of tests towards a synthetic small cube made by I Bush in order to
# test all attributes in detail
# ======================================================================================


@pytest.fixture(name="loadsfile1")
def fixture_loadsfile1(testdata_path):
    """Fixture for loading a SFILE1"""
    logger.info("Load seismic file 1")
    return xtgeo.cube_from_file(testdata_path / SFILE1)


@pytest.fixture(name="loadsfile2")
def fixture_loadsfile2(testdata_path):
    """Fixture for loading a SFILE2"""
    logger.info("Load seismic file 2")
    return xtgeo.cube_from_file(testdata_path / SFILE2)


@pytest.fixture(name="loadsfile3")
def fixture_loadsfile3(testdata_path):
    """Fixture for loading a SFILE3"""
    logger.info("Load seismic file 3")
    return xtgeo.cube_from_file(testdata_path / SFILE3)


@pytest.mark.bigtest
def test_large_cube_memory_usage():
    """Get attribute around a constant cube slices and check peak memory."""
    import gc

    # Cube size: 500 * 600 * 700 * 4 bytes ~= 840 MB
    mycube = xtgeo.Cube(
        ncol=500, nrow=600, nlay=700, xinc=12, yinc=12, zinc=4, values=666.0
    )

    level1 = 10
    level2 = 1500

    @measure_peak_memory_usage
    @functimer(output="print")
    def compute_with_mem_tracking(cube, alg, interp):
        """Helper function to be decorated."""
        return cube.compute_attributes_in_window(
            level1,
            level2,
            algorithm=alg,
            interpolation=interp,
        )

    # Force garbage collection to get a cleaner baseline
    gc.collect()

    for algorithm in [1, 2]:
        for intp in ["linear", "cubic"]:
            print(f"\nTesting algorithm={algorithm}, interpolation={intp}")
            peak_mem_increase_bytes, result = compute_with_mem_tracking(
                mycube, algorithm, intp
            )

            del result
            gc.collect()  # Clean up before the next run

            peak_mem_increase_gb = peak_mem_increase_bytes / (1024 * 1024 * 1024)

            # The cube itself is ~840MB. The operation needs more.
            # Let's set a reasonable upper bound for this cube size.
            assert peak_mem_increase_gb > 0.001, (
                "Peak memory increase is unexpectedly low."
            )
            assert peak_mem_increase_gb < 25, "Memory usage increase exceeded threshold"


def test_various_attrs_new_algorithm_1_vs_2_linear(loadsfile2):
    """New algorithm, variant algorithm 1 and 2 with linear interpolation"""
    cube1 = loadsfile2
    surf1 = xtgeo.surface_from_cube(cube1, 1560)
    surf2 = xtgeo.surface_from_cube(cube1, 1760)

    @functimer(output="print")
    def get_result1():
        return cube1.compute_attributes_in_window(
            surf1,
            surf2,
            ndiv=10,
            algorithm=1,
            interpolation="linear",
        )

    @functimer(output="print")
    def get_result2():
        return cube1.compute_attributes_in_window(
            surf1,
            surf2,
            ndiv=10,
            algorithm=2,
            interpolation="linear",
        )

    result1 = get_result1()
    result2 = get_result2()

    for key in result1:
        assert np.allclose(
            result1[key].values,
            result2[key].values,
            atol=1e-9,
        ), f"Attribute {key} differs between algorithm 1 and 2"


def test_various_attrs_new_algorithm_1_vs_2_cubic(loadsfile2):
    """New algorithm, variant algorithm 1 and 2 now with cubic spline interpolation"""
    cube1 = loadsfile2
    surf1 = xtgeo.surface_from_cube(cube1, 1560)
    surf2 = xtgeo.surface_from_cube(cube1, 1760)

    @functimer(output="print")
    def get_result1():
        return cube1.compute_attributes_in_window(
            surf1,
            surf2,
            ndiv=10,
            algorithm=1,
            interpolation="cubic",
        )

    @functimer(output="print")
    def get_result2():
        return cube1.compute_attributes_in_window(
            surf1,
            surf2,
            ndiv=10,
            algorithm=2,
            interpolation="cubic",
        )

    result1 = get_result1()
    result2 = get_result2()

    for key in result1:
        if key not in ["meanpos", "meanneg"]:
            assert np.allclose(
                result1[key].values,
                result2[key].values,
                atol=1e-6,
            ), f"Attribute {key} differs between algorithm 1 and 2"
        else:
            # meanpos and meanneg are very sensititive to slight numerical differences
            # between scipy's interpolation and the in-house custom cubic interpolation,
            # so tests are more relaxed.
            assert result1[key].values.mean() == pytest.approx(
                result2[key].values.mean(), rel=1e-2
            ), f"Attribute {key} mean differs between algorithm 1 and 2"
            assert result1[key].values.std() == pytest.approx(
                result2[key].values.std(),
                rel=1e-1,
            ), f"Attribute {key} std differs between algorithm 1 and 2"
