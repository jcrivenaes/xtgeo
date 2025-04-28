"""Test the grid3d points inside function.

Here focus is on a Points array and the full 3D grid, not each cells
(cf test_grid3d_cell_point_inside.py).

"""

import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.xyz import Point  # type: ignore
from xtgeo.common.log import functimer


@pytest.fixture
def small_grid():
    """Create a 3x3x3 grid with simple geometry."""
    ncol, nrow, nlay = 3, 3, 3
    xinc, yinc, zinc = 100.0, 100.0, 50.0
    rotation = 0.0  # rotation to test non-aligned grid
    xori, yori, zori = 0.0, 0.0, 1000.0
    grid = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    return _internal.grid3d.Grid(grid)


@pytest.fixture(scope="module", name="drogon_grid")
def fixture_get_drogondata(testdata_path):
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/drogon/2/geogrid.roff")

    return (grid, _internal.grid3d.Grid(grid))


def test_points_inside_small_grid(small_grid):
    """Test the points_inside function with a small grid."""
    # Create a set of points to test
    points_input = [
        (-10, 2, 1010.0),  # outside the grid
        (2, 2, 1010.0),  # inside the grid in first cell (0, 0, 0)
        (102, 105, 1010.0),  # inside the grid in second cell in IJ (1, 1, 0)
        (299, 299, 1149.0),  # inside the grid in last cell (2, 2, 2)
        (2, 2, 1200.0),  # outside the grid in Z (0, 0, 2)
    ]
    points = xtgeo.Points(points_input)

    x_arr = points.get_dataframe(copy=False)["X_UTME"].to_numpy()
    y_arr = points.get_dataframe(copy=False)["Y_UTMN"].to_numpy()
    z_arr = points.get_dataframe(copy=False)["Z_TVDSS"].to_numpy()

    arr = np.array([x_arr, y_arr, z_arr]).T
    print(arr)

    points_cpp = _internal.xyz.PointSet(arr)

    iarr, jarr, karr = small_grid.get_indices_from_pointset(points_cpp)

    res = np.array([iarr, jarr, karr]).T
    print(res)

    np.testing.assert_array_equal(
        res,
        [[-1, -1, -1], [0, 0, 0], [1, 1, 0], [2, 2, 2], [-1, -1, -1]],
    )


def test_points_inside_drogon(drogon_grid):
    """Test the points_inside function using_drogon."""
    # Create a set of points to test
    grid, grid_cpp = drogon_grid

    x_from_grid, y_from_grid, z_from_grid = grid.get_xyz(asmasked=False)
    x_arr = x_from_grid.values[30, 50, 0:40]
    y_arr = y_from_grid.values[30, 50, 0:40]
    z_arr = z_from_grid.values[30, 50, 0:40]

    arr = np.array([x_arr, y_arr, z_arr]).T

    points_cpp = _internal.xyz.PointSet(arr)

    grid_one_cpp = grid_cpp.extract_onelayer_grid()  # same grid IJ but with one layer

    @functimer(output="print")
    def calc():  # inner function to compute effective time on this function
        return grid_cpp.get_indices_from_pointset(points_cpp, grid_one_cpp)

    iarr, jarr, karr = calc()

    res = np.array([iarr, jarr, karr]).T
    print(res)

    exp_i = np.ones_like(iarr) * 30
    exp_j = np.ones_like(jarr) * 50
    exp_k = np.arange(0, 40)
    expected = np.array([exp_i, exp_j, exp_k]).T

    np.testing.assert_array_equal(res, expected)


# assert result is not None
# assert len(result) == len(points)
