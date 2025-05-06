"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Fccus here on hexahedron functions
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.xyz import Point  # type: ignore
from xtgeo.common.log import null_logger

logger = null_logger(__name__)


@pytest.fixture
def hcorners_normal():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 1.0),
        Point(100.0, 0.0, 1.0),
        Point(100.0, 100.0, 1.0),
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_concave():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 1.0),
        Point(100.0, 0.0, 1.0),
        Point(10.0, 10.0, 1.0),  # Concave point
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),  # Concave point
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_thin():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 0.001),
        Point(100.0, 0.0, 0.001),
        Point(100.0, 100.0, 0.001),
        Point(0.0, 100.0, 0.001),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


def test_hexahedron_volume():
    """Test the hexahedron volume function, which returns a double from C++."""
    # Create a simple grid
    grid = xtgeo.create_box_grid((3, 4, 5))

    # Get the corners of the first cell
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(0, 0, 0)

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(1.0)


def test_hexahedron_volume_banal6_cell(testdata_path):
    """Test the hexahedron function using banal6 synth case"""
    # Read the banal6 grid
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")

    # Get the corners of a skew cell (2,1,2 in RMS using 1-based indexing)
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(1, 0, 1)

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(1093.75, rel=1e-3)  # 1093.75 is RMS' value

    # Get the corners of a another skew cell (4,1,2)
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(3, 0, 1)

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(468.75, rel=1e-3)  # 468.75 is RMS' value

    # some work on the corners
    corners_np = corners.to_numpy()
    assert corners_np.shape == (8, 3)
    assert corners_np.dtype == np.float64
    assert corners_np[0, 0] == corners.upper_sw.x
    assert corners_np[7, 2] == corners.lower_ne.z


THIN_RATIO = 1e-6  # this is typical a cell thickness 0.01 m and a cell size of 100 m


def test_is_hexahedron_thin(hcorners_normal, hcorners_thin):
    """Test the is_hexahedron_thin function."""

    # Create a regular hexahedron (not thin)
    assert not _internal.geometry.is_hexahedron_thin(hcorners_normal, THIN_RATIO)

    assert _internal.geometry.is_hexahedron_thin(hcorners_thin, THIN_RATIO)


def test_is_hexahedron_concave_projected(hcorners_normal, hcorners_concave):
    """Test the is_hexahedron_concave_projected function."""
    # Create a regular convex hexahedron
    assert not _internal.geometry.is_hexahedron_concave_projected(hcorners_normal)

    assert _internal.geometry.is_hexahedron_concave_projected(hcorners_concave)


@given(
    base_x=st.floats(0, 50),
    base_y=st.floats(0, 50),
    height=st.floats(0, 100),
    width=st.floats(1, 100),
    depth=st.floats(1, 100),
)
def test_hexahedron_volume_with_randomized_corners(
    base_x, base_y, height, width, depth
):
    """Test the hexahedron volume function with randomized but logical corners."""
    # Define corners based on the base position, width, depth, and height
    corners = [
        (base_x, base_y, 0.0),  # Lower SW
        (base_x + width, base_y, 0.0),  # Lower SE
        (base_x, base_y + depth, 0.0),  # Lower NW
        (base_x + width, base_y + depth, 0.0),  # Lower NE
        (base_x, base_y, height),  # Upper SW
        (base_x + width, base_y, height),  # Upper SE
        (base_x, base_y + depth, height),  # Upper NW
        (base_x + width, base_y + depth, height),  # Upper NE
    ]

    # Convert the list of tuples to CellCorners
    cell_corners = _internal.grid3d.CellCorners(*[Point(*corner) for corner in corners])

    # Calculate the volume
    volume = _internal.geometry.hexahedron_volume(cell_corners, 3)

    # Assert that the volume is non-negative
    assert volume >= 0, f"Volume should be non-negative, got {volume}"
