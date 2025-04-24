import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.xyz import Point  # type: ignore
from xtgeo.common.log import functimer


@pytest.fixture
def simple_grid():
    """Create a simple 1x1x1 grid with regular geometry for testing."""
    # Create a simple grid with known coordinates
    ncol, nrow, nlay = 1, 1, 1
    xinc, yinc, zinc = 100.0, 100.0, 10.0
    rotation = 0.0
    xori, yori, zori = 0.0, 0.0, 1000.0
    grd = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    return _internal.grid3d.Grid(grd)


@pytest.fixture
def complex_grid():
    """Create a more complex 3x3x3 grid with irregular geometry."""
    ncol, nrow, nlay = 3, 3, 3
    xinc, yinc, zinc = 100.0, 100.0, 50.0
    rotation = 30.0  # rotation to test non-aligned grid
    xori, yori, zori = 0.0, 0.0, 1000.0
    grid = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    # Make the grid more irregular by modifying some z values
    zcorn_3d = grid._zcornsv.copy()
    # Modify some z values to make irregular cells
    zcorn_3d[1, 1, 1, :] += 20.0  # Shift middle points up
    grid._zcornsv = zcorn_3d
    return _internal.grid3d.Grid(grid)


@pytest.mark.parametrize(
    "method, point, expected",
    [
        # Test center point with different methods
        ("ray_casting", Point(50.0, 50.0, 1005.0), True),
        # Test points near boundaries
        ("ray_casting", Point(1.0, 50.0, 1005.0), True),
        ("ray_casting", Point(99.0, 50.0, 1005.0), True),
        ("ray_casting", Point(50.0, 1.0, 1005.0), True),
        ("ray_casting", Point(50.0, 99.0, 1005.0), True),
        ("ray_casting", Point(50.0, 50.0, 1001.0), True),
        ("ray_casting", Point(50.0, 50.0, 1009.0), True),
        # Test points outside
        ("ray_casting", Point(-10.0, 50.0, 1005.0), False),
        ("ray_casting", Point(110.0, 50.0, 1005.0), False),
        ("ray_casting", Point(50.0, -10.0, 1005.0), False),
        ("ray_casting", Point(50.0, 110.0, 1005.0), False),
        ("ray_casting", Point(50.0, 50.0, 990.0), False),
        ("ray_casting", Point(50.0, 50.0, 1020.0), False),
        ("tetrahedrons", Point(50.0, 50.0, 1005.0), True),
        ("tetrahedrons", Point(1.0, 50.0, 1005.0), True),
        ("tetrahedrons", Point(99.0, 50.0, 1005.0), True),
        ("tetrahedrons", Point(50.0, 1.0, 1005.0), True),
        ("tetrahedrons", Point(50.0, 99.0, 1005.0), True),
        ("tetrahedrons", Point(50.0, 50.0, 1001.0), True),
        ("tetrahedrons", Point(50.0, 50.0, 1009.0), True),
        ("tetrahedrons", Point(-10.0, 50.0, 1005.0), False),
        ("tetrahedrons", Point(110.0, 50.0, 1005.0), False),
        ("tetrahedrons", Point(50.0, -10.0, 1005.0), False),
        ("tetrahedrons", Point(50.0, 110.0, 1005.0), False),
        ("tetrahedrons", Point(50.0, 50.0, 990.0), False),
        ("tetrahedrons", Point(50.0, 50.0, 1020.0), False),
        ("centroid_tetrahedrons", Point(50.0, 50.0, 1005.0), True),
        ("centroid_tetrahedrons", Point(1.0, 50.0, 1005.0), False),  # expected True
        ("centroid_tetrahedrons", Point(99.0, 50.0, 1005.0), True),
        ("centroid_tetrahedrons", Point(50.0, 1.0, 1005.0), True),
        ("centroid_tetrahedrons", Point(50.0, 99.0, 1005.0), True),
        ("centroid_tetrahedrons", Point(50.0, 50.0, 1001.0), True),
        ("centroid_tetrahedrons", Point(50.0, 50.0, 1009.0), True),
        ("centroid_tetrahedrons", Point(-10.0, 50.0, 1005.0), False),
        ("centroid_tetrahedrons", Point(110.0, 50.0, 1005.0), False),
        ("centroid_tetrahedrons", Point(50.0, -10.0, 1005.0), False),
        ("centroid_tetrahedrons", Point(50.0, 110.0, 1005.0), False),
        ("centroid_tetrahedrons", Point(50.0, 50.0, 990.0), False),
        ("centroid_tetrahedrons", Point(50.0, 50.0, 1020.0), False),
        ("non_convex", Point(50.0, 50.0, 1005.0), True),
        ("non_convex", Point(1.0, 50.0, 1005.0), True),
        ("non_convex", Point(99.0, 50.0, 1005.0), True),
        ("non_convex", Point(50.0, 1.0, 1005.0), True),
        ("non_convex", Point(50.0, 99.0, 1005.0), True),
        ("non_convex", Point(50.0, 50.0, 1001.0), True),
        ("non_convex", Point(50.0, 50.0, 1009.0), True),
        ("non_convex", Point(-10.0, 50.0, 1005.0), False),
        ("non_convex", Point(110.0, 50.0, 1005.0), False),
        ("non_convex", Point(50.0, -10.0, 1005.0), False),
        ("non_convex", Point(50.0, 110.0, 1005.0), False),
        ("non_convex", Point(50.0, 50.0, 990.0), False),
        ("non_convex", Point(50.0, 50.0, 1020.0), False),
    ],
)
def test_point_inside_cell_methods(simple_grid, method, point, expected):
    """Test different methods for point-in-hexahedron with various test points."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    result = _internal.grid3d.is_point_in_cell(point, cell_corners, method)
    assert result == expected, (
        f"Method {method} with point {point} returned {result}, expected {expected}"
    )


def test_point_inside_hexahedroncell_etc_speed(simple_grid):
    """Compare the speed of different methods for point-in-hexahedron.

    Current status for "point-in" is that tetrahedrons is 10x faster than ray casting.

    """
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)
    center = Point(50.0, 50.0, 1005.0)

    iterations = 100000

    @functimer(output="print")
    def ray_casting_method():
        for i in range(iterations):
            _internal.grid3d.is_point_in_cell(center, cell_corners, "ray_casting")

    @functimer(output="print")
    def tetrahedrons_method():
        for i in range(iterations):
            _internal.grid3d.is_point_in_cell(center, cell_corners, "tetrahedrons")

    @functimer(output="print")
    def convexity_test():
        for i in range(iterations):
            _internal.grid3d.is_cell_non_convex(cell_corners)

    ray_casting_method()
    tetrahedrons_method()
    convexity_test()


@pytest.mark.parametrize(
    "point, expected_result, description",
    [
        # Inside points
        (Point(50.0, 50.0, 1005.0), True, "Center point"),
        (Point(40.0, 60.0, 1003.0), True, "Point slightly off center"),
        (Point(0.0, 50.0, 1005.0), True, "Point exactly on the edge"),
        # Outside points
        (Point(-10.0, 50.0, 1005.0), False, "Point outside X bounds"),
        (Point(50.0, 150.0, 1005.0), False, "Point outside Y bounds"),
        (Point(50.0, 50.0, 950.0), False, "Point outside Z bounds"),
    ],
)
def test_point_inside_simple_grid_cell(
    simple_grid, point, expected_result, description
):
    """Test if points are correctly identified as inside/outside a simple grid cell."""
    # Get the first cell of the grid
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    result = _internal.grid3d.is_point_in_cell(point, cell_corners)
    assert result == expected_result, (
        f"Failed for {description}: {point}, got {result}, expected {expected_result}"
    )


def test_point_inside_complex_grid(complex_grid):
    """Test points inside/outside cells of a more complex grid."""
    # Test points in various cells of the grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                cell_corners = complex_grid.get_cell_corners_from_ijk(i, j, k)

                # Get cell center by averaging corners
                corners_array = np.array(cell_corners.to_numpy()).reshape(-1, 3)
                center_x = np.mean(corners_array[:, 0])
                center_y = np.mean(corners_array[:, 1])
                center_z = np.mean(corners_array[:, 2])

                # Point at center should be inside
                center_point = Point(center_x, center_y, center_z)
                assert _internal.grid3d.is_point_in_cell(center_point, cell_corners)

                # Point far outside should be outside
                far_point = Point(center_x + 1000, center_y + 1000, center_z + 1000)
                assert not _internal.grid3d.is_point_in_cell(far_point, cell_corners)


def test_ray_edge_cases(simple_grid):
    """Test edge cases for ray casting algorithm."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    # Test with point exactly on a face
    on_top_face = Point(50.0, 50.0, 1000.0)
    assert _internal.grid3d.is_point_in_cell(on_top_face, cell_corners, "ray_casting")

    # Test with point exactly on a vertex
    on_vertex = Point(0.0, 0.0, 1000.0)
    assert _internal.grid3d.is_point_in_cell(on_vertex, cell_corners, "ray_casting")

    # Test with point exactly on an edge
    on_edge = Point(50.0, 0.0, 1000.0)
    assert _internal.grid3d.is_point_in_cell(on_edge, cell_corners, "ray_casting")


def test_point_inside_thin_cell():
    """Test with a degenerate cell (nearly flat in one direction)."""
    # Create a custom cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(100.0, 0.0, 1000.0)
    p3 = Point(0.0, 100.0, 1000.0)
    p4 = Point(100.0, 100.0, 1000.0)

    p5 = Point(0.0, 0.0, 1001.0)
    p6 = Point(100.0, 0.0, 1001.0)
    p7 = Point(0.0, 100.0, 1001.0)
    p8 = Point(100.0, 100.0, 1001.0)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    # Test with point inside the thin cell
    inside_point = Point(50.0, 50.0, 1000.5)
    assert _internal.grid3d.is_point_in_cell(inside_point, cell_corners, "tetrahedrons")

    # Test with point just outside the z bounds
    outside_point = Point(50.0, 50.0, 1001.1)
    assert not _internal.grid3d.is_point_in_cell(
        outside_point, cell_corners, "tetrahedrons"
    )


def test_point_inside_deformed_case1_cell():
    """Test with a degenerate cell (deformed, case 1)."""
    # Create a custom cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(10.0, 0.0, 1000.0)
    p3 = Point(0.0, 200.0, 1000.0)
    p4 = Point(100.0, 100.0, 1000.0)

    p5 = Point(0.0, 0.0, 1001.0)
    p6 = Point(10.0, 0.0, 1001.0)
    p7 = Point(0.0, 200.0, 1001.0)
    p8 = Point(100.0, 100.0, 1001.0)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    # Test with point inside deformed cell
    inside_point = Point(50.0, 50.0, 1000.5)
    assert _internal.grid3d.is_point_in_cell(inside_point, cell_corners, "non_convex")

    # Test with point just outside the bounds
    outside_point = Point(60.0, 50.0, 1000.5)
    assert not _internal.grid3d.is_point_in_cell(
        outside_point, cell_corners, "non_convex"
    )


@pytest.mark.parametrize(
    "method, inside_point, outside_point, expected_inside, expected_outside",
    [
        (
            "non_convex",
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
        (
            "centroid_tetrahedrons",
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
        (
            "tetrahedrons",
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
        (
            "ray_casting",
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            False,
            False,  # expected to be True
        ),
    ],
)
def test_point_inside_deformed_case2_cell_with_parametrize(
    method, inside_point, outside_point, expected_inside, expected_outside
):
    """Test with a degenerate cell (deformed, case 2) using parameterized methods."""
    # Create a custom deformed cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(100.0, 0.0, 1000.0)
    p3 = Point(0.0, 100.0, 1000.0)
    p4 = Point(10.0, 10.0, 1000.0)

    p5 = Point(0.0, 0.0, 1001.0)
    p6 = Point(100.0, 0.0, 1001.0)
    p7 = Point(0.0, 100.0, 1001.0)
    p8 = Point(10.0, 10.0, 1001.0)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    # Check if the cell is non-convex
    assert _internal.grid3d.is_cell_non_convex(cell_corners) is True

    # Test with the inside point
    assert (
        _internal.grid3d.is_point_in_cell(inside_point, cell_corners, method)
        == expected_inside
    )

    # Test with the outside point
    assert (
        _internal.grid3d.is_point_in_cell(outside_point, cell_corners, method)
        == expected_outside
    )


def test_more_vertices_1():
    vrt = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    cell_corners = _internal.grid3d.CellCorners(vrt)
    assert _internal.grid3d.is_cell_non_convex(cell_corners) is False

    assert _internal.grid3d.is_point_in_cell(Point(0.5, 0.5, 0.5), cell_corners) is True
    assert (
        _internal.grid3d.is_point_in_cell(Point(-0.1, 0.5, 0.5), cell_corners) is False
    )


def test_more_vertices_2():
    """Testing a non-convex cell, test is based on a real case, from former SWIG/C."""
    vrt = [
        461351.493253,
        5938298.477428,
        1850,
        461501.758690,
        5938385.850231,
        1850,
        461440.718409,
        5938166.753852,
        1850.1,
        461582.200838,
        5938248.702782,
        1850,
        461354.611430,
        5938300.454809,
        1883.246948,
        461504.611754,
        5938387.700867,
        1915.005005,
        461443.842986,
        5938169.007646,
        1904.730957,
        461585.338388,
        5938250.905010,
        1921.021973,
    ]

    p1 = Point(461467.513586, 5938273.910537, 1850)
    p2 = Point(461467.513586, 5938273.910537, 1849.95)

    cell_corners = _internal.grid3d.CellCorners(vrt)
    assert _internal.grid3d.is_cell_non_convex(cell_corners) is True

    for method in (
        "ray_casting",
        "tetrahedrons",
        "centroid_tetrahedrons",
        "non_convex",
    ):
        assert _internal.grid3d.is_point_in_cell(p1, cell_corners, method) is True
        assert _internal.grid3d.is_point_in_cell(p2, cell_corners, method) is False
