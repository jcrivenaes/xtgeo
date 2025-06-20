"""Private module, Grid ETC 1 methods, info/modify/report."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import atan2, degrees
from typing import TYPE_CHECKING, Literal, no_type_check

import numpy as np
import pandas as pd
from packaging.version import parse as versionparse

import xtgeo._internal as _internal  # type: ignore
from xtgeo import _cxtgeo
from xtgeo._internal.geometry import PointInHexahedronMethod as M  # type: ignore
from xtgeo.common.constants import UNDEF_INT, UNDEF_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.common.types import Dimensions
from xtgeo.grid3d.grid_properties import GridProperties
from xtgeo.xyz.polygons import Polygons

from . import _gridprop_lowlevel
from .grid_property import GridProperty

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid
    from xtgeo.grid3d.types import METRIC
    from xtgeo.surface.regular_surface import RegularSurface
    from xtgeo.surface.surfaces import Surfaces
    from xtgeo.xyz.points import Points

logger = null_logger(__name__)


def create_box(
    dimension: Dimensions,
    origin: tuple[float, float, float],
    oricenter: bool,
    increment: tuple[int, int, int],
    rotation: float,
    flip: Literal[1, -1],
) -> dict[str, np.ndarray]:
    """Create a shoebox grid from cubi'sh spec, xtgformat=2."""

    from xtgeo.cube.cube1 import Cube

    ncol, nrow, nlay = dimension
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    coordsv = np.zeros((nncol, nnrow, 6), dtype=np.float64)
    zcornsv = np.zeros((nncol, nnrow, nnlay, 4), dtype=np.float32)
    actnumsv = np.zeros((ncol, nrow, nlay), dtype=np.int32)

    cube = Cube(
        ncol=ncol,
        nrow=nrow,
        nlay=nlay,
        xinc=increment[0],
        yinc=increment[1],
        zinc=increment[2],
        xori=origin[0],
        yori=origin[1],
        zori=origin[2],
        rotation=rotation,
    )

    cubecpp = _internal.cube.Cube(cube)
    logger.debug("Calling CPP internal 'create_grid_from_cube'...")
    coordsv, zcornsv, actnumsv = _internal.grid3d.create_grid_from_cube(
        cubecpp, oricenter, flip
    )
    return {
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv.astype(np.int32),
    }


def create_grid_from_surfaces(
    srfs: Surfaces,
    ij_dimension: tuple[int, int] | None = None,
    ij_origin: tuple[float, float] | None = None,
    ij_increment: tuple[float, float] | None = None,
    rotation: float | None = None,
    tolerance: float = _internal.numerics.TOLERANCE,
) -> Grid:
    """Use a stack of surfaces to create a nonfaulted grid.

    Technically, a shoebox grid is made first, then the layers are adjusted to follow
    surfaces.
    """
    from xtgeo.grid3d.grid import Grid, create_box_grid

    n_surfaces = len(srfs.surfaces)

    # ensure that surfaces are consistent
    if not srfs.is_depth_consistent():
        raise ValueError(
            "Surfaces are not depth consistent, they must not cross is depth"
        )
    top = srfs.surfaces[0]
    base = srfs.surfaces[-1]

    zinc = (base.values.mean() - top.values.mean()) / (n_surfaces - 1)
    kdim: int = n_surfaces - 1
    zori = top.values.mean()
    ncol: int = top.ncol - 1  # since surface are nodes while grid is cell centered
    nrow: int = top.nrow - 1

    if ij_dimension:  # mypy needs this:
        dimension = Dimensions(int(ij_dimension[0]), int(ij_dimension[1]), kdim)
    else:
        dimension = Dimensions(ncol, nrow, kdim)

    increment = (*ij_increment, zinc) if ij_increment else (top.xinc, top.yinc, zinc)
    origin = (*ij_origin, zori) if ij_origin else (top.xori, top.yori, zori)
    rotation = rotation if rotation is not None else top.rotation

    bgrd = create_box_grid(
        dimension=dimension,
        origin=origin,
        increment=increment,
        rotation=rotation,
        oricenter=False,
        flip=1,
    )

    # now adjust the grid to surfaces
    surf_list = []
    for surf in srfs.surfaces:
        cpp_surf = _internal.regsurf.RegularSurface(surf)
        surf_list.append(cpp_surf)

    new_zcorns, new_actnum = bgrd._get_grid_cpp().adjust_boxgrid_layers_from_regsurfs(
        surf_list, tolerance
    )

    grd = Grid(coordsv=bgrd._coordsv.copy(), zcornsv=new_zcorns, actnumsv=new_actnum)

    # set the subgrid index (zones)
    subgrids = {f"zone_{i + 1}": 1 for i in range(n_surfaces - 1)}
    grd.set_subgrids(subgrids)

    return grd


method_factory = {
    "euclid": _cxtgeo.euclid_length,
    "horizontal": _cxtgeo.horizontal_length,
    "east west vertical": _cxtgeo.east_west_vertical_length,
    "north south vertical": _cxtgeo.north_south_vertical_length,
    "x projection": _cxtgeo.x_projection,
    "y projection": _cxtgeo.y_projection,
    "z projection": _cxtgeo.z_projection,
}


def get_dz(
    self: Grid,
    name: str = "dZ",
    flip: bool = True,
    asmasked: bool = True,
    metric: METRIC = "z projection",
) -> GridProperty:
    """Get average cell height (dz) as property.

    Args:
        flip (bool): whether to flip the z direction, ie. increasing z is
            increasing depth (defaults to True)
        asmasked (bool): Whether to mask property by whether
        name (str): Name of resulting grid property, defaults to "dZ".
    """
    if metric not in method_factory:
        raise ValueError(f"Unknown metric {metric}")
    metric_fun = method_factory[metric]

    self._set_xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros(nx * ny * nz)
    _cxtgeo.grdcp3d_calc_dz(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    if not flip:
        result *= -1

    result = np.ma.masked_array(result, self._actnumsv == 0 if asmasked else False)

    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.ravel(),
        name=name,
        discrete=False,
    )


@lru_cache(maxsize=1)
def get_dx(
    self: Grid, name: str = "dX", asmasked: bool = False, metric: METRIC = "horizontal"
) -> GridProperty:
    if metric not in method_factory:
        raise ValueError(f"Unknown metric {metric}")
    metric_fun = method_factory[metric]

    self._set_xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros(nx * ny * nz)
    _cxtgeo.grdcp3d_calc_dx(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    result = np.ma.masked_array(result, self._actnumsv == 0 if asmasked else False)

    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.reshape((nx, ny, nz)),
        name=name,
        discrete=False,
    )


@lru_cache(maxsize=1)
def get_dy(
    self: Grid, name: str = "dY", asmasked: bool = False, metric: METRIC = "horizontal"
) -> GridProperty:
    if metric not in method_factory:
        raise ValueError(f"Unknown metric {metric}")
    metric_fun = method_factory[metric]

    self._set_xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros(nx * ny * nz)
    _cxtgeo.grdcp3d_calc_dy(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    result = np.ma.masked_array(result, self._actnumsv == 0 if asmasked else False)

    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.reshape((nx, ny, nz)),
        name=name,
        discrete=False,
    )


def get_bulk_volume(
    grid: Grid,
    name: str = "bulkvol",
    asmasked: bool = True,
    precision: Literal[1, 2, 4] = 2,
) -> GridProperty:
    """Get cell bulk volume as a GridProperty() instance."""
    if precision not in (1, 2, 4):
        raise ValueError("The precision key has an invalid entry, use 1, 2, or 4")

    grid._set_xtgformat2()

    grid_cpp = grid._get_grid_cpp()

    prec_cpp = _internal.geometry.HexVolumePrecision.P2
    if precision == 1:
        prec_cpp = _internal.geometry.HexVolumePrecision.P1
    elif precision == 4:
        prec_cpp = _internal.geometry.HexVolumePrecision.P4

    bulk_values = grid_cpp.get_cell_volumes(prec_cpp, asmasked)
    if asmasked:
        bulk_values = np.ma.masked_greater(bulk_values, UNDEF_LIMIT)

    return GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        name=name,
        values=bulk_values,
        discrete=False,
    )


def get_heights_above_ffl(
    grid: Grid,
    ffl: GridProperty,
    option: Literal[
        "cell_center_above_ffl", "cell_corners_above_ffl"
    ] = "cell_center_above_ffl",
) -> tuple[GridProperty, GridProperty, GridProperty]:
    """Compute delta heights for cell top, bottom and midpoints above a given level."""

    valid_options = ("cell_center_above_ffl", "cell_corners_above_ffl")
    if option not in valid_options:
        raise ValueError(
            f"The option key <{option}> is invalid, must be one of {valid_options}"
        )

    grid._set_xtgformat2()

    grid_cpp = grid._get_grid_cpp()
    htop_arr, hbot_arr, hmid_arr = grid_cpp.get_height_above_ffl(
        ffl.values.ravel(),
        1 if option == "cell_center_above_ffl" else 2,
    )

    htop = GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        name="htop",
        values=htop_arr,
        discrete=False,
    )
    hbot = GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        name="hbot",
        values=hbot_arr,
        discrete=False,
    )
    hmid = GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        name="hmid",
        values=hmid_arr,
        discrete=False,
    )
    return htop, hbot, hmid


def get_property_between_surfaces(
    grid: Grid,
    top: RegularSurface,
    base: RegularSurface,
    value: int = 1,
    name: str = "between_surfaces",
) -> GridProperty:
    """For a grid, create a grid property with value <value> between two surfaces.

    The value would be zero elsewhere, or if surfaces has inactive nodes.
    """
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"Value (integer) must be positive, >= 1, got: {value}")

    grid._set_xtgformat2()
    logger.debug("Creating property between surfaces...")

    grid_cpp = grid._get_grid_cpp()

    top_ = top
    base_ = base
    if top.yflip == -1:
        top_ = top.copy()
        top_.make_lefthanded()
        logger.debug("Top surface is right-handed, flipping a copy prior to operation")
    if base.yflip == -1:
        base_ = base.copy()
        base_.make_lefthanded()
        logger.debug("Base surface is right-handed, flipping a copy prior to operation")

    diff = base_ - top_
    if (diff.values).all() <= 0:
        raise ValueError(
            "Top surface must be equal or above base surface for all nodes"
        )

    # array is always 0, 1 integer
    array = grid_cpp.get_gridprop_value_between_surfaces(
        _internal.regsurf.RegularSurface(top_),
        _internal.regsurf.RegularSurface(base_),
    )

    logger.debug("Creating property between surfaces... done")

    return GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        name=name,
        values=array * value,
        discrete=True,
    )


def get_ijk(
    self: Grid,
    names: tuple[str, str, str] = ("IX", "JY", "KZ"),
    asmasked: bool = True,
    zerobased: bool = False,
) -> tuple[GridProperty, GridProperty, GridProperty]:
    """Get I J K as properties."""
    ashape = self.dimensions

    ix_idx, jy_idx, kz_idx = np.indices(ashape)

    ix = ix_idx.ravel()
    jy = jy_idx.ravel()
    kz = kz_idx.ravel()

    if asmasked:
        actnum = self.get_actnum()

        ix = np.ma.masked_where(actnum.values1d == 0, ix)
        jy = np.ma.masked_where(actnum.values1d == 0, jy)
        kz = np.ma.masked_where(actnum.values1d == 0, kz)

    if not zerobased:
        ix += 1
        jy += 1
        kz += 1

    ix_gprop = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=ix.reshape(ashape),
        name=names[0],
        discrete=True,
    )
    jy_gprop = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=jy.reshape(ashape),
        name=names[1],
        discrete=True,
    )
    kz_gprop = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=kz.reshape(ashape),
        name=names[2],
        discrete=True,
    )

    # return the objects
    return ix_gprop, jy_gprop, kz_gprop


def get_ijk_from_points(
    self: Grid,
    points: Points,
    activeonly: bool = True,
    zerobased: bool = False,
    dataframe: bool = True,
    includepoints: bool = True,
    columnnames: tuple[str, str, str] = ("IX", "JY", "KZ"),
    fmt: Literal["int", "float"] = "int",
    undef: int = -1,
) -> pd.DataFrame | list:
    """Get I J K indices as a list of tuples or a dataframe.

    It is here tried to get fast execution. This requires a preprosessing
    of the grid to store a onelayer version, and maps with IJ positions. This is
    stored as a cache variable we can derive.
    """
    logger.info("Getting IJK indices from Points...")

    self._set_xtgformat2()

    cache = self._get_cache()

    points_df = points.get_dataframe(copy=False)

    p_array = points.get_xyz_arrays()

    iarr, jarr, karr = self._get_grid_cpp().get_indices_from_pointset(
        _internal.xyz.PointSet(p_array),
        cache.onegrid_cpp,
        cache.top_i_index_cpp,
        cache.top_j_index_cpp,
        cache.base_i_index_cpp,
        cache.base_j_index_cpp,
        cache.top_depth_cpp,
        cache.base_depth_cpp,
        cache.threshold_magic_1,
        activeonly,
        M.Optimized,
    )

    if not zerobased:
        iarr = np.where(iarr >= 0, iarr + 1, iarr)
        jarr = np.where(jarr >= 0, jarr + 1, jarr)
        karr = np.where(karr >= 0, karr + 1, karr)

    proplist = {}
    if includepoints:
        proplist["X_UTME"] = points_df[points.xname].to_numpy()
        proplist["Y_UTMN"] = points_df[points.yname].to_numpy()
        proplist["Z_TVDSS"] = points_df[points.zname].to_numpy()

    proplist[columnnames[0]] = iarr
    proplist[columnnames[1]] = jarr
    proplist[columnnames[2]] = karr

    mydataframe = pd.DataFrame.from_dict(proplist)
    mydataframe = mydataframe.replace(UNDEF_INT, -1)

    if fmt == "float":
        mydataframe[columnnames[0]] = mydataframe[columnnames[0]].astype("float")
        mydataframe[columnnames[1]] = mydataframe[columnnames[1]].astype("float")
        mydataframe[columnnames[2]] = mydataframe[columnnames[2]].astype("float")

    if undef != -1:
        mydataframe[columnnames[0]] = mydataframe[columnnames[0]].replace(-1, undef)
        mydataframe[columnnames[1]] = mydataframe[columnnames[1]].replace(-1, undef)
        mydataframe[columnnames[2]] = mydataframe[columnnames[2]].replace(-1, undef)

    logger.info(
        "Getting IJK indices from Points... done, found %d points", len(mydataframe)
    )
    if dataframe:
        return mydataframe

    return list(mydataframe.itertuples(index=False, name=None))


@lru_cache(maxsize=1)
def get_xyz(
    self: Grid,
    names: tuple[str, str, str] = ("X_UTME", "Y_UTMN", "Z_TVDSS"),
    asmasked: bool = True,
) -> tuple[GridProperty, GridProperty, GridProperty]:
    """Get X Y Z as properties."""

    self._set_xtgformat2()

    # note: using _internal here is 2-3 times faster than using the former cxtgeo!
    grid_cpp = self._get_grid_cpp()
    xv, yv, zv = grid_cpp.get_cell_centers(asmasked)

    xv = np.ma.masked_invalid(xv)
    yv = np.ma.masked_invalid(yv)
    zv = np.ma.masked_invalid(zv)

    xo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=xv,
        name=names[0],
        discrete=False,
    )

    yo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=yv,
        name=names[1],
        discrete=False,
    )

    zo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=zv,
        name=names[2],
        discrete=False,
    )

    # return the objects
    return xo, yo, zo


def get_xyz_cell_corners_internal(
    grid: Grid,
    ijk: tuple[int, int, int] = (1, 1, 1),
    activeonly: bool = True,
    zerobased: bool = False,
) -> tuple[int, ...] | None:
    """Get X Y Z cell corners for one cell."""
    grid._set_xtgformat2()

    i, j, k = ijk
    shift = 1 if zerobased else 0

    if activeonly:
        actnum = grid.get_actnum()
        iact = actnum.values[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if np.all(iact == 0):
            return None

    # there are some cases where we don't want to use cache due to recusion issues
    corners = grid._get_grid_cpp().get_cell_corners_from_ijk(
        i + shift - 1,
        j + shift - 1,
        k + shift - 1,
    )

    corners = corners.to_numpy().flatten().tolist()
    return tuple(corners)


def get_xyz_corners(
    self: Grid, names: tuple[str, str, str] = ("X_UTME", "Y_UTMN", "Z_TVDSS")
) -> tuple[GridProperty, ...]:
    """Get X Y Z cell corners for all cells (as 24 GridProperty objects)."""
    self._set_xtgformat1()

    ntot = self.dimensions

    grid_props = []

    for i in range(8):
        xname = names[0] + str(i)
        yname = names[1] + str(i)
        zname = names[2] + str(i)
        x = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=xname,
            discrete=False,
        )

        y = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=yname,
            discrete=False,
        )

        z = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=zname,
            discrete=False,
        )

        grid_props.append(x)
        grid_props.append(y)
        grid_props.append(z)

    ptr_coord = []
    for i in range(24):
        some = _cxtgeo.new_doublearray(self.ntotal)
        ptr_coord.append(some)
        logger.debug("SWIG object %s   %s", i, some)

    option = 0

    # note, fool the argument list to unpack ptr_coord with * ...
    _cxtgeo.grd3d_get_all_corners(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        *(ptr_coord + [option]),
    )

    for i in range(0, 24, 3):
        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i], ptr_coord[i], np.float64, delete=True
        )

        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i + 1], ptr_coord[i + 1], np.float64, delete=True
        )

        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i + 2], ptr_coord[i + 2], np.float64, delete=True
        )

    # return the 24 objects (x1, y1, z1, ... x8, y8, z8)
    return tuple(grid_props)


def get_vtk_esg_geometry_data(
    self: Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get geometry data consisting of vertices and cell connectivities suitable for
    use with VTK's vtkExplicitStructuredGrid.

    Returned tuple contains:
    - numpy array with dimensions in terms of points (not cells)
    - vertex array, numpy array with vertex coordinates
    - connectivity array for all the cells, numpy array with integer indices
    - inactive cell indices, numpy array with integer indices
    """

    self._set_xtgformat2()

    # Number of elements to allocate in the vertex and connectivity arrays
    num_cells = self.ncol * self.nrow * self.nlay
    n_vertex_arr = 3 * 8 * num_cells
    n_conn_arr = 8 * num_cells

    # Note first value in return tuple which is the actual number of vertices that
    # was written into vertex_arr and which we'll use to shrink the array.
    vertex_count, vertex_arr, conn_arr = _cxtgeo.grdcp3d_get_vtk_esg_geometry_data(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        n_vertex_arr,
        n_conn_arr,
    )

    # Need to shrink the vertex array
    vertex_arr = np.resize(vertex_arr, 3 * vertex_count)
    vertex_arr = vertex_arr.reshape(-1, 3)

    point_dims = np.asarray((self.ncol, self.nrow, self.nlay)) + 1
    inact_indices = self.get_actnum_indices(order="F", inverse=True)

    return point_dims, vertex_arr, conn_arr, inact_indices


def get_vtk_geometries(self: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actnum, corners and dims arrays for VTK ExplicitStructuredGrid usage."""
    self._set_xtgformat2()

    narr = 8 * self.ncol * self.nrow * self.nlay
    xarr, yarr, zarr = _cxtgeo.grdcp3d_get_vtk_grid_arrays(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        narr,
        narr,
        narr,
    )
    corners = np.stack((xarr, yarr, zarr))
    corners = corners.transpose()

    dims = np.asarray((self.ncol, self.nrow, self.nlay)) + 1

    actindices = self.get_actnum_indices(order="F", inverse=True)

    return dims, corners, actindices


def get_cell_volume(
    grid: Grid,
    ijk: tuple[int, int, int] = (1, 1, 1),
    activeonly: bool = True,
    zerobased: bool = False,
    precision: Literal[1, 2, 4] = 2,
) -> float | None:
    """Get bulk cell volume for one cell."""
    if precision not in (1, 2, 4):
        raise ValueError("The precision key has an invalid entry, use 1, 2, or 4")
    grid._set_xtgformat2()

    i, j, k = ijk
    shift = 1 if zerobased else 0

    if activeonly:
        actnum = grid.get_actnum()
        iact = actnum.values[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if np.all(iact == 0):
            return None

    corners = grid._get_grid_cpp().get_cell_corners_from_ijk(
        i + shift - 1,
        j + shift - 1,
        k + shift - 1,
    )
    prec = _internal.geometry.HexVolumePrecision.P2
    if precision == 1:
        prec = _internal.geometry.HexVolumePrecision.P1
    elif precision == 4:
        prec = _internal.geometry.HexVolumePrecision.P4

    return _internal.geometry.hexahedron_volume(corners, prec)


def get_layer_slice(
    self: Grid, layer: int, top: bool = True, activeonly: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Get X Y cell corners (XY per cell; 5 per cell) as array."""
    self._set_xtgformat1()
    ntot = self._ncol * self._nrow * self._nlay

    opt1 = 0 if top else 1
    opt2 = 1 if activeonly else 0

    icn, lay_array, ic_array = _cxtgeo.grd3d_get_lay_slice(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        layer,
        opt1,
        opt2,
        10 * ntot,
        ntot,
    )

    lay_array = lay_array[: 10 * icn]
    ic_array = ic_array[:icn]

    lay_array = lay_array.reshape((icn, 5, 2))

    return lay_array, ic_array


def get_geometrics(
    self: Grid,
    allcells: bool = False,
    cellcenter: bool = True,
    return_dict: bool = False,
    _ver: Literal[1, 2] = 1,
) -> dict | tuple:
    """Getting cell geometrics."""
    self._set_xtgformat1()

    geom_function = _get_geometrics_v1 if _ver == 1 else _get_geometrics_v2
    return geom_function(
        self, allcells=allcells, cellcenter=cellcenter, return_dict=return_dict
    )


def _get_geometrics_v1(
    self: Grid,
    allcells: bool = False,
    cellcenter: bool = True,
    return_dict: bool = False,
) -> dict | tuple:
    ptr_x = [_cxtgeo.new_doublepointer() for i in range(13)]

    option1 = 0 if allcells else 1
    option2 = 1 if cellcenter else 0

    quality = _cxtgeo.grd3d_geometrics(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        ptr_x[0],
        ptr_x[1],
        ptr_x[2],
        ptr_x[3],
        ptr_x[4],
        ptr_x[5],
        ptr_x[6],
        ptr_x[7],
        ptr_x[8],
        ptr_x[9],
        ptr_x[10],
        ptr_x[11],
        ptr_x[12],
        option1,
        option2,
    )

    glist = [_cxtgeo.doublepointer_value(item) for item in ptr_x]
    glist.append(quality)

    logger.info("Cell geometrics done")

    if not return_dict:
        return tuple(glist)

    gkeys = [
        "xori",
        "yori",
        "zori",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "avg_rotation",
        "avg_dx",
        "avg_dy",
        "avg_dz",
        "grid_regularity_flag",
    ]
    return dict(zip(gkeys, glist))


def _get_geometrics_v2(
    self: Grid,
    allcells: bool = False,
    cellcenter: bool = True,
    return_dict: bool = False,
) -> dict | tuple:
    # Currently a workaround as there seems to be bugs in v1
    # Will only work with allcells False and cellcenter True

    glist = []
    if cellcenter and allcells:
        xcor, ycor, zcor = self.get_xyz(asmasked=False)
        glist.append(xcor.values[0, 0, 0])
        glist.append(ycor.values[0, 0, 0])
        glist.append(zcor.values[0, 0, 0])
        glist.append(xcor.values.min())
        glist.append(xcor.values.max())
        glist.append(ycor.values.min())
        glist.append(ycor.values.max())
        glist.append(zcor.values.min())
        glist.append(zcor.values.max())

        # rotation (approx) for mid column
        midcol = int(self.nrow / 2)
        midlay = int(self.nlay / 2)
        x0 = xcor.values[0, midcol, midlay]
        y0 = ycor.values[0, midcol, midlay]
        x1 = xcor.values[self.ncol - 1, midcol, midlay]
        y1 = ycor.values[self.ncol - 1, midcol, midlay]
        glist.append(degrees(atan2(y1 - y0, x1 - x0)))

        dx = self.get_dx(asmasked=False)
        dy = self.get_dy(asmasked=False)
        dz = self.get_dz(asmasked=False)
        glist.append(dx.values.mean())
        glist.append(dy.values.mean())
        glist.append(dz.values.mean())
        glist.append(1)

    if not return_dict:
        return tuple(glist)

    gkeys = [
        "xori",
        "yori",
        "zori",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "avg_rotation",
        "avg_dx",
        "avg_dy",
        "avg_dz",
        "grid_regularity_flag",
    ]
    return dict(zip(gkeys, glist))


def inactivate_by_dz(self: Grid, threshold: float, flip: bool = True) -> None:
    """Set cell to inactive if dz does not exceed threshold.
    Args:
        threshold (float): The threshold for which the absolute value
            of dz should exceed.
        flip (bool): Whether the z-direction should be flipped.

    """
    self._set_xtgformat2()
    dz_values = self.get_dz(asmasked=False, flip=flip).values
    self._actnumsv[dz_values.reshape(self._actnumsv.shape) < threshold] = 0


def make_zconsistent(self: Grid, zsep: float | int) -> None:
    """Make consistent in z."""
    self._set_xtgformat1()

    if isinstance(zsep, int):
        zsep = float(zsep)

    if not isinstance(zsep, float):
        raise ValueError('The "zsep" is not a float or int')

    _cxtgeo.grd3d_make_z_consistent(
        self.ncol,
        self.nrow,
        self.nlay,
        self._zcornsv,
        zsep,
    )


def inactivate_inside(
    self: Grid,
    poly: Polygons,
    layer_range: tuple[int, int] | None = None,
    inside: bool = True,
    force_close: bool = False,
) -> None:
    """Inactivate inside a polygon (or outside)."""
    self._set_xtgformat1()

    if not isinstance(poly, Polygons):
        raise ValueError("Input polygon not a XTGeo Polygons instance")

    if layer_range is not None:
        k1, k2 = layer_range
    else:
        k1, k2 = 1, self.nlay

    method = 0 if inside else 1
    iforce = 0 if not force_close else 1

    # get dataframe where each polygon is ended by a 999 value
    dfxyz = poly.get_xyz_dataframe()

    xc = dfxyz["X_UTME"].values.copy()
    yc = dfxyz["Y_UTMN"].values.copy()

    ier = _cxtgeo.grd3d_inact_outside_pol(
        xc,
        yc,
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,  # is modified!
        k1,
        k2,
        iforce,
        method,
    )

    if ier == 1:
        raise RuntimeError("Problems with one or more polygons. Not closed?")


def collapse_inactive_cells(self: Grid) -> None:
    """Collapse inactive cells."""
    self._set_xtgformat1()

    _cxtgeo.grd3d_collapse_inact(
        self.ncol, self.nrow, self.nlay, self._zcornsv, self._actnumsv
    )


def copy(self: Grid) -> Grid:
    """Copy a grid instance.

    Returns:
        A new instance (attached grid properties will also be unique)
    """
    self._set_xtgformat2()

    copy_tag = " (copy)"

    filesrc = str(self._filesrc)
    if filesrc is not None and copy_tag not in filesrc:
        filesrc += copy_tag

    return self.__class__(
        coordsv=self._coordsv.copy(),
        zcornsv=self._zcornsv.copy(),
        actnumsv=self._actnumsv.copy(),
        subgrids=deepcopy(self.subgrids),
        dualporo=self.dualporo,
        dualperm=self.dualperm,
        name=self.name + copy_tag if self.name else None,
        roxgrid=self.roxgrid,
        roxindexer=self.roxindexer,
        props=self._props.copy() if self._props else None,
        filesrc=filesrc,
    )


@no_type_check  # due to some hard-to-solve issues with mypy
def crop(
    self: Grid,
    spec: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    props: Literal["all"] | list[GridProperty] | None = None,
) -> None:
    """Do cropping of geometry (and properties).

    If props is 'all' then all properties assosiated (linked) to then
    grid are also cropped, and the instances are updated.

    Args:
        spec (tuple): A nested tuple on the form ((i1, i2), (j1, j2), (k1, k2))
            where 1 represents start number, and 2 reperesent end. The range
            is inclusive for both ends, and the number start index is 1 based.
        props (list or str): None is default, while properties can be listed.
            If 'all', then all GridProperty objects which are linked to the
            Grid instance are updated.

    Returns:
        The instance is updated (cropped)
    """
    self._set_xtgformat1()

    (ic1, ic2), (jc1, jc2), (kc1, kc2) = spec

    if (
        ic1 < 1
        or ic2 > self.ncol
        or jc1 < 1
        or jc2 > self.nrow
        or kc1 < 1
        or kc2 > self.nlay
    ):
        raise ValueError("Boundary for tuples not matching grid NCOL, NROW, NLAY")

    oldnlay = self._nlay

    # compute size of new cropped grid
    nncol = ic2 - ic1 + 1
    nnrow = jc2 - jc1 + 1
    nnlay = kc2 - kc1 + 1

    ntot = nncol * nnrow * nnlay
    ncoord = (nncol + 1) * (nnrow + 1) * 2 * 3
    nzcorn = nncol * nnrow * (nnlay + 1) * 4

    new_num_act = _cxtgeo.new_intpointer()
    new_coordsv = np.zeros(ncoord, dtype=np.float64)
    new_zcornsv = np.zeros(nzcorn, dtype=np.float64)
    new_actnumsv = np.zeros(ntot, dtype=np.int32)

    _cxtgeo.grd3d_crop_geometry(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        new_coordsv,
        new_zcornsv,
        new_actnumsv,
        ic1,
        ic2,
        jc1,
        jc2,
        kc1,
        kc2,
        new_num_act,
        0,
    )

    self._coordsv = new_coordsv
    self._zcornsv = new_zcornsv
    self._actnumsv = new_actnumsv

    self._ncol = nncol
    self._nrow = nnrow
    self._nlay = nnlay

    if isinstance(self.subgrids, dict):
        newsub = {}
        # easier to work with numpies than lists
        newarr = np.array(range(1, oldnlay + 1))
        newarr[newarr < kc1] = 0
        newarr[newarr > kc2] = 0
        newaxx = newarr.copy() - kc1 + 1
        for sub, arr in self.subgrids.items():
            arrx = np.array(arr)
            arrxmap = newaxx[arrx[0] - 1 : arrx[-1]]
            arrxmap = arrxmap[arrxmap > 0]
            if arrxmap.size > 0:
                newsub[sub] = arrxmap.astype(np.int32).tolist()

        self.subgrids = newsub

    # crop properties
    props = self.props if props == "all" else props
    if props is not None:
        for prop in props:
            logger.info("Crop %s", prop.name)
            prop.crop(spec)


def reduce_to_one_layer(self: Grid) -> None:
    """Reduce the grid to one single layer.

    This can be useful for algorithms that need to test if a point is within
    the full grid.

    Example::

        >>> import xtgeo
        >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
        >>> grid.nlay
        14
        >>> grid.reduce_to_one_layer()
        >>> grid.nlay
        1

    """
    # need new pointers in C (not for coord)
    # Note this could probably be done with pure numpy operations
    self._set_xtgformat1()

    ptr_new_num_act = _cxtgeo.new_intpointer()

    nnum = (1 + 1) * 4

    new_zcorn = np.zeros(self.ncol * self.nrow * nnum, dtype=np.float64)
    new_actnum = np.zeros(self.ncol * self.nrow * 1, dtype=np.int32)

    _cxtgeo.grd3d_reduce_onelayer(
        self.ncol,
        self.nrow,
        self.nlay,
        self._zcornsv,
        new_zcorn,
        self._actnumsv,
        new_actnum,
        ptr_new_num_act,
        0,
    )

    self._nlay = 1
    self._zcornsv = new_zcorn
    self._actnumsv = new_actnum
    self._props = None
    self._subgrids = None


def translate_coordinates(
    self: Grid,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    flip: tuple[int, int, int] = (1, 1, 1),
) -> None:
    """Translate grid coordinates."""
    self._set_xtgformat1()

    tx, ty, tz = translate
    fx, fy, fz = flip

    ier = _cxtgeo.grd3d_translate(
        self._ncol,
        self._nrow,
        self._nlay,
        fx,
        fy,
        fz,
        tx,
        ty,
        tz,
        self._coordsv,
        self._zcornsv,
    )
    if ier != 0:
        raise RuntimeError(f"Something went wrong in translate, code: {ier}")

    logger.info("Translation of coords done")


def reverse_row_axis(
    self: Grid, ijk_handedness: Literal["left", "right"] | None = None
) -> None:
    """Reverse rows (aka flip) for geometry and assosiated properties."""
    if ijk_handedness == self.ijk_handedness:
        return

    # update the handedness
    if ijk_handedness is None:
        self._ijk_handedness = estimate_handedness(self)

    original_handedness = self._ijk_handedness
    original_xtgformat = self._xtgformat

    self._set_xtgformat1()

    ier = _cxtgeo.grd3d_reverse_jrows(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        self._actnumsv.ravel(),
    )

    if ier != 0:
        raise RuntimeError(f"Something went wrong in jswapping, code: {ier}")

    if self._props is None:
        return

    # do it for properties
    if self._props.props:
        for prp in self._props.props:
            prp.values = prp.values[:, ::-1, :]

    # update the handedness
    if ijk_handedness is None:
        self._ijk_handedness = estimate_handedness(self)

    if original_handedness == "left":
        self._ijk_handedness = "right"
    else:
        self._ijk_handedness = "left"

    if original_xtgformat == 2:
        self._set_xtgformat2()

    logger.info("Reversing of rows done")


def get_adjacent_cells(
    self: Grid,
    prop: GridProperty,
    val1: int,
    val2: int,
    activeonly: bool = True,
) -> GridProperty:
    """Get adjacents cells."""
    self._set_xtgformat1()

    if not isinstance(prop, GridProperty):
        raise ValueError("The argument prop is not a xtgeo.GridPropery")

    if prop.isdiscrete is False:
        raise ValueError("The argument prop is not a discrete property")

    result = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(self.ntotal, dtype=np.int32),
        name="ADJ_CELLS",
        discrete=True,
    )

    p_prop1 = _gridprop_lowlevel.update_carray(prop)
    p_prop2 = _cxtgeo.new_intarray(self.ntotal)

    iflag1 = 0 if activeonly else 1
    iflag2 = 1

    _cxtgeo.grd3d_adj_cells(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        p_prop1,
        self.ntotal,
        val1,
        val2,
        p_prop2,
        self.ntotal,
        iflag1,
        iflag2,
    )

    _gridprop_lowlevel.update_values_from_carray(result, p_prop2, np.int32, delete=True)
    # return the property object
    return result


def estimate_design(
    self: Grid,
    nsubname: str | None = None,
) -> dict[str, str | float]:
    """Estimate (guess) (sub)grid design by examing DZ in median thickness column."""
    actv = self.get_actnum().values

    dzv = self.get_dz(asmasked=False).values

    # treat inactive thicknesses as zero
    dzv[actv == 0] = 0.0

    if nsubname is None:
        vrange = np.array(range(self.nlay))
    else:
        assert self.subgrids is not None
        vrange = np.array(list(self.subgrids[nsubname])) - 1

    # find the dz for the actual subzone
    dzv = dzv[:, :, vrange]

    # find cumulative thickness as a 2D array
    dzcum: np.ndarray = np.sum(dzv, axis=2, keepdims=False)

    # find the average thickness for nonzero thicknesses
    dzcum2 = dzcum.copy()
    dzcum2[dzcum == 0.0] = np.nan
    dzavg = np.nanmean(dzcum2) / dzv.shape[2]

    # find the I J indices for the median value
    if versionparse(np.__version__) < versionparse("1.22"):
        median_value = np.percentile(dzcum, 50, interpolation="nearest")  # type: ignore
    else:
        median_value = np.percentile(dzcum, 50, method="nearest")

    argmed = np.stack(np.nonzero(dzcum == median_value), axis=1)

    im, jm = argmed[0]
    # find the dz stack of the median
    dzmedian = dzv[im, jm, :]
    logger.info("DZ median column is %s", dzmedian)

    # to compare thicknesses with (divide on 2 to assure)
    target = dzcum[im, jm] / (dzmedian.shape[0] * 2)
    eps = target / 100.0

    logger.info("Target and EPS values are %s, %s", target, eps)

    status = "X"  # unknown or cannot determine

    if dzmedian[0] > target and dzmedian[-1] <= eps:
        status = "T"
        dzavg = dzmedian[0]
    elif dzmedian[0] < eps and dzmedian[-1] > target:
        status = "B"
        dzavg = dzmedian[-1]
    elif dzmedian[0] > target and dzmedian[-1] > target:
        ratio = dzmedian[0] / dzmedian[-1]
        if 0.5 < ratio < 1.5:
            status = "P"
    elif dzmedian[0] < eps and dzmedian[-1] < eps:
        status = "M"
        middleindex = int(dzmedian.shape[0] / 2)
        dzavg = dzmedian[middleindex]

    return {"design": status, "dzsimbox": dzavg}


def estimate_handedness(self: Grid) -> Literal["left", "right"]:
    """Estimate if grid is left or right handed, returning string."""
    nflip = self.estimate_flip()

    return "left" if nflip == 1 else "right"


def _convert_xtgformat2to1(self: Grid) -> None:
    """Convert arrays from new structure xtgformat=2 to legacy xtgformat=1."""
    if self._xtgformat == 1:
        logger.info("No conversion, format is already xtgformat == 1 or unset")
        return

    logger.info("Convert grid from new xtgformat to legacy format...")

    newcoordsv = np.zeros(((self._ncol + 1) * (self._nrow + 1) * 6), dtype=np.float64)
    newzcornsv = np.zeros(
        (self._ncol * self._nrow * (self._nlay + 1) * 4), dtype=np.float64
    )
    newactnumsv = np.zeros((self._ncol * self._nrow * self._nlay), dtype=np.int32)

    _cxtgeo.grd3cp3d_xtgformat2to1_geom(
        self._ncol,
        self._nrow,
        self._nlay,
        newcoordsv,
        self._coordsv,
        newzcornsv,
        self._zcornsv,
        newactnumsv,
        self._actnumsv,
    )

    self._coordsv = newcoordsv
    self._zcornsv = newzcornsv
    self._actnumsv = newactnumsv
    self._xtgformat = 1

    logger.info("Convert grid from new xtgformat to legacy format... done")


def _convert_xtgformat1to2(self: Grid) -> None:
    """Convert arrays from old structure xtgformat=1 to new xtgformat=2."""
    if self._xtgformat == 2 or self._coordsv is None:
        logger.info("No conversion, format is already xtgformat == 2 or unset")
        return

    logger.info("Convert grid from legacy xtgformat to new format...")

    newcoordsv = np.zeros((self._ncol + 1, self._nrow + 1, 6), dtype=np.float64)
    newzcornsv = np.zeros(
        (self._ncol + 1, self._nrow + 1, self._nlay + 1, 4), dtype=np.float32
    )
    newactnumsv = np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.int32)

    _cxtgeo.grd3cp3d_xtgformat1to2_geom(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        newcoordsv,
        self._zcornsv,
        newzcornsv,
        self._actnumsv,
        newactnumsv,
    )

    self._coordsv = newcoordsv
    self._zcornsv = newzcornsv
    self._actnumsv = newactnumsv
    self._xtgformat = 2

    logger.info("Convert grid from legacy xtgformat to new format... done")


def get_gridquality_properties(self: Grid) -> GridProperties:
    """Get the grid quality properties."""
    self._set_xtgformat2()

    qcnames = {
        0: "minangle_topbase",
        1: "maxangle_topbase",
        2: "minangle_topbase_proj",
        3: "maxangle_topbase_proj",
        4: "minangle_sides",
        5: "maxangle_sides",
        6: "collapsed",
        7: "faulted",
        8: "negative_thickness",
        9: "concave_proj",
    }

    # some of the properties shall be discrete:
    qcdiscrete = [6, 7, 8, 9]

    fresults = np.ones(
        (len(qcnames), self.ncol * self.nrow * self.nlay), dtype=np.float32
    )

    _cxtgeo.grdcp3d_quality_indicators(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        fresults,
    )

    grdprops = GridProperties()

    for num, name in qcnames.items():
        discrete = num in qcdiscrete
        prop = GridProperty(
            self,
            name=name,
            discrete=discrete,
            values=fresults[num, :].astype(np.int32 if discrete else np.float32),
            codes={0: "None", 1: name} if discrete else None,
        )
        grdprops.append_props([prop])

    return grdprops
