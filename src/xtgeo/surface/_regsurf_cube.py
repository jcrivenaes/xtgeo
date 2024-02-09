"""Regular surface vs Cube"""


from warnings import warn

import numpy as np

import xtgeo
from xtgeo import _cxtgeo
from xtgeo.common import null_logger

logger = null_logger(__name__)


def slice_cube(
    self,
    cube,
    zsurf=None,
    sampling="nearest",
    mask=True,
    snapxy=False,
    deadtraces=True,
    algorithm=1,
):
    """Slicing a cube, using different algorithms"""

    if algorithm == 2:
        if snapxy:
            logger.debug("Slicing cube algorithm 2, with snapxy active...")
            return _slice_cube_v2(
                self,
                cube,
                zsurf=zsurf,
                sampling=sampling,
                mask=mask,
                deadtraces=deadtraces,
            )
        logger.debug("Slicing cube algorithm 2, with resample...")
        return _slice_cube_v2_resample(
            self,
            cube,
            zsurf=zsurf,
            sampling=sampling,
            mask=mask,
            deadtraces=deadtraces,
        )

    # legacy version:
    return _slice_cube_v1(
        self,
        cube,
        zsurf=zsurf,
        sampling=sampling,
        mask=mask,
        snapxy=snapxy,
        deadtraces=deadtraces,
    )


def _slice_cube_v1(
    self, cube, zsurf=None, sampling="nearest", mask=True, snapxy=False, deadtraces=True
):
    """
    Private function for the Cube slicing. This is the legacy version up to version
    2.8.x but may remain default for a while
    """

    logger.info("Slice cube algorithm 1")
    if zsurf is not None:
        other = zsurf
    else:
        logger.info("The current surface is copied as 'other'")
        other = self.copy()

    if not self.compare_topology(other, strict=False):
        raise RuntimeError("Topology of maps differ. Stop!")

    opt2 = 0 if mask else 1

    if deadtraces:
        # set dead traces to cxtgeo UNDEF -> special treatment in the C code
        olddead = cube.values_dead_traces(xtgeo.UNDEF)

    cubeval1d = np.ravel(cube.values, order="C")

    nsurf = self.ncol * self.nrow

    usesampling = 0
    if sampling == "trilinear":
        usesampling = 1
        if snapxy:
            usesampling = 2

    logger.debug("Running method from C... (using typemaps for numpies!:")
    istat, v1d = _cxtgeo.surf_slice_cube(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        cube.zori,
        cube.zinc,
        cube.rotation,
        cube.yflip,
        cubeval1d,
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.yflip,
        self.rotation,
        other.get_values1d(),
        nsurf,
        usesampling,
        opt2,
    )

    self.set_values1d(v1d)

    if deadtraces:
        cube.values_dead_traces(olddead)  # reset value for dead traces

    return istat


def _slice_cube_v2(
    self, cube, zsurf=None, sampling="nearest", mask=True, deadtraces=True
):
    """
    This is the new version, optimised for the case where the surface has exact same
    topology as the cube. This should both simplify and speed up calculations

    From xtgeo 2.9
    """

    logger.info("Slice cube algorithm 2 with sampling %s", sampling)
    other = zsurf if zsurf is not None else self.copy()

    if not self.compare_topology(other, strict=False):
        raise RuntimeError("Topology of maps differ. Stop!")

    optmask = 0
    if mask:
        optmask = 1

    if deadtraces:
        # set dead traces to cxtgeo UNDEF -> special treatment in the C code
        olddead = cube.values_dead_traces(xtgeo.UNDEF)

    optnearest = 1
    if sampling == "trilinear":
        optnearest = 0

    # cube and surf shall share same topology, e.g. cube.col == surf.ncol etc
    # print(self.values.mask)
    istat = _cxtgeo.surf_slice_cube_v3(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.zori,
        cube.zinc,
        cube.values,
        other.values.data,
        self.values.data,
        self.values.mask,
        optnearest,
        optmask,
    )

    if istat != 0:
        logger.warning("Problem, ISTAT = %s", istat)

    if deadtraces:
        cube.values_dead_traces(olddead)  # reset value for dead traces

    return istat


def _slice_cube_v2_resample(
    self, cube, zsurf=None, sampling="nearest", mask=True, deadtraces=True
):
    """Slicing with surfaces that not match the cube geometry, snapxy=False

    The idea here is to resample the surface to the cube, then afterwards
    do an inverse sampling
    """

    initial_count = self.nactive if self.nactive is not None else 0

    surface_from_cube = xtgeo.surface_from_cube(cube, 0)

    if self.compare_topology(surface_from_cube, strict=False):
        logger.debug("The cube and input surface matches")
        return _slice_cube_v2(self, cube, zsurf, sampling, mask, deadtraces)

    logger.debug("Resample surface...")
    logger.debug("Active input nodes (instance): %s", self.nactive)
    logger.debug("Active surface from cube nodes: %s", surface_from_cube.nactive)
    surface_from_cube.resample(self)
    logger.debug(
        "Active after resample nodes (surf from cube): %s", surface_from_cube.nactive
    )
    logger.debug(
        "Average of surface_from_cube after resample: %s",
        surface_from_cube.values.mean(),
    )

    print(self)
    print(surface_from_cube)

    if surface_from_cube.nactive == 0:
        warn(
            "No surface values will sampled from cube. This is usually caused by "
            "spatial misalignment between cube and surface, or that the input surface "
            "only have masked (undefined) values.",
            UserWarning,
        )
        return -5  #  -5 means no coverage

    zcube = None
    if zsurf:
        zcube = surface_from_cube.copy()
        zcube.resample(zsurf)

    istat = _slice_cube_v2(
        surface_from_cube,
        cube=cube,
        zsurf=zcube,
        sampling=sampling,
        mask=mask,
        deadtraces=deadtraces,
    )

    # sample back
    self.resample(surface_from_cube, mask=mask)

    updated_count = self.nactive if self.nactive is not None else 0

    if updated_count <= 0.1 * initial_count:
        return -4

    return istat
