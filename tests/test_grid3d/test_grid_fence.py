# coding: utf-8


import os

import xtgeo
import tests.test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TPATH = xtg.testpathobj

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

REEKROOT = TPATH / "3dgrids/reek/REEK"
WELL1 = TPATH / "wells/reek/1/OP_1.w"
FENCE1 = TPATH / "polygons/reek/1/fence.pol"
FENCE2 = TPATH / "polygons/reek/1/minifence.pol"

BIGGRID = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg.roff"
BIGPORO = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg_phix.roff"
BIGGRIDBOX = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_box.roff"
BIGPOROBOX = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_box_phix.roff"
BIGWELL1 = "../xtgeo-testdata-equinor/data/wells/gfb/1/34_10-A-42.w"

# =============================================================================
# Do tests
# =============================================================================


def test_randomline_fence_from_well():
    """Import ROFF grid with props and make fences"""

    grd = xtgeo.Grid(REEKROOT, fformat="eclipserun", initprops=["PORO"])
    wll = xtgeo.Well(WELL1, zonelogname="Zonelog")

    print(grd.describe(details=True))

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)
    print(fspec.dataframe)

    tsetup.assert_almostequal(fspec.dataframe[fspec.dhname][4], 12.6335, 0.001)
    logger.info(fspec.dataframe)

    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1600, zmax=1700, zincrement=1.0
    )

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


@tsetup.equinor
@tsetup.bigtest
def test_randomline_fence_from_well_bigcase_box():
    """Import ROFF grid with props from big case limited to box!"""

    t1 = xtg.timer()
    grd = xtgeo.Grid(BIGGRIDBOX)
    print(f"Get big grid in {xtg.timer(t1):4.3f} seconds.")
    t1 = xtg.timer()

    poro = xtgeo.GridProperty(BIGPOROBOX, name="PHIX")
    print(f"Get big grid poro prop in {xtg.timer(t1):4.3f} seconds.")
    t1 = xtg.timer()
    grd.append_prop(poro)

    wll = xtgeo.Well(BIGWELL1, zonelogname="ZONELOG")
    print(f"Get big well in {xtg.timer(t1):4.3f} seconds.")

    print(grd.describe(details=True))

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)
    # print(fspec.dataframe)

    logger.info(fspec.dataframe)

    t1 = xtg.timer()
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)
    print(f"Get fence polygon in {xtg.timer(t1):4.3f} seconds.")

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    t1 = xtg.timer()
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PHIX", zmin=1700, zmax=2200, zincrement=1.0
    )
    print(f"Sample randomline in {xtg.timer(t1):4.3f} seconds.")

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


@tsetup.equinor
@tsetup.bigtest
def test_randomline_fence_from_well_bigcase():
    """Import ROFF grid with props from big case!"""

    t1 = xtg.timer()
    grd = xtgeo.Grid(BIGGRID)
    print(f"Get big grid in {xtg.timer(t1):4.3f} seconds.")
    t1 = xtg.timer()

    poro = xtgeo.GridProperty(BIGPORO, name="PHIX")
    print(f"Get big grid poro prop in {xtg.timer(t1):4.3f} seconds.")
    t1 = xtg.timer()
    grd.append_prop(poro)

    wll = xtgeo.Well(BIGWELL1, zonelogname="ZONELOG")
    print(f"Get big well in {xtg.timer(t1):4.3f} seconds.")

    print(grd.describe(details=True))

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)
    # print(fspec.dataframe)

    logger.info(fspec.dataframe)

    t1 = xtg.timer()
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)
    print(f"Get fence polygon in {xtg.timer(t1):4.3f} seconds.")

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    t1 = xtg.timer()
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PHIX", zmin=1700, zmax=2200, zincrement=1.0
    )
    print(f"Sample randomline in {xtg.timer(t1):4.3f} seconds.")

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_from_polygon():
    """Import ROFF grid with props and make fence from polygons"""

    grd = xtgeo.Grid(REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(FENCE1)

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    tsetup.assert_almostequal(fspec.dataframe[fspec.dhname][4], 10, 1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    logger.info("Getting randomline...")
    timer1 = xtg.timer()
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1680, zmax=1750, zincrement=0.5
    )
    logger.info("Getting randomline... took {0:5.3f} secs".format(xtg.timer(timer1)))

    timer1 = xtg.timer()
    hmin, hmax, vmin, vmax, perm = grd.get_randomline(
        fspec, "PERMX", zmin=1680, zmax=1750, zincrement=0.5
    )
    logger.info(
        "Getting randomline (2 time)... took {0:5.3f} secs".format(xtg.timer(timer1))
    )

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.figure()
        plt.imshow(perm, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_calczminzmax():
    """Import ROFF grid with props and make fence from polygons, zmin/zmax auto"""

    grd = xtgeo.Grid(REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(FENCE1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=None, zmax=None
    )
    tsetup.assert_almostequal(vmin, 1548.10098, 0.0001)
