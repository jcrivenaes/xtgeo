# coding: utf-8
"""Various operations on XYZ data"""

import numpy as np
import pandas as pd
import shapely.geometry as sg
from scipy.interpolate import UnivariateSpline

import xtgeo
from xtgeo import _cxtgeo
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGeoDialog

xtg = XTGeoDialog()
logger = null_logger(__name__)


def mark_in_polygons_mpl(self, poly, name, inside_value, outside_value):
    """Make column to mark if XYZ df is inside/outside polygons, using matplotlib."""
    points = np.array(
        [
            self.get_dataframe(copy=False)[self.xname].values,
            self.get_dataframe(copy=False)[self.yname].values,
        ]
    ).T

    if name in self.protected_columns():
        raise ValueError("The proposed name: {name}, is protected and cannot be used")

    # allow a single Polygons instance or a list of Polygons instances
    if isinstance(poly, xtgeo.Polygons):
        usepolys = [poly]
    elif isinstance(poly, list) and all(
        isinstance(pol, xtgeo.Polygons) for pol in poly
    ):
        usepolys = poly
    else:
        raise ValueError("The poly values is not a Polygons or a list of Polygons")

    dataframe = self.get_dataframe()
    dataframe[name] = outside_value

    import matplotlib.path as mplpath

    for pol in usepolys:
        idgroups = pol.get_dataframe().groupby(pol.pname)
        for _, grp in idgroups:
            singlepoly = np.array([grp[pol.xname].values, grp[pol.yname].values]).T
            poly_path = mplpath.Path(singlepoly)
            is_inside = poly_path.contains_points(points)
            dataframe.loc[is_inside, name] = inside_value

    self.set_dataframe(dataframe)


def operation_polygons_v1(self, poly, value, opname="add", inside=True, where=True):
    """
    Operations re restricted to closed polygons, for points or polyline points.

    If value is not float but 'poly', then the avg of each polygon Z value will
    be used instead.

    'Inside' several polygons will become a union, while 'outside' polygons
    will be the intersection.

    The "where" filter is reserved for future use.

    This is now the legacy version, which will be deprecated later.
    """
    logger.warning("Where is not implemented: %s", where)

    oper = {"set": 1, "add": 2, "sub": 3, "mul": 4, "div": 5, "eli": 11}

    insidevalue = 0
    if inside:
        insidevalue = 1

    logger.info("Operations of points inside polygon(s)...")
    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError("The poly input is not a single Polygons instance")

    idgroups = poly.get_dataframe(copy=False).groupby(poly.pname)

    xcor = self.get_dataframe(copy=False)[self.xname].values
    ycor = self.get_dataframe(copy=False)[self.yname].values
    zcor = self.get_dataframe(copy=False)[self.zname].values

    usepoly = False
    if isinstance(value, str) and value == "poly":
        usepoly = True

    for id_, grp in idgroups:
        pxcor = grp[poly.xname].values
        pycor = grp[poly.yname].values
        pvalue = value
        pvalue = grp[poly.zname].values.mean() if usepoly else value

        logger.info("C function for polygon %s...", id_)

        ies = _cxtgeo.pol_do_points_inside(
            xcor, ycor, zcor, pxcor, pycor, pvalue, oper[opname], insidevalue
        )
        logger.info("C function for polygon %s... done", id_)

        if ies != 0:
            raise RuntimeError(f"Something went wrong, code {ies}")

    zcor[zcor > UNDEF_LIMIT] = np.nan
    dataframe = self.get_dataframe()
    dataframe[self.zname] = zcor
    # removing rows where Z column is undefined
    dataframe.dropna(how="any", subset=[self.zname], inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    self.set_dataframe(dataframe)
    logger.info("Operations of points inside polygon(s)... done")


def operation_polygons_v2(self, poly, value, opname="add", inside=True, where=True):
    """
    Operations re restricted to closed polygons, for points or polyline points.

    This (from v1) is NOT implemented here: "If value is not float but 'poly', then the
    avg of each polygon Z value will be used instead."

    NOTE! Both 'inside' and "outside" several polygons will become a union. Hence v2
    it will TREAT OUTSIDE POINTS DIFFERENT than v1!

    The "where" filter is reserved for future use.

    The version v2 uses mark_in_polygons_mpl(), and should be much faster.
    """
    logger.info("Operations of points inside polygon(s), version 2")

    allowed_opname = ("set", "add", "sub", "mul", "div", "eli")

    logger.warning("Where is not implemented: %s", where)
    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError("The poly input is not a Polygons instance")

    tmp = self.copy()
    ivalue = 1 if inside else 0
    ovalue = 0 if inside else 1
    mark_in_polygons_mpl(tmp, poly, "_TMP", inside_value=ivalue, outside_value=ovalue)
    tmpdf = tmp.get_dataframe()

    dataframe = self.get_dataframe()

    if opname == "add":
        dataframe.loc[tmpdf._TMP == 1, self.zname] += value
    elif opname == "sub":
        dataframe.loc[tmpdf._TMP == 1, self.zname] -= value
    elif opname == "mul":
        dataframe.loc[tmpdf._TMP == 1, self.zname] *= value
    elif opname == "div":
        if value != 0.0:
            dataframe.loc[tmpdf._TMP == 1, self.zname] /= value
        else:
            dataframe.loc[tmpdf._TMP == 1, self.zname] = 0.0
    elif opname == "set":
        dataframe.loc[tmpdf._TMP == 1, self.zname] = value
    elif opname == "eli":
        dataframe = dataframe[tmpdf._TMP == 0]
        dataframe.reset_index(inplace=True, drop=True)
    else:
        raise KeyError(f"The opname={opname} is not one of {allowed_opname}")

    self.set_dataframe(dataframe)

    logger.info("Operations of points inside polygon(s)... done")


def rescale_polygons(self, distance=10, addlen=False, kind="simple", mode2d=False):
    """Rescale (resample) a polygons segment
    Default settings will make it backwards compatible with 2.0
    New options were added in 2.1:
    * addlen
    * kind
    * mode2d
    """

    kind = "simple" if kind is None else kind

    kinds_allowed = ("simple", "slinear", "cubic")
    if kind not in kinds_allowed:
        raise ValueError(
            f"The input 'kind' {kind} is not valid. Allowed: {kinds_allowed}"
        )

    if kind in ("slinear", "cubic"):
        _rescale_v2(self, distance, addlen, kind=kind, mode2d=mode2d)

    else:
        _rescale_v1(self, distance, addlen, mode2d=mode2d)


def _rescale_v1(self, distance, addlen, mode2d):
    # version 1, simple approach, will rescale in 2D since Shapely use 2D lengths
    if not mode2d:
        raise KeyError("Cannot combine 'simple' with mode2d False")

    idgroups = self.get_dataframe(copy=False).groupby(self.pname)

    dfrlist = []
    for idx, grp in idgroups:
        if len(grp.index) < 2:
            logger.warning("Cannot rescale polygons with less than two points. Skip")
            continue

        pxcor = grp[self.xname].values
        pycor = grp[self.yname].values
        pzcor = grp[self.zname].values
        spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))

        new_spoly = _redistribute_vertices(spoly, distance)
        dfr = pd.DataFrame(
            np.array(new_spoly.coords), columns=[self.xname, self.yname, self.zname]
        )

        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.set_dataframe(dfr.reset_index(drop=True))

    if addlen:
        self.hlen()
        self.tlen()


def _redistribute_vertices(geom, distance):
    """Local function to interpolate in a polyline using Shapely"""
    if geom.geom_type == "LineString":
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return sg.LineString(
            [
                geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )

    if geom.geom_type == "MultiLineString":
        parts = [_redistribute_vertices(part, distance) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])

    raise ValueError(f"Unhandled geometry {geom.geom_type}")


def _handle_vertical_input(self, inframe):
    """Treat the special vertical case.

    A special case occurs for e.g. a 100% vertical well. Here the trick is distort
    first and last row with a relative small number so that numerical problems are
    avoided.
    """

    result = inframe.copy().reset_index()

    # e.g. tvd.mean is 2000, then tolerance will be 0.002, and edit will be 2
    tolerance = self.get_dataframe(copy=False)[self.zname].mean() * 0.000001
    edit = tolerance * 1000
    pseudo = self.copy()

    if inframe[self.dhname].max() < tolerance:
        result.at[0, self.xname] -= edit
        result.at[result.index[-1], self.xname] += edit
        pseudo.set_dataframe(result)
        pseudo.hlen()
        pseudo.tlen()
        result = pseudo.get_dataframe()

    return result


def _rescale_v2(self, distance, addlen, kind="slinear", mode2d=True):
    # Rescaling to constant increment is perhaps impossible, but this is
    # perhaps quite close

    self.hlen()
    self.tlen()

    idgroups = self.get_dataframe(copy=False).groupby(self.pname)

    dfrlist = []
    for idx, grp in idgroups:
        grp = _handle_vertical_input(self, grp)

        # avoid duplicates when *_DELTALEN are 0.0 (makes scipy interp1d() fail
        # when scipy >= 1.9)
        grp = grp.drop(grp[(grp[self.dhname] == 0.0) | (grp[self.dtname] == 0.0)].index)

        if len(grp.index) < 2:
            logger.warning("Cannot rescale polygons with less than two points. Skip")
            continue

        points = [grp[self.xname], grp[self.yname], grp[self.zname]]

        leng = grp[self.hname].iloc[-1]
        gname = self.hname

        if not mode2d:
            leng = grp[self.tname].iloc[-1]
            gname = self.tname

        # to avoid numerical trouble of pure vertical sections
        leng = leng - 0.001 * leng

        nstep = int(leng / distance)
        alpha = np.linspace(0, leng, num=nstep, endpoint=True)

        if kind == "slinear":
            splines = [UnivariateSpline(grp[gname], crd, k=1, s=0) for crd in points]
        elif kind == "cubic":
            splines = [UnivariateSpline(grp[gname], crd) for crd in points]

        else:
            raise ValueError(f"Invalid kind chosen: {kind}")

        ip = np.vstack([spl(alpha) for spl in splines]).T

        dfr = pd.DataFrame(np.array(ip), columns=[self.xname, self.yname, self.zname])

        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.set_dataframe(dfr.reset_index(drop=True))

    if addlen:
        self.tlen()
        self.hlen()
    else:
        self.delete_columns([self.hname, self.dhname, self.tname, self.dtname])


def get_fence(
    self, distance=20, atleast=5, nextend=2, name=None, asnumpy=True, polyid=None
):
    """Get a fence suitable for plotting xsections, either as a numpy or as a
    new Polygons instance.

    The atleast parameter will win over the distance, meaning that if total length
    horizontally is 50, and distance is set to 20, the actual length will be 50/5=10
    In such cases, nextend will be modified automatically also to fulfill the original
    intention of nextend*distance (approx).

    The routine is still not perfect for "close to very vertical polygon"
    but assumed to be sufficient for all practical cases

    """
    if atleast < 3:
        raise ValueError("The atleast key must be 3 or greater")

    orig_extend = nextend * distance
    orig_distance = distance

    fence = self.copy()

    fence.hlen()

    if len(fence.get_dataframe(copy=False)) < 2:
        xtg.warn(f"Too few points in polygons for fence, return False (name: {name})")
        return False

    fence.filter_byid(polyid)

    hxlen = fence.get_shapely_objects()[0].length

    # perhaps a way to treat very vertical polys from e.g. wells:
    if hxlen < 0.1 * orig_distance:
        hxlen = 0.1 * orig_distance

    if hxlen / (atleast + 1) < orig_distance:
        distance = hxlen / (atleast + 1)

    fence_keep = fence.copy()
    fence.rescale(distance, kind="slinear", mode2d=True)

    if len(fence.get_dataframe(copy=False)) < 2:
        fence = fence_keep

    fence.hlen()
    updated_distance = fence.get_dataframe(copy=False)[fence.dhname].median()

    if updated_distance < 0.5 * distance:
        updated_distance = 0.5 * distance

    newnextend = int(round(orig_extend / updated_distance))
    fence.extend(updated_distance, nsamples=newnextend)

    df = fence.get_dataframe()
    df0 = df.drop(df.index[1:])  # keep always first which has per def H_DELTALEN=0
    df2 = df[updated_distance * 0.01 < df.H_DELTALEN]  # skip very close points
    df = pd.concat([df0, df2], axis=0, ignore_index=True)

    # duplicates may still exist; skip those
    df.drop_duplicates(subset=[fence.xname, fence.yname], keep="first", inplace=True)
    df.reset_index(inplace=True, drop=True)
    fence.set_dataframe(df)

    if name:
        fence.name = name

    if asnumpy is True:
        rval = np.concatenate(
            (
                df[fence.xname].values,
                df[fence.yname].values,
                df[fence.zname].values,
                df[fence.hname].values,
                df[fence.dhname].values,
            ),
            axis=0,
        )
        return np.reshape(rval, (fence.nrow, 5), order="F")

    return fence


def snap_surface(self, surf, activeonly=True):
    """Snap (or transfer) operation.

    Points that falls outside the surface will be UNDEF, and they will be removed
    if activeonly. Otherwise, the old values will be kept.
    """

    if not isinstance(surf, xtgeo.RegularSurface):
        raise ValueError("Input object of wrong data type, must be RegularSurface")

    dataframe = self.get_dataframe()
    zval = dataframe[self.zname].values

    ier = _cxtgeo.surf_get_zv_from_xyv(
        dataframe[self.xname].values,
        dataframe[self.yname].values,
        zval,
        surf.ncol,
        surf.nrow,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.yflip,
        surf.rotation,
        surf.get_values1d(),
        0,
    )

    if ier != 0:
        raise RuntimeError(f"Error code from C routine surf_get_zv_from_xyv is {ier}")
    if activeonly:
        dataframe[self.zname] = zval
        dataframe = dataframe[dataframe[self.zname] < xtgeo.UNDEF_LIMIT]
        dataframe.reset_index(inplace=True, drop=True)
    else:
        out = np.where(
            zval < xtgeo.UNDEF_LIMIT, zval, self.get_dataframe()[self.zname].values
        )
        dataframe[self.zname] = out

    self.set_dataframe(dataframe)


def hlen(self, hname="H_CUMLEN", dhname="H_DELTALEN", atindex=0):
    """Get the horizontal distance (cumulative and delta) between points in polygons."""

    _generic_length(self, gname=hname, dgname=dhname, atindex=atindex, mode2d=True)


def tlen(self, tname="T_CUMLEN", dtname="T_DELTALEN", atindex=0):
    """Get the true 3D distance (cumulative and delta) between points in polygons."""

    _generic_length(self, gname=tname, dgname=dtname, atindex=atindex, mode2d=False)


def _generic_length(
    self, gname="G_CUMLEN", dgname="G_DELTALEN", atindex=0, mode2d=True
):
    """Get the true or horizontal distance (cum/delta) between points in polygons.

    The properties gname and ghname will be updated.

    Note that Dxx at first location will be set equal to that of location 1
    """

    # Potential todo: Add an option that dH never gets 0.0 to avoid numerical trouble
    # for e.g. rescale?

    if not isinstance(self, xtgeo.Polygons):
        raise ValueError("Input object of wrong data type, must be Polygons")

    # delete existing self.hname and self.dhname columns
    self.delete_columns([gname, dgname])

    idgroups = self.get_dataframe(copy=False).groupby(self.pname)

    gdist = np.array([])
    dgdist = np.array([])
    for _id, grp in idgroups:
        ier, tlenv, dtlenv, hlenv, dhlenv = _cxtgeo.pol_geometrics(
            grp[self.xname].values.astype(np.float64),
            grp[self.yname].values.astype(np.float64),
            grp[self.zname].values.astype(np.float64),
            len(grp),
            len(grp),
            len(grp),
            len(grp),
        )
        if ier != 0:
            raise RuntimeError(f"Error code from _cxtgeo.pol_geometrics is {ier}")

        if mode2d:
            dhlenv[0] = dhlenv[1]
            if atindex > 0:
                cumval = hlenv[atindex]
                hlenv -= cumval

            gdist = np.append(gdist, hlenv)
            dgdist = np.append(dgdist, dhlenv)

        else:
            dtlenv[0] = dtlenv[1]
            if atindex > 0:
                cumval = tlenv[atindex]
                tlenv -= cumval

            gdist = np.append(gdist, tlenv)
            dgdist = np.append(dgdist, dtlenv)

    dataframe = self.get_dataframe()
    dataframe[gname] = gdist
    dataframe[dgname] = dgdist
    self.set_dataframe(dataframe)

    if mode2d:
        self._hname = gname
        self._dhname = dgname
    else:
        self._tname = gname
        self._dtname = dgname


def extend(self, distance, nsamples, addhlen=True):
    """Extend polygon by distance, nsamples times.

    It is default to recompute HLEN from nsamples.
    """

    if not isinstance(self, xtgeo.Polygons):
        raise ValueError("Input object of wrong data type, must be Polygons")

    dataframe = self.get_dataframe()
    for _ in range(nsamples):
        # beginning of poly
        row0 = dataframe.iloc[0]
        row1 = dataframe.iloc[1]

        rown = row0.copy()

        # setting row0[2] as row1[2] is intentional, as this shall be a 2D lenght!
        ier, newx, newy, _ = _cxtgeo.x_vector_linint2(
            row1.iloc[0],
            row1.iloc[1],
            row1.iloc[2],
            row0.iloc[0],
            row0.iloc[1],
            row1.iloc[2],
            distance,
            12,
        )

        if ier != 0:
            raise RuntimeError(f"Error code from _cxtgeo.x_vector_linint2 is {ier}")

        rown[self.xname] = newx
        rown[self.yname] = newy

        df_to_add = rown.to_frame().T
        df_to_add = df_to_add.astype(dataframe.dtypes.to_dict())  # ensure same dtypes

        dataframe = pd.concat([df_to_add, dataframe]).reset_index(drop=True)

        # end of poly
        row0 = dataframe.iloc[-2]
        row1 = dataframe.iloc[-1]

        rown = row1.copy()

        # setting row1[2] as row0[2] is intentional, as this shall be a 2D lenght!
        ier, newx, newy, _ = _cxtgeo.x_vector_linint2(
            row0.iloc[0],
            row0.iloc[1],
            row0.iloc[2],
            row1.iloc[0],
            row1.iloc[1],
            row0.iloc[2],
            distance,
            11,
        )

        rown[self.xname] = newx
        rown[self.yname] = newy

        df_to_add = rown.to_frame().T
        df_to_add = df_to_add.astype(dataframe.dtypes.to_dict())  # ensure same dtypes
        dataframe = pd.concat([dataframe, df_to_add]).reset_index(drop=True)

    self.set_dataframe(dataframe)

    if addhlen:
        self.hlen(atindex=nsamples)
