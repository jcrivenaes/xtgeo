"""Well input and ouput, private module"""

import json
from copy import deepcopy

import numpy as np
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.log import null_logger
from xtgeo.metadata.metadata import MetaDataWell

logger = null_logger(__name__)


def import_rms_ascii(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
):
    """Import RMS ascii table well"""

    wlogtype = {}
    wlogrecords = {}

    xlognames_all = [
        _AttrName.XNAME.value,
        _AttrName.YNAME.value,
        _AttrName.ZNAME.value,
    ]
    xlognames = []

    lnum = 1
    with open(wfile.file, "r", encoding="UTF-8") as fwell:
        for line in fwell:
            if lnum == 1:
                _ffver = line.strip()  # noqa, file version
            elif lnum == 2:
                _wtype = line.strip()  # noqa, well type
            elif lnum == 3:
                # usually 4 fields, but last (rkb) can be missing. A
                # complication is that first field (well name) may have spaces,
                # hence some clever guessing is needed. However, this cannot be
                # 100% foolproof... if Ycoord < 1000 and last item of a well
                # name with spaces is a number, then this may fail.
                assume_rkb = False
                row = line.strip().split()
                newrow = []
                if len(row) > 3:
                    for item in row:
                        try:
                            item = float(item)
                        except ValueError:
                            item = str(item)
                        newrow.append(item)

                    if all(isinstance(var, float) for var in newrow[-3:]) and abs(
                        newrow[-1] < 1000.0
                    ):
                        assume_rkb = True

                rkb = float(row.pop()) if assume_rkb else None
                ypos = float(row.pop())
                xpos = float(row.pop())
                wname = " ".join(map(str, row))

            elif lnum == 4:
                nlogs = int(line)
                nlogread = 1
                logger.debug("Number of logs: %s", nlogs)

            else:
                row = line.strip().split()
                lname = row[0]

                # if i_index etc, make uppercase to I_INDEX
                # however it is most practical to treat indexes as CONT logs
                if "_index" in lname:
                    lname = lname.upper()

                ltype = row[1].upper()

                rxv = row[2:]

                xlognames_all.append(lname)
                xlognames.append(lname)

                wlogtype[lname] = ltype

                logger.debug("Reading log name %s of type %s", lname, ltype)

                if ltype == _AttrType.DISC.value:
                    xdict = {int(rxv[i]): rxv[i + 1] for i in range(0, len(rxv), 2)}
                    wlogrecords[lname] = xdict
                else:
                    wlogrecords[lname] = tuple(row[1:])

                nlogread += 1

                if nlogread > nlogs:
                    break

            lnum += 1

    # now import all logs as pandas framework

    dfr = pd.read_csv(
        wfile.file,
        sep=r"\s+",
        skiprows=lnum,
        header=None,
        names=xlognames_all,
        dtype=np.float64,
        na_values=-999,
    )

    # undef values have a high float number? or keep Nan?
    # df.fillna(Well.UNDEF, inplace=True)

    dfr = _trim_on_lognames(dfr, lognames, lognames_strict, wname)
    mdlogname, zonelogname = _check_special_logs(
        dfr, mdlogname, zonelogname, strict, wname
    )

    return {
        "wlogtypes": wlogtype,
        "wlogrecords": wlogrecords,
        "rkb": rkb,
        "xpos": xpos,
        "ypos": ypos,
        "wname": wname,
        "df": dfr,
        "mdlogname": mdlogname,
        "zonelogname": zonelogname,
    }


def _trim_on_lognames(dfr, lognames, lognames_strict, wname):
    """Reduce the dataframe based on provided list of lognames"""
    if lognames == "all":
        return dfr

    uselnames = [_AttrName.XNAME.value, _AttrName.YNAME.value, _AttrName.ZNAME.value]
    if isinstance(lognames, str):
        uselnames.append(lognames)
    elif isinstance(lognames, list):
        uselnames.extend(lognames)

    newdf = pd.DataFrame()
    for lname in uselnames:
        if lname in dfr.columns:
            newdf[lname] = dfr[lname]
        else:
            if lognames_strict:
                msg = f"Logname <{lname}> is not present for <{wname}>"
                msg += " (required log under condition lognames_strict=True)"
                raise ValueError(msg)

    return newdf


def _check_special_logs(dfr, mdlogname, zonelogname, strict, wname):
    """Check for MD log and Zonelog, if requested"""

    mname = mdlogname
    zname = zonelogname

    if mdlogname is not None and mdlogname not in dfr.columns:
        msg = (
            f"mdlogname={mdlogname} was requested but no such log found for "
            f"well {wname}"
        )
        mname = None
        if strict:
            raise ValueError(msg)

        logger.warning(msg)

    # check for zone log:
    if zonelogname is not None and zonelogname not in dfr.columns:
        msg = (
            f"zonelogname={zonelogname} was requested but no such log found "
            f"for well {wname}"
        )
        zname = None
        if strict:
            raise ValueError(msg)

        logger.warning(msg)

    return mname, zname


def export_rms_ascii(self, wfile, precision=4):
    """Export to RMS well format."""
    with open(wfile, "w", encoding="utf-8") as fwell:
        print("1.0", file=fwell)
        print("Unknown", file=fwell)
        if self._rkb is None:
            print(f"{self._wname} {self._xpos} {self._ypos}", file=fwell)
        else:
            print(
                f"{self._wname} {self._xpos} {self._ypos} {self._rkb}",
                file=fwell,
            )
        print(f"{len(self.lognames)}", file=fwell)
        for lname in self.lognames:
            usewrec = "linear"
            wrec = []
            if isinstance(self.wlogrecords[lname], dict):
                for key in self.wlogrecords[lname]:
                    wrec.append(key)
                    wrec.append(self.wlogrecords[lname][key])
                usewrec = " ".join(str(x) for x in wrec)

            print(f"{lname} {self.get_logtype(lname)} {usewrec}", file=fwell)

    # now export all logs as pandas framework
    tmpdf = self._wdata.data.copy().fillna(value=-999)

    # make the disc as is np.int
    for lname in self.wlogtypes:
        if self.wlogtypes[lname] == _AttrType.DISC.value:
            tmpdf[[lname]] = tmpdf[[lname]].fillna(-999).astype(int)

    cformat = "%-." + str(precision) + "f"
    tmpdf.to_csv(
        wfile,
        sep=" ",
        header=False,
        index=False,
        float_format=cformat,
        escapechar="\\",
        mode="a",
    )


def export_hdf5_well(self, wfile, compression="lzf"):
    """Save to HDF5 format."""
    logger.debug("Export to hdf5 format...")

    self._ensure_consistency()

    self.metadata.required = self

    meta = self.metadata.get_metadata()
    jmeta = json.dumps(meta)

    complib = "zlib"  # same as default lzf
    complevel = 5
    if compression and compression == "blosc":
        complib = "blosc"
    else:
        complevel = 0

    with pd.HDFStore(wfile.file, "w", complevel=complevel, complib=complib) as store:
        logger.debug("export to HDF5 %s", wfile.name)
        store.put("Well", self._wdata.data)
        store.get_storer("Well").attrs["metadata"] = jmeta
        store.get_storer("Well").attrs["provider"] = "xtgeo"
        store.get_storer("Well").attrs["format_idcode"] = 1401

    logger.debug("Export to hdf5 format... done!")


def import_wlogs(wlogs: dict):
    """
    This converts joined wlogtypes/wlogrecords such as found in
    the hdf5 format to the format used in the Well object.

    >>> import_wlogs(dict())
    {'wlogtypes': {}, 'wlogrecords': {}}
    >>> import_wlogs(dict([("X_UTME", ("CONT", None))]))
    {'wlogtypes': {'X_UTME': 'CONT'}, 'wlogrecords': {'X_UTME': None}}

    Returns:
        dictionary with "wlogtypes" and "wlogrecords" as keys
        and corresponding values.
    """
    wlogtypes = {}
    wlogrecords = {}
    for key in wlogs:
        typ, rec = wlogs[key]

        if typ in {_AttrType.DISC.value, _AttrType.CONT.value}:
            wlogtypes[key] = deepcopy(typ)
        else:
            raise ValueError(f"Invalid log type found in input: {typ}")

        if rec is None or isinstance(rec, dict):
            wlogrecords[key] = deepcopy(rec)
        else:
            raise ValueError(f"Invalid log record found in input: {rec}")
    return {"wlogtypes": wlogtypes, "wlogrecords": wlogrecords}


def import_hdf5_well(wfile, **kwargs):
    """Load from HDF5 format."""
    logger.debug("The kwargs may be unused: %s", kwargs)
    reqattrs = MetaDataWell.REQUIRED

    with pd.HDFStore(wfile.file, "r") as store:
        data = store.get("Well")
        wstore = store.get_storer("Well")
        jmeta = wstore.attrs["metadata"]
        # provider = wstore.attrs["provider"]
        # format_idcode = wstore.attrs["format_idcode"]

    if isinstance(jmeta, bytes):
        jmeta = jmeta.decode()

    meta = json.loads(jmeta, object_pairs_hook=dict)
    req = meta["_required_"]
    result = {}
    for myattr in reqattrs:
        if myattr == "wlogs":
            result.update(import_wlogs(req[myattr]))
        elif myattr == "name":
            result["wname"] = req[myattr]
        else:
            result[myattr] = req[myattr]

    result["df"] = data
    return result
