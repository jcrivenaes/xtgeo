"""Loading rmsapi or roxar module based on availability"""

import importlib
from typing import Any, Optional

RoxarType = Any
RoxarTypeGrid3D = Any  # rmsapi.grids.Grid3D


def load_module(module_name: str) -> Optional[type]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


rmsapi = load_module("rmsapi")
roxar = rmsapi if rmsapi else load_module("roxar")

if rmsapi:
    roxar_well_picks = load_module("rmsapi.well_picks")
    roxar_jobs = load_module("rmsapi.jobs")
    roxar_grids = load_module("rmsapi.grids")
else:
    roxar_well_picks = load_module("roxar.well_picks")
    roxar_jobs = load_module("roxar.jobs")
    roxar_grids = load_module("roxar.grids")


print("XXXXXX roxar", roxar, roxar_grids)

# Explicitly type the roxar as Optional to indicate it may be None
roxar: Optional[RoxarType] = roxar
