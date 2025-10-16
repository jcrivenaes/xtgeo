"""Loading rmsapi package based on availability.

As xtgeo from version 4 only support RMSAPI > 1.10, we do not support roxar
package anymore. This module handles the optional import of the rmsapi package
and its submodules, as well as type annotations for rmsapi classes.

"""

import importlib
from typing import TYPE_CHECKING, Any, Optional, Union


def load_package(package_name: str) -> Optional[Any]:
    """Load a Python package by name, return None if not available."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


# Load main packages
rmsapi = load_package("rmsapi")
# Load internal packages
_rmsapi = load_package("_rmsapi")

# Load submodules from the available package
rmsapi_well_picks = load_package("rmsapi.well_picks")
rmsapi_jobs = load_package("rmsapi.jobs")
rmsapi_grids = load_package("rmsapi.grids")

# Explicitly type the roxar* as Optional to indicate it may be None
# roxar: Optional[Any] = roxar
# _roxar: Optional[Any] = _roxar
# roxar_grids: Optional[Any] = roxar_grids
# roxar_well_picks: Optional[Any] = roxar_well_picks
# roxar_jobs: Optional[Any] = roxar_jobs

# Create RoxarType namespace for type annotations
if TYPE_CHECKING:
    import pathlib

    try:
        from rmsapi import Project as RmsProject
        from rmsapi.grids import Grid3D as RmsGrid3D
        from rmsapi.jobs import Jobs as RmsJobs
        from rmsapi.well_picks import (
            WellPick,  # noqa: F401 - imported for type checking
            WellPickAttribute,  # noqa: F401 - imported for type checking
            WellPickAttributeType,  # noqa: F401 - imported for type checking
            WellPicks as RmsWellPicks,  # noqa: F401 - imported for type checking
            WellPickSet,  # noqa: F401 - imported for type checking
            WellPickType,  # noqa: F401 - imported for type checking
        )
    except ImportError:
        # Fallback when rmsapi is not available
        RmsProject = Any  # type: ignore[misc,assignment]
        RmsGrid3D = Any  # type: ignore[misc,assignment]
        RmsJobs = Any  # type: ignore[misc,assignment]
        RmsWellPicks = Any  # type: ignore[misc,assignment]
        WellPickType = Any  # type: ignore[misc,assignment]
        WellPickAttribute = Any  # type: ignore[misc,assignment]
        WellPickAttributeType = Any  # type: ignore[misc,assignment]
        WellPick = Any  # type: ignore[misc,assignment]
        WellPickSet = Any  # type: ignore[misc,assignment]

    # Type aliases
    RmsProjectType = Union[str, pathlib.Path, RmsProject]
    RmsGrid3DType = RmsGrid3D

else:
    # Runtime fallback - only RmsProjectType is actually used by other modules
    RmsProjectType = Union[str, object]  # type: ignore[misc]
