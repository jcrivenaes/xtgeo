"""Shared base for RMS RegularSurface reader/writer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms.rmsapi_utils import (
    RmsApiUtils,
    _StorageTypeRegularSurface as StorageType,
)

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.interfaces.rms._rmsapi_package import RmsProjectType
else:
    RmsProjectType = Union[str, object]


class _BaseRmsData:
    """
    Common init, input validation etc for RMS RegularSurface, Points, Polygions I/O.
    """

    def __init__(
        self,
        project: RmsProjectType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
    ):
        self.project = project
        self.name = name
        self.category = category
        self.stype = stype
        self.realisation = realisation

        # Internal state
        self._rmsapi_utils: RmsApiUtils | None = None
        self._stype_enum: StorageType | None = None

    def _init_utils(self, readonly: bool) -> None:
        """Create RmsApiUtils with the selected readonly mode."""
        self._rmsapi_utils = RmsApiUtils(self.project, readonly=readonly)

    def _validate_inputs(self) -> StorageType:
        """Validate input parameters and return StorageType enum."""
        stype_lower = self.stype.lower()

        if stype_lower not in StorageType.values():
            raise ValueError(
                f"Given stype '{stype_lower}' is not supported. "
                f"Legal stypes are: {StorageType.values()}"
            )

        stype_enum = StorageType(stype_lower)

        if not self.name:
            raise ValueError("The name is missing or empty.")

        if (
            stype_enum in (StorageType.HORIZONS, StorageType.ZONES)
            and not self.category
        ):
            raise ValueError(
                "Need to specify both name and category for horizons and zones"
            )

        if stype_enum == StorageType.GENERAL2D_DATA:
            assert self._rmsapi_utils is not None  # initialized in _init_utils
            if not self._rmsapi_utils.version_required("1.6"):
                raise NotImplementedError(
                    f"API Support for general2d_data is missing in this RMS version "
                    f"(current API version is {self._rmsapi_utils.rmsversion} - "
                    "required is 1.6)"
                )

        return stype_enum

    def _cleanup(self) -> None:
        """Clean up RmsApiUtils instance."""
        if self._rmsapi_utils:
            self._rmsapi_utils.safe_close()
            self._rmsapi_utils = None
