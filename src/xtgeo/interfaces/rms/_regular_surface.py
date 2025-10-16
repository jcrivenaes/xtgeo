"""RMS API functions for RegularSurface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms._rms_base import _BaseRmsData
from xtgeo.interfaces.rms._rmsapi_package import rmsapi
from xtgeo.interfaces.rms.rmsapi_utils import (
    RmsApiUtils,
    _DomainTypeClipBoardGeneral2D as DomainType,
    _StorageTypeRegularSurface as StorageType,
)

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.interfaces.rms._rmsapi_package import RmsProjectType
else:
    RmsProjectType = Union[str, object]


assert rmsapi is not None, "RMS API is required for RMS interface."


@dataclass(frozen=True)
class RegularSurfaceDataRms:
    """Immutable data container for regular surface information."""

    name: str
    xori: float
    yori: float
    ncol: int
    nrow: int
    xinc: float
    yinc: float
    rotation: float
    values: np.ma.MaskedArray


class RegularSurfaceReader(_BaseRmsData):
    """Handles reading (loading) regular surface data from RMS projects.

    This class handles all the complex loading logic, validation, and cleanup.
    """

    def __init__(
        self,
        project: RmsProjectType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
    ):
        super().__init__(project, name, category, stype, realisation)

    def load(self) -> RegularSurfaceDataRms:
        """Load surface data from RMS and return immutable data object."""
        try:
            return self._perform_load()
        except Exception as exc:
            self._cleanup()
            raise RuntimeError(f"Failed to load surface '{self.name}': {exc}") from exc

    def _perform_load(self) -> RegularSurfaceDataRms:
        """Internal method to handle the actual loading."""
        # Initialize RMS API Utils
        self._rmsapi_utils = RmsApiUtils(self.project, readonly=True)

        # Validate and convert stype
        self._stype_enum = self._validate_inputs()

        # Extract surface data
        surface_data = self._extract_surface_from_rms()

        # Cleanup
        self._cleanup()

        return surface_data

    def _extract_surface_from_rms(self) -> RegularSurfaceDataRms:
        """Extract surface data from RMS project and create data object."""
        assert self._rmsapi_utils is not None
        assert rmsapi is not None

        proj = self._rmsapi_utils.project
        rms_item = None

        if self._stype_enum in (StorageType.HORIZONS, StorageType.ZONES):
            rms_item = self._get_horizon_zone_item(proj)
        elif self._stype_enum in (StorageType.CLIPBOARD, StorageType.GENERAL2D_DATA):
            rms_item = self._get_clipboard_general2d_item(proj)
        elif self._stype_enum == StorageType.TRENDS:
            rms_item = self._get_trends_item(proj)
        else:
            raise ValueError(f"Unsupported storage type: {self._stype_enum}")

        # Validate surface object
        if not isinstance(rms_item, rmsapi.Surface):
            raise TypeError(
                f"Expected a Surface for '{self.name}', but got "
                f"{type(rms_item).__name__} for category '{self.category}'. "
                "Check that the item requested is indeed a surface."
            )

        # Extract grid data
        try:
            rmssurf = rms_item.get_grid(self.realisation)
            return self._create_surface_data(rmssurf)
        except KeyError as exc:
            raise RuntimeError(
                f"Could not load surface '{self.name}' from RMS API. "
                f"Realisation '{self.realisation}' may not exist."
            ) from exc

    def _create_surface_data(
        self,
        rms_surface: Any,  # rmsapi.Surface
    ) -> RegularSurfaceDataRms:
        """Create RegularSurfaceData from RMSAPI surface."""
        logger.info("Creating surface data from RMSAPI surface")

        return RegularSurfaceDataRms(
            name=self.name,
            xori=rms_surface.origin[0],
            yori=rms_surface.origin[1],
            ncol=rms_surface.dimensions[0],
            nrow=rms_surface.dimensions[1],
            xinc=rms_surface.increment[0],
            yinc=rms_surface.increment[1],
            rotation=rms_surface.rotation,
            values=rms_surface.get_values(),  # always np.float64 in RMS API
        )

    def _get_horizon_zone_item(self, proj: RmsProjectType) -> Any:  # rmsapi.Surface
        """Get surface item from horizons or zones."""
        assert proj is not None
        assert rmsapi is not None
        assert isinstance(proj, rmsapi.Project), "Project must be initialized"
        assert self._stype_enum is not None, "stype must be set"

        container = (
            proj.horizons if self._stype_enum == StorageType.HORIZONS else proj.zones
        )
        container_name = (
            "Horizons" if self._stype_enum == StorageType.HORIZONS else "Zones"
        )

        if self.name not in container:
            raise ValueError(f"Name '{self.name}' is not within {container_name}")
        if self.category not in container.representations:
            raise ValueError(
                f"Category '{self.category}' is not within {container_name} categories"
            )

        return container[self.name][self.category]

    def _get_clipboard_general2d_item(
        self, proj: RmsProjectType
    ) -> Any:  # rmsapi.Surface
        """Get surface item from clipboard or general2d_data."""
        assert rmsapi is not None
        assert isinstance(proj, rmsapi.Project), "Project must be initialized"
        assert self._stype_enum is not None, "stype must be set"

        try:
            container = getattr(proj, self._stype_enum.value)
            if self.category:
                folders = self.category.split("|" if "|" in self.category else "/")
                return container.folders[folders][self.name]
            return container[self.name]
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Could not access '{self.name}' in {self._stype_enum.value}"
                + (f" with category '{self.category}'" if self.category else "")
            ) from exc

    def _get_trends_item(self, proj: RmsProjectType) -> Any:  # rmsapi.Surface
        """Get surface item from trends."""
        assert rmsapi is not None
        assert isinstance(proj, rmsapi.Project), "Project must be initialized"

        if self.name not in proj.trends.surfaces:
            raise ValueError(f"Name '{self.name}' is not within Trends")
        return proj.trends.surfaces[self.name]


class RegularSurfaceWriter(_BaseRmsData):
    """Handles writing regular surface data to RMS API."""

    def __init__(
        self,
        project: RmsProjectType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
        domain: Literal["time", "depth", "unknown"] = "depth",  # clipboard/general2d
    ):
        super().__init__(project, name, category, stype, realisation)
        self.domain = domain.lower()

    def save(self, data: RegularSurfaceDataRms) -> None:
        """Write surface data to RMS."""
        try:
            self._perform_save(data)
        except Exception as exc:
            self._cleanup()
            raise RuntimeError(f"Failed to save surface '{self.name}': {exc}") from exc
        finally:
            self._cleanup()

    def _perform_save(self, data: RegularSurfaceDataRms) -> None:
        # Initialize RMS API Utils
        self._rmsapi_utils = RmsApiUtils(self.project, readonly=False)

        self._check_valid_domain()

        # Validate and convert stype
        self._stype_enum = self._validate_inputs()

        # Validate payload consistency
        self._validate_payload(data)

        # Do the write
        self._write_to_rms(data)

        # Save project if external
        assert rmsapi is not None
        proj = self._rmsapi_utils.project
        assert isinstance(proj, rmsapi.Project), "Project must be initialized"
        if getattr(self._rmsapi_utils, "_roxexternal", False):
            proj.save()

    def _check_valid_domain(self) -> None:
        """Check that domain is valid for the given storage type."""
        if self.domain not in DomainType.values():
            raise ValueError(f"domain must be {DomainType.values()}")
        self._domain_enum = DomainType[self.domain.upper()]
        self._api_vertical_domain = self._resolve_api_vertical_domain()
        logger.debug("Using domain: %s", self._domain_enum)

    def _resolve_api_vertical_domain(self) -> Any:  # rmsapi.VerticalDomain
        """Map internal domain enum to RMS API VerticalDomain enum."""
        assert rmsapi is not None
        if self._domain_enum == DomainType.DEPTH:
            return rmsapi.VerticalDomain.depth
        if self._domain_enum == DomainType.TIME:
            return rmsapi.VerticalDomain.time
        return rmsapi.VerticalDomain.unknown

    @staticmethod
    def _validate_payload(data: RegularSurfaceDataRms) -> None:
        if data.ncol <= 0 or data.nrow <= 0:
            raise ValueError("ncol and nrow must be positive.")
        if data.xinc <= 0 or data.yinc <= 0:
            raise ValueError("xinc and yinc must be positive.")
        if not isinstance(data.values, np.ma.MaskedArray):
            # Normalize to MaskedArray
            data.values = np.ma.array(data.values)  # type: ignore[attr-defined]
        if data.values.shape != (data.ncol, data.nrow):
            raise ValueError(
                f"values shape {data.values.shape} does not match "
                f"(ncol, nrow)=({data.ncol}, {data.nrow})"
            )

    def _write_to_rms(self, data: RegularSurfaceDataRms) -> None:
        assert rmsapi is not None
        assert self._rmsapi_utils is not None, "RMS API utils must be initialized"
        proj = self._rmsapi_utils.project
        assert isinstance(proj, rmsapi.Project), "Project must be initialized"
        assert self._stype_enum is not None, "stype must be set"

        # Build grid geometry
        grid = rmsapi.RegularGrid2D.create(
            x_origin=data.xori,
            y_origin=data.yori,
            i_inc=data.xinc,
            j_inc=data.yinc,
            ni=data.ncol,
            nj=data.nrow,
            rotation=data.rotation,
        )

        # Prepare values: float64, finite data, respect mask
        values = self._sanitize_values_for_rms(data.values)

        if self._stype_enum in (StorageType.HORIZONS, StorageType.ZONES):
            container = (
                proj.horizons
                if self._stype_enum == StorageType.HORIZONS
                else proj.zones
            )
            container_name = (
                "Horizons" if self._stype_enum == StorageType.HORIZONS else "Zones"
            )

            if self.name not in container:
                raise ValueError(f"Name '{self.name}' is not within {container_name}")
            if self.category not in container.representations:
                raise ValueError(
                    f"Category '{self.category}' is not within {container_name} "
                    "categories"
                )

            root = container[self.name][self.category]
            grid.set_values(values)
            # Pass realisation where supported
            try:
                root.set_grid(grid, realisation=self.realisation)
            except TypeError:
                # Some APIs don't take realisation argument here
                root.set_grid(grid)

        elif self._stype_enum in (StorageType.CLIPBOARD, StorageType.GENERAL2D_DATA):
            styperef = getattr(proj, self._stype_enum.value)

            def _get_current_item() -> Any:  # rmsapi.Surface or container
                folders = []
                if self.category:
                    folders = self.category.split("|" if "|" in self.category else "/")
                    if folders:
                        styperef.folders.create(folders)
                current_item = styperef.folders[folders] if folders else styperef
                logger.debug("Current item: %s", current_item)
                logger.debug("Folders: %s", folders)
                return current_item, folders

            current_item, folders = _get_current_item()
            if self.name in current_item:
                root = current_item[self.name]
                if not isinstance(root, rmsapi.Surface):
                    raise TypeError(
                        f"Expected a Surface for '{self.name}', but got "
                        f"{type(root).__name__} for category '{self.category}'. "
                        "Check that the item requested is indeed a surface."
                    )
                logger.debug("Using domain: %s", self._domain_enum)
                current_domain = getattr(root, "vertical_domain", DomainType.UNKNOWN)
                if current_domain != self.domain:
                    # RMS API does not allow changing domain of existing surface but
                    # we can brute-force remove the current surface and create a new one
                    logger.debug("Force remove current: %s", current_item[self.name])
                    del current_item[self.name]
                    logger.debug("Creating new surface with domain: %s", self.domain)
                    current_item, folders = _get_current_item()
                    root = styperef.create_surface(
                        self.name, folders, self._api_vertical_domain
                    )
            else:
                root = styperef.create_surface(
                    self.name, folders, self._api_vertical_domain
                )
            grid.set_values(values)
            root.set_grid(grid)

        elif self._stype_enum == StorageType.TRENDS:
            styperef = getattr(proj, self._stype_enum.value)
            if self.name in styperef.surfaces:
                root = styperef.surfaces[self.name]
                if not isinstance(root, rmsapi.Surface):
                    raise TypeError(
                        f"Expected a Surface for '{self.name}', but got "
                        f"{type(root).__name__}. Check that the item requested is "
                        "indeed a surface."
                    )
            else:
                root = styperef.surfaces.create(self.name)

    @staticmethod
    def _sanitize_values_for_rms(
        values: np.ma.MaskedArray,
    ) -> np.ma.MaskedArray:
        vals = np.ma.array(values, dtype=np.float64, copy=False)
        # RMS API does not accept NaNs/Inf even behind the mask; replace in data buffer
        applied_fill_value = np.finfo(np.float64).max
        filled = np.ma.filled(vals, fill_value=applied_fill_value)
        filled = np.where(
            np.isnan(filled) | np.isinf(filled), applied_fill_value, filled
        )
        return np.ma.masked_equal(filled, applied_fill_value)
