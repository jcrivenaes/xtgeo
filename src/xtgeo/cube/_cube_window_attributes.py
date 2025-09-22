"""Attributes for a Cube to maps (surfaces), slice an interval, in pure numpy."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.interpolate import make_interp_spline

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.surface.regular_surface import RegularSurface

    from . import Cube

logger = null_logger(__name__)


STAT_ATTRS: Final = [
    "min",
    "max",
    "mean",
    "var",
    "rms",
    "maxpos",
    "maxneg",
    "maxabs",
    "meanpos",
    "meanneg",
    "meanabs",
]
SUM_ATTRS: Final = [
    "sumpos",
    "sumneg",
    "sumabs",
]


@dataclass
class CubeAttrs:
    """Internal class for computing attributes in window between two surfaces."""

    # This is the former implemantation which is now made hidden for the end user.
    # However, keep this for a while for small while (say mid 2026) for test purposes
    # (benchmarking) and emergency fall-back.

    cube: Cube
    upper_surface: RegularSurface | float | int
    lower_surface: RegularSurface | float | int
    ndiv: int = 10
    interpolation: str = "cubic"  # cf. scipy's make_interp_spline() when k=3
    minimum_thickness: float = 0.0

    # internal attributes
    _template_surface: RegularSurface | None = None
    _depth_array: np.ndarray | None = None
    _outside_depth: float | None = None  # detected and updated from the depth cube
    _reduced_cube: Cube = None
    _reduced_depth_array: np.ndarray | None = None
    _refined_cube: Cube | None = None
    _refined_depth_array: np.ndarray | None = None

    _upper: RegularSurface | None = None  # upper surf, resampled to cube map resolution
    _lower: RegularSurface | None = None  # lower surf, resampled to cube map resolution
    _min_thickness_mask: RegularSurface | None = None  # mask for min. thickness trunc.
    _mask_map_by_traceidcode: RegularSurface | None = None  # mask for traceidcode 2

    _result_attr_maps: dict = field(default_factory=dict)  # holds the resulting maps

    def __post_init__(self) -> None:
        self._process_upper_lower_surface()
        self._create_depth_array()
        self._create_reduced_cube()
        self._refine_interpolate()
        self._depth_mask()
        self._compute_statistical_attribute_surfaces()

    def result(self) -> dict[RegularSurface]:
        # return the resulting attribute maps
        return self._result_attr_maps

    def _process_upper_lower_surface(self) -> None:
        """Extract upper and lower surface, sampled to cube resolution."""

        from xtgeo import surface_from_cube  # avoid circular import by having this here

        logger.debug("Process upper and lower surface...")

        upper = (
            surface_from_cube(self.cube, self.upper_surface)
            if isinstance(self.upper_surface, (float, int))
            else self.upper_surface
        )
        lower = (
            surface_from_cube(self.cube, self.lower_surface)
            if isinstance(self.lower_surface, (float, int))
            else self.lower_surface
        )

        # the template surface is the topology that defines the resulting attribute maps
        self._template_surface = (
            upper
            if isinstance(self.upper_surface, (float, int))
            else self.upper_surface
        )

        # determine which of "this" and "other" is actually upper and lower
        if (lower - upper).values.mean() < 0:
            raise ValueError(
                "The upper surface is below the lower surface. "
                "Please provide the surfaces in the correct order."
            )

        # although not an attribute, we store the upper and lower surfaces
        self._result_attr_maps["upper"] = upper
        self._result_attr_maps["lower"] = lower

        # get the surfaces on cube resolution
        self._upper = surface_from_cube(self.cube, self.cube.zori)
        self._lower = surface_from_cube(self.cube, self.cube.zori)
        self._upper.resample(upper)
        self._lower.resample(lower)

        self._upper.fill()
        self._lower.fill()

        self._min_thickness_mask = self._lower - self._upper

        self._min_thickness_mask.values = np.where(
            self._min_thickness_mask.values <= self.minimum_thickness, 0, 1
        )
        if np.all(self._min_thickness_mask.values == 0):
            raise ValueError(
                "The minimum thickness is too large, no valid data in the interval. "
                "Perhaps surfaces are overlapping?"
            )
        logger.debug("Process upper and lower surface... done")

    def _create_depth_array(self) -> None:
        """Create a 1D array where values are cube depths; to be used as filter.

        Belowe and above the input surfaces (plus a buffer), the values are set to
        a constant value self._outside_depth.

        Will also issue warnings or errors if the surfaces are outside the cube,
        depending on severity.
        """
        logger.debug("Create depth array...")

        self._depth_array = np.array(
            [
                self.cube.zori + n * self.cube.zinc
                for n in range(self.cube.values.shape[2])
            ]
        ).astype(np.float32)

        # check that surfaces are within the cube
        if self._upper.values.min() > self._depth_array.max():
            raise ValueError("Upper surface is fully below the cube")
        if self._lower.values.max() < self._depth_array.min():
            raise ValueError("Lower surface is fully above the cube")
        if self._upper.values.max() < self._depth_array.min():
            warnings.warn("Upper surface is fully above the cube", UserWarning)
        if self._lower.values.min() > self._depth_array.max():
            warnings.warn("Lower surface is fully below the cube", UserWarning)

        self._outside_depth = self._depth_array.max() + 1

        add_extra_depth = 2 * self.cube.zinc  # add buffer on upper/lower edges

        self._depth_array = np.where(
            self._depth_array < self._upper.values.min() - add_extra_depth,
            self._outside_depth,
            self._depth_array,
        )

        self._depth_array = np.where(
            self._depth_array > self._lower.values.max() + add_extra_depth,
            self._outside_depth,
            self._depth_array,
        )
        logger.debug("Create depth array... done")

    def _create_reduced_cube(self) -> None:
        """Create a smaller cube based on the depth cube filter.

        The purpose is to limit the computation to the relevant volume, to save
        CPU time. I.e. cube values above the upper surface and below the lower are
        now excluded.
        """
        from xtgeo import Cube  # avoid circular import by having this here

        logger.debug("Create reduced cube...")

        cubev = self.cube.values.copy()  # copy, so we don't change the input instance
        cubev[self.cube.traceidcodes == 2] = 0.0  # set traceidcode 2 to zero

        # Create a boolean mask based on the threshold
        mask = self._depth_array < self._outside_depth

        # Find the bounding box of the true values
        non_zero_indices = np.nonzero(mask)[0]

        if len(non_zero_indices) == 0:
            raise RuntimeError(  # e.g. if cube and surfaces are at different locations
                "No valid data found in the depth cube. Perhaps the surfaces are "
                "outside the cube?"
            )

        min_indices = np.min(non_zero_indices)
        max_indices = np.max(non_zero_indices) + 1  # Add 1 to include the upper bound

        # Extract the reduced cube using slicing
        reduced = cubev[:, :, min_indices:max_indices]

        zori = float(self._depth_array.min())

        self._reduced_cube = Cube(
            ncol=reduced.shape[0],
            nrow=reduced.shape[1],
            nlay=reduced.shape[2],
            xinc=self.cube.xinc,
            yinc=self.cube.yinc,
            zinc=self.cube.zinc,
            xori=self.cube.xori,
            yori=self.cube.yori,
            zori=zori,
            rotation=self.cube.rotation,
            yflip=self.cube.yflip,
            values=reduced.astype(np.float32),
        )

        self._reduced_depth_array = self._depth_array[min_indices:max_indices]

        logger.debug("Create reduced cube... done")
        logger.debug(
            "Reduced cube created with shape %s", self._reduced_cube.values.shape
        )

    def _refine_interpolate(self) -> None:
        """Apply reduced cubes and interpolate to a finer grid vertically.

        This is done to get a more accurate representation of the cube values.
        """
        from xtgeo import Cube

        logger.debug("Refine cubes and interpolate...")
        arr = self._reduced_cube.values
        ndiv = self.ndiv

        # Create linear interpolation function along the last axis
        fdepth = make_interp_spline(
            np.arange(arr.shape[2]),
            self._reduced_depth_array,
            axis=0,
            k=1,
        )

        # Create interpolation function along the last axis
        if self.interpolation not in ["cubic", "linear"]:
            raise ValueError("Interpolation must be either 'cubic' or 'linear'")

        fcube = make_interp_spline(
            np.arange(arr.shape[2]),
            arr,
            axis=2,
            k=3 if self.interpolation == "cubic" else 1,
        )
        # Define new sampling points along the last axis
        nlay = arr.shape[2]
        new_z = np.linspace(0, nlay - 1, (nlay - 1) * ndiv + 1)

        # Resample the cube array
        resampled_arr = fcube(new_z)

        # Resample the depth array (always linear)
        self._refined_depth_array = new_depth = fdepth(new_z)
        new_zinc = (new_depth.max() - new_depth.min()) / (new_depth.shape[0] - 1)

        self._refined_cube = Cube(
            ncol=resampled_arr.shape[0],
            nrow=resampled_arr.shape[1],
            nlay=resampled_arr.shape[2],
            xinc=self.cube.xinc,
            yinc=self.cube.yinc,
            zinc=new_zinc,
            xori=self.cube.xori,
            yori=self.cube.yori,
            zori=self._refined_depth_array.min(),
            rotation=self._reduced_cube.rotation,
            yflip=self._reduced_cube.yflip,
            values=resampled_arr.astype(np.float32),
        )
        logger.debug("Refine cubes and interpolate... done")

    def _depth_mask(self) -> None:
        """Set nan values outside the interval defined by the upper + lower surface.

        In addition, set nan values where the thickness is less than the minimum.

        """
        logger.debug("Apply depth_mask...")
        darry = np.expand_dims(self._refined_depth_array, axis=(0, 1))
        upper_exp = np.expand_dims(self._upper.values, 2)
        lower_exp = np.expand_dims(self._lower.values, 2)
        mask_2d_exp = np.expand_dims(self._min_thickness_mask.values, 2)

        self._refined_cube.values = np.where(
            (darry < upper_exp) | (darry > lower_exp) | (mask_2d_exp == 0),
            np.nan,
            self._refined_cube.values,
        ).astype(np.float32)

        # similar for reduced cubes with original resolution
        darry = np.expand_dims(self._reduced_depth_array, axis=(0, 1))

        self._reduced_cube.values = np.where(
            (darry < upper_exp) | (darry > lower_exp) | (mask_2d_exp == 0),
            np.nan,
            self._reduced_cube.values,
        ).astype(np.float32)
        logger.debug("Apply depth_mask... done")

    def _add_to_attribute_map(self, attr_name: str, values: np.ndarray) -> None:
        """Compute the attribute map and add to result dictionary."""
        logger.debug("Add to attribute map...")
        attr_map = self._upper.copy()
        attr_map.values = np.ma.masked_invalid(values)

        # apply mask for the cube's dead traces (traceidcode 2)
        attr_map.values = np.ma.masked_where(
            self.cube.traceidcodes == 2, attr_map.values
        )

        # now resample to the original input map
        attr_map_resampled = self._template_surface.copy()
        attr_map_resampled.resample(attr_map)

        # Use template_surface consistently for masking (it's already set correctly)
        if hasattr(self._template_surface.values, "mask"):
            attr_map_resampled.values = np.ma.masked_where(
                self._template_surface.values.mask, attr_map_resampled.values
            )

        self._result_attr_maps[attr_name] = attr_map_resampled
        logger.debug("Add to attribute map... done")

    def _compute_statistical_attribute_surfaces(self) -> None:
        """Compute stats very fast by using internal C++ bindings."""
        logger.debug("Compute statistical attribute surfaces...")

        # compute statistics for vertically refined cube
        cubecpp = _internal.cube.Cube(self._refined_cube)
        all_attrs = cubecpp.cube_stats_along_z()

        for attr in STAT_ATTRS:
            self._add_to_attribute_map(attr, all_attrs[attr])

        # compute statistics for reduced cube (for sum attributes)
        cubecpp = _internal.cube.Cube(self._reduced_cube)
        all_attrs = cubecpp.cube_stats_along_z()

        # add sum attributes which are the last 3 in the list
        for attr in SUM_ATTRS:
            self._add_to_attribute_map(attr, all_attrs[attr])
        logger.debug("Compute statistical attribute surfaces... done")


@dataclass
class CubeAttrsV2:
    """Internal class (v2) for computing attributes in window between two surfaces.

    Compared with the former implementation, more logic is moved to the C++ routine,
    ensuring:
    - Significantly smaller memory overhead (e.g. 0.5 GB vs 20 GB)
    - Much faster execution, in particularly when using multiple processers. (5-10 x)
    """

    cube: Cube
    upper_surface: RegularSurface | float | int
    lower_surface: RegularSurface | float | int
    ndiv: int = 10
    interpolation: str = "cubic"  # cf. scipy's make_interp_spline() when k=3
    minimum_thickness: float = 0.0

    # internal attributes
    _template_surface: RegularSurface | None = None
    _depth_array: np.ndarray | None = None
    _outside_depth: float | None = None  # detected and updated from the depth cube
    _min_indices: int = 0  # minimum Z index for cube slicing
    _max_indices: int = 0  # maximum Z index for cube slicing
    _reduced_cube: Cube = None
    _reduced_depth_array: np.ndarray | None = None
    _refined_cube: Cube | None = None
    _refined_depth_array: np.ndarray | None = None

    _upper: RegularSurface | None = None  # upper surf, resampled to cube map resolution
    _lower: RegularSurface | None = None  # lower surf, resampled to cube map resolution
    _min_thickness_mask: RegularSurface | None = None  # mask for min. thickness trunc.
    _mask_map_by_traceidcode: RegularSurface | None = None  # mask for traceidcode 2

    _result_attr_maps: dict = field(default_factory=dict)  # holds the resulting maps

    def __post_init__(self) -> None:
        self._process_upper_lower_surface()
        self._create_depth_array()
        self._determine_slice_indices()
        self._compute_statistical_attribute_surfaces()

    def result(self) -> dict[RegularSurface]:
        # return the resulting attribute maps
        return self._result_attr_maps

    def _process_upper_lower_surface(self) -> None:
        """Extract upper and lower surface, sampled to cube resolution."""

        from xtgeo import surface_from_cube  # avoid circular import by having this here

        logger.debug("Process upper and lower surface...")

        upper = (
            surface_from_cube(self.cube, self.upper_surface)
            if isinstance(self.upper_surface, (float, int))
            else self.upper_surface
        )
        lower = (
            surface_from_cube(self.cube, self.lower_surface)
            if isinstance(self.lower_surface, (float, int))
            else self.lower_surface
        )

        # the template surface is the topology that defines the resulting attribute maps
        self._template_surface = (
            upper
            if isinstance(self.upper_surface, (float, int))
            else self.upper_surface
        )

        # determine which of "this" and "other" is actually upper and lower
        if (lower - upper).values.mean() < 0:
            raise ValueError(
                "The upper surface is below the lower surface. "
                "Please provide the surfaces in the correct order."
            )

        # although not an attribute, we store the upper and lower surfaces
        self._result_attr_maps["upper"] = upper
        self._result_attr_maps["lower"] = lower

        # get the surfaces on cube resolution
        self._upper = surface_from_cube(self.cube, self.cube.zori)
        self._lower = surface_from_cube(self.cube, self.cube.zori)
        self._upper.resample(upper)
        self._lower.resample(lower)

        self._upper.fill()
        self._lower.fill()

        self._min_thickness_mask = self._lower - self._upper

        self._min_thickness_mask.values = np.where(
            self._min_thickness_mask.values <= self.minimum_thickness, 0, 1
        )
        if np.all(self._min_thickness_mask.values == 0):
            raise ValueError(
                "The minimum thickness is too large, no valid data in the interval. "
                "Perhaps surfaces are overlapping?"
            )
        logger.debug("Process upper and lower surface... done")

    def _create_depth_array(self) -> None:
        """Create a 1D array where values are cube depths; to be used as filter.

        Belowe and above the input surfaces (plus a buffer), the values are set to
        a constant value self._outside_depth.

        Will also issue warnings or errors if the surfaces are outside the cube,
        depending on severity.
        """
        logger.debug("Create depth array...")

        self._depth_array = np.array(
            [
                self.cube.zori + n * self.cube.zinc
                for n in range(self.cube.values.shape[2])
            ]
        ).astype(np.float32)

        # check that surfaces are within the cube
        if self._upper.values.min() > self._depth_array.max():
            raise ValueError("Upper surface is fully below the cube")
        if self._lower.values.max() < self._depth_array.min():
            raise ValueError("Lower surface is fully above the cube")
        if self._upper.values.max() < self._depth_array.min():
            warnings.warn("Upper surface is fully above the cube", UserWarning)
        if self._lower.values.min() > self._depth_array.max():
            warnings.warn("Lower surface is fully below the cube", UserWarning)

        self._outside_depth = self._depth_array.max() + 1

        add_extra_depth = 2 * self.cube.zinc  # add buffer on upper/lower edges

        self._depth_array = np.where(
            self._depth_array < self._upper.values.min() - add_extra_depth,
            self._outside_depth,
            self._depth_array,
        )

        self._depth_array = np.where(
            self._depth_array > self._lower.values.max() + add_extra_depth,
            self._outside_depth,
            self._depth_array,
        )
        logger.debug("Create depth array... done")

    def _determine_slice_indices(self) -> None:
        """Create parameters for cube slicing.

        The purpose is to limit the computation to the relevant volume, to save
        CPU time. I.e. cube values above the upper surface and below the lower are
        now excluded.
        """
        logger.debug("Determine cube slice indices...")

        # Create a boolean mask based on the threshold
        mask = self._depth_array < self._outside_depth

        # Find the bounding box of the true values
        non_zero_indices = np.nonzero(mask)[0]

        if len(non_zero_indices) == 0:
            raise RuntimeError(  # e.g. if cube and surfaces are at different locations
                "No valid data found in the depth cube. Perhaps the surfaces are "
                "outside the cube?"
            )

        self._min_indices = int(np.min(non_zero_indices))
        # Add 1 to include the upper bound
        self._max_indices = int(np.max(non_zero_indices) + 1)

        logger.debug("Determine cube slice indices... done")
        logger.debug(
            "Cube slice indices: %d to %d", self._min_indices, self._max_indices
        )

    def _add_to_attribute_map(self, attr_name: str, values: np.ndarray) -> None:
        """Compute the attribute map and add to result dictionary."""
        logger.debug("Add to attribute map...")
        attr_map = self._upper.copy()
        attr_map.values = np.ma.masked_invalid(values)

        # now resample to the original input map
        attr_map_resampled = self._template_surface.copy()
        attr_map_resampled.resample(attr_map)

        # Use template_surface consistently for masking (it's already set correctly)
        if hasattr(self._template_surface.values, "mask"):
            attr_map_resampled.values = np.ma.masked_where(
                self._template_surface.values.mask, attr_map_resampled.values
            )

        self._result_attr_maps[attr_name] = attr_map_resampled
        logger.debug("Add to attribute map... done")

    def _compute_statistical_attribute_surfaces(self) -> None:
        """Compute stats very fast by using internal C++ bindings."""
        logger.debug("Compute statistical attribute surfaces...")

        # compute statistics for vertically refined cube using original cube
        cubecpp = _internal.cube.Cube(self.cube)
        all_attrs = cubecpp.cube_stats_along_z_v2(
            self._upper.values,
            self._lower.values,
            self._depth_array,  # use original depth array
            self.ndiv,
            self.interpolation,
            self.minimum_thickness,
            self._min_indices,  # pass slice indices
            self._max_indices,
        )

        for attr in STAT_ATTRS:
            self._add_to_attribute_map(attr, all_attrs[attr])

        # compute statistics with ndiv=1 (for sum attributes)
        all_attrs = cubecpp.cube_stats_along_z_v2(
            self._upper.values,
            self._lower.values,
            self._depth_array,  # use original depth array
            1,
            self.interpolation,
            self.minimum_thickness,
            self._min_indices,  # pass slice indices
            self._max_indices,
        )

        for attr in SUM_ATTRS:
            self._add_to_attribute_map(attr, all_attrs[attr])

        logger.debug("Compute statistical attribute surfaces... done")
