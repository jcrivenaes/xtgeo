#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>
#include <xtgeo/xyz.hpp>

/**
 * @brief This file focus on interaction with the cell and points/polylines
 */

namespace py = pybind11;

namespace xtgeo::grid3d {

/**
 * @brief Estimate if a XY point is inside a cell face top (option != 1) or cell face
 * bottom (option = 1), seen from above, and return True if it is inside, False
 * otherwise.
 *
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param CellCorners struct
 * @param option 0: Use cell top, 1: Use cell bottom, 2 for center
 * @return Boolean
 */
bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const CellCorners &corners,
                    int option)
{
    if (option < 0 || option > 2) {
        throw std::invalid_argument("BUG! Invalid option");
    }

    // determine if point is inside the polygon
    if (option == 0) {
        return geometry::is_xy_point_in_quadrilateral(
          x, y, corners.upper_sw, corners.upper_se, corners.upper_ne, corners.upper_nw);
    } else if (option == 1) {
        return geometry::is_xy_point_in_quadrilateral(
          x, y, corners.lower_sw, corners.lower_se, corners.lower_ne, corners.lower_nw);
    } else if (option == 2) {
        // find the center Z point of the cell
        auto mid_sw = numerics::lerp3d(corners.upper_sw.x, corners.upper_sw.y,
                                       corners.upper_sw.z, corners.lower_sw.x,
                                       corners.lower_sw.y, corners.lower_sw.z, 0.5);
        auto mid_se = numerics::lerp3d(corners.upper_se.x, corners.upper_se.y,
                                       corners.upper_se.z, corners.lower_se.x,
                                       corners.lower_se.y, corners.lower_se.z, 0.5);
        auto mid_nw = numerics::lerp3d(corners.upper_nw.x, corners.upper_nw.y,
                                       corners.upper_nw.z, corners.lower_nw.x,
                                       corners.lower_nw.y, corners.lower_nw.z, 0.5);
        auto mid_ne = numerics::lerp3d(corners.upper_ne.x, corners.upper_ne.y,
                                       corners.upper_ne.z, corners.lower_ne.x,
                                       corners.lower_ne.y, corners.lower_ne.z, 0.5);

        return geometry::is_xy_point_in_quadrilateral(
          x, y, { mid_sw.x, mid_sw.y, mid_sw.z }, { mid_se.x, mid_se.y, mid_se.z },
          { mid_ne.x, mid_ne.y, mid_ne.z }, { mid_nw.x, mid_nw.y, mid_nw.z });
    }
    return false;  // unreachable
}  // is_xy_point_in_cell

static bool
is_cell_thin(const CellCorners &corners)
{
    auto &logger = xtgeo::logging::LoggerManager::get("grid3d::is_cell_thin");
    // must check all corners

    double sw_dz = std::abs(corners.upper_sw.z - corners.lower_sw.z);
    double se_dz = std::abs(corners.upper_se.z - corners.lower_se.z);
    double nw_dz = std::abs(corners.upper_nw.z - corners.lower_nw.z);
    double ne_dz = std::abs(corners.upper_ne.z - corners.lower_ne.z);

    double dz_avg = (sw_dz + se_dz + nw_dz + ne_dz) / 4.0;
    double area_upper = geometry::quadrilateral_area(
      corners.upper_sw, corners.upper_se, corners.upper_ne, corners.upper_nw);
    double area_lower = geometry::quadrilateral_area(
      corners.lower_sw, corners.lower_se, corners.lower_ne, corners.lower_nw);

    double dxy_avg = std::sqrt(0.5 * (area_lower + area_upper));

    double dz_factor = dz_avg / dxy_avg;

    // criteria for thin: cell thickness is less than 5% of XY dimensions in average
    // or cell thickness is less than 1% of XY dimensions in corners
    double thin_factor_avg = 0.05 * dxy_avg;
    double minimum_dz = 0.01 * dxy_avg;  // ie if cell is 50m wide, use 0.5m as minimum

    if (dz_factor < thin_factor_avg) {
        return true;
    }

    // Check if any of the corners are too thin
    if (sw_dz < minimum_dz || se_dz < minimum_dz || nw_dz < minimum_dz ||
        ne_dz < minimum_dz) {
        return true;
    }

}  // is_cell_thin

/**
 * @brief Check if a point is inside a cell defined by its corners.
 *
 * Uses one method for normal cells and a different method for thin cells.
 * The function first checks if the point is within the bounding box of the cell for
 * quick rejection.
 *
 * @param point The point to check
 * @param corners The corners of the cell
 * @return true if the point is inside the cell, false otherwise
 */
bool
is_point_in_cell(const xyz::Point &point, const CellCorners &corners)
{
    auto &logger = xtgeo::logging::LoggerManager::get("grid3d::is_point_in_cell");

    // Check if the cell is thin or normal, and use the appropriate method
    if (is_cell_thin(corners)) {
        // Check if the cell is thin, using special method
        logger.debug("Cell is considered thin, using tetrahedron method.");
        return geometry::is_point_in_hexahedron(point, corners, "tetrahedrons");
    }

    logger.debug("Cell is not considered thin, using ray casting method.");

    // For "normal" cells...
    return geometry::is_point_in_hexahedron(point, corners, "tetrahedrons");
}

}  // namespace xtgeo::grid3d
