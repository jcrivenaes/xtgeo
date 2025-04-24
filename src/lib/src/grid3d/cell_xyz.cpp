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

/**
 * @brief Check if a point is inside a cell defined by its corners.
 *
 * Uses one method for normal cells and a different method for "distorted" cells.
 *
 * @param point The point to check
 * @param corners The corners of the cell
 * @return true if the point is inside the cell, false otherwise
 */
bool
is_point_in_cell(const xyz::Point &point,
                 const CellCorners &corners,
                 const std::string &method)
{
    auto &logger = xtgeo::logging::LoggerManager::get("grid3d::is_point_in_cell");

    // convert to right handed system
    auto hexahedron_corners = corners.to_hexahedron_corners();
    auto rh_point = xyz::Point(point.x, point.y, -point.z);

    if (method != "auto") {
        return geometry::is_point_in_hexahedron(rh_point, hexahedron_corners, method);
    }

    // when auto, check if the cell is distorted
    if (geometry::is_hexahedron_non_convex(hexahedron_corners)) {
        // Check if the cell is non-convex, using special method
        logger.debug(
          "Cell is considered non-convex, using centroid tetrahedron method.");
        return geometry::is_point_in_hexahedron(rh_point, hexahedron_corners,
                                                "centroid_tetrahedrons");
    }

    // For "normal" cells...
    return geometry::is_point_in_hexahedron(rh_point, hexahedron_corners,
                                            "tetrahedrons");
}

}  // namespace xtgeo::grid3d
