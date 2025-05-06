#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

static std::tuple<int, int, int, int>
estimate_ij_range(const xyz::Point &point,
                  const regsurf::RegularSurface &top_i,
                  const regsurf::RegularSurface &top_j,
                  const regsurf::RegularSurface &base_i,
                  const regsurf::RegularSurface &base_j,
                  const int ncol,
                  const int nrow)
{
    int buffer = 1;

    double i_top = regsurf::get_z_from_xy(top_i, point.x, point.y);
    double j_top = regsurf::get_z_from_xy(top_j, point.x, point.y);
    double i_base = regsurf::get_z_from_xy(base_i, point.x, point.y);
    double j_base = regsurf::get_z_from_xy(base_j, point.x, point.y);

    // if any of these are Nan, just return the maximum range
    if (std::isnan(i_top) || std::isnan(j_top) || std::isnan(i_base) ||
        std::isnan(j_base)) {
        return std::make_tuple(0, ncol - 1, 0, nrow - 1);
    }

    // now cast to int and find range adding a buffer but limit to 0
    // and ncol or nrow
    int imin = static_cast<int>(std::max(0.0, std::floor(i_top))) - buffer;
    int imax = static_cast<int>(std::min(ncol - 1.0, std::ceil(i_base))) + buffer;
    int jmin = static_cast<int>(std::max(0.0, std::floor(j_top))) - buffer;
    int jmax = static_cast<int>(std::min(nrow - 1.0, std::ceil(j_base))) + buffer;
    // make sure we are within the grid
    imin = std::max(0, imin);
    imax = std::min(ncol - 1, imax);
    jmin = std::max(0, jmin);
    jmax = std::min(nrow - 1, jmax);

    return std::make_tuple(imin, imax, jmin, jmax);
}

static std::tuple<int, int>
get_proposed_ij(const Grid &one_grid,
                const xyz::Point &point,
                int i_min,
                int i_max,
                int j_min,
                int j_max)
{

    int i_res = -1;
    int j_res = -1;

    bool found = false;

    for (int i = i_min; i <= i_max; ++i) {
        for (int j = j_min; j <= j_max; ++j) {
            // Get the corners of the current cell
            auto cell_corners = get_cell_corners_from_ijk(one_grid, i, j, 0);

            // Check if the point is inside the cell
            if (is_point_in_cell(point, cell_corners, "score_based")) {
                i_res = i;
                j_res = j;
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }

    return std::make_tuple(i_res, j_res);
}

/**
 * @brief Given an array of Points (organized as Polygon), return the grid indices
 * that contains the points
 */
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
get_indices_from_pointset(const Grid &grid,
                          const xyz::PointSet &points,
                          const Grid &one_grid,
                          const regsurf::RegularSurface &top_i,
                          const regsurf::RegularSurface &top_j,
                          const regsurf::RegularSurface &base_i,
                          const regsurf::RegularSurface &base_j,
                          const bool active_only)

{
    auto &logger = xtgeo::logging::LoggerManager::get("xtgeo.grid3d");

    logger.debug("Finding grid indices for points in a polygon or pointset");

    // Get the number of points in the polygon
    size_t num_points = points.size();

    // Create output arrays for indices
    py::array_t<int> i_indices(num_points);
    py::array_t<int> j_indices(num_points);
    py::array_t<int> k_indices(num_points);

    auto i_indices_ = i_indices.mutable_unchecked<1>();
    auto j_indices_ = j_indices.mutable_unchecked<1>();
    auto k_indices_ = k_indices.mutable_unchecked<1>();

    auto actnumsv_ = grid.actnumsv.unchecked<3>();

    // Initialize all indices to -1 (default for points not found in any cell)
    for (size_t idx = 0; idx < num_points; ++idx) {
        i_indices_(idx) = -1;
        j_indices_(idx) = -1;
        k_indices_(idx) = -1;
    }

    // bounding box of the grid
    auto [min_point, max_point] = get_bounding_box(grid);
    constexpr double EPSILON = 1e-9;

    int previous_k = 0;

    // Loop through each point in the PointSet
    for (size_t idx = 0; idx < num_points; ++idx) {
        const auto &point =
          points.get_point(idx);  // Access the point using get_point()

        // Check if the point is within the bounding box of the grid
        if (point.x < min_point.x - EPSILON || point.x > max_point.x + EPSILON ||
            point.y < min_point.y - EPSILON || point.y > max_point.y + EPSILON ||
            point.z < min_point.z - EPSILON || point.z > max_point.z + EPSILON) {
            continue;  // Skip points outside the bounding box for the grid
        }
        bool found = false;

        auto [imin, imax, jmin, jmax] =
          estimate_ij_range(point, top_i, top_j, base_i, base_j, grid.ncol, grid.nrow);

        auto [i, j] = get_proposed_ij(one_grid, point, imin, imax, jmin, jmax);

        // Loop through all K cells in the grid

        if (i == -1 || j == -1) {
            previous_k = 0;  // Reset previous_k if no valid i,j found
            continue;
        } else {
            // Start the search from the previous k index and loop in both directions
            previous_k = std::clamp(previous_k, 0, static_cast<int>(grid.nlay - 1));

            for (int offset = 0; offset < grid.nlay; ++offset) {

                int k_up = previous_k - offset;    // Search upwards
                int k_down = previous_k + offset;  // Search downwards

                // Check if k_up is within bounds
                if (k_up >= 0 && k_up < grid.nlay) {
                    auto cell_corners = get_cell_corners_from_ijk(grid, i, j, k_up);
                    if ((!active_only || (active_only && actnumsv_(i, j, k_up) > 0)) &&
                        is_point_in_cell(point, cell_corners, "score_based")) {
                        i_indices_(idx) = i;
                        j_indices_(idx) = j;
                        k_indices_(idx) = k_up;
                        previous_k = k_up;  // Update the previous k index
                        found = true;
                        break;
                    }
                }

                // Check if k_down is within bounds
                if (k_down >= 0 && k_down < grid.nlay) {
                    auto cell_corners = get_cell_corners_from_ijk(grid, i, j, k_down);
                    if ((!active_only ||
                         (active_only && actnumsv_(i, j, k_down) > 0)) &&
                        is_point_in_cell(point, cell_corners, "score_based")) {
                        i_indices_(idx) = i;
                        j_indices_(idx) = j;
                        k_indices_(idx) = k_down;
                        previous_k = k_down;  // Update the previous k index
                        found = true;
                        break;
                    }
                }
            }

            // If no cell contains the point, set indices to -1
            if (!found) {
                i_indices_(idx) = -1;
                j_indices_(idx) = -1;
                k_indices_(idx) = -1;
                previous_k = 0;  // Reset previous_k if no valid cell found
            }
        }
    }
    logger.debug("Finding grid indices for points in a polygon or pointset - DONE");

    return std::make_tuple(i_indices, j_indices, k_indices);
}
}  // namespace xtgeo::grid3d
