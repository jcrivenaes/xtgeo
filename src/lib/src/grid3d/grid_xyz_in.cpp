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

/**
 * @brief Estimate the i,j range for a point based on top/base surfaces
 */
static std::tuple<int, int, int, int>
estimate_ij_range(const xyz::Point &point,
                  const regsurf::RegularSurface &top_i,
                  const regsurf::RegularSurface &top_j,
                  const regsurf::RegularSurface &base_i,
                  const regsurf::RegularSurface &base_j,
                  const int ncol,
                  const int nrow)
{
    constexpr int buffer = 1;

    double i_top = regsurf::get_z_from_xy(top_i, point.x, point.y);
    double j_top = regsurf::get_z_from_xy(top_j, point.x, point.y);
    double i_base = regsurf::get_z_from_xy(base_i, point.x, point.y);
    double j_base = regsurf::get_z_from_xy(base_j, point.x, point.y);

    // If all values are NaN, the point is outside the grid
    if (std::isnan(i_top) && std::isnan(j_top) && std::isnan(i_base) &&
        std::isnan(j_base)) {
        return std::make_tuple(-1, -1, -1, -1);
    }

    // If any value is NaN, search the entire grid
    if (std::isnan(i_top) || std::isnan(j_top) || std::isnan(i_base) ||
        std::isnan(j_base)) {
        return std::make_tuple(0, ncol - 1, 0, nrow - 1);
    }

    // Calculate range with buffer, ensuring it stays within grid bounds
    int imin = std::max(0, static_cast<int>(std::floor(i_top)) - buffer);
    int imax = std::min(ncol - 1, static_cast<int>(std::ceil(i_base)) + buffer);
    int jmin = std::max(0, static_cast<int>(std::floor(j_top)) - buffer);
    int jmax = std::min(nrow - 1, static_cast<int>(std::ceil(j_base)) + buffer);

    return std::make_tuple(imin, imax, jmin, jmax);
}

/**
 * @brief Find a proposed i,j coordinate for a point in a 2D grid
 */
static std::tuple<int, int>
get_proposed_ij(const Grid &one_grid,
                const xyz::Point &point,
                int i_min,
                int i_max,
                int j_min,
                int j_max)
{
    for (int i = i_min; i <= i_max; ++i) {
        for (int j = j_min; j <= j_max; ++j) {
            auto cell_corners = get_cell_corners_from_ijk(one_grid, i, j, 0);
            if (is_point_in_cell(point, cell_corners)) {
                return std::make_tuple(i, j);
            }
        }
    }

    return std::make_tuple(-1, -1);  // No match found
}

/**
 * @brief Check if a point is inside the grid's bounding box
 */
static bool
is_point_in_grid_bounds(const xyz::Point &point,
                        const xyz::Point &min_point,
                        const xyz::Point &max_point,
                        const double epsilon = 1e-9)
{
    return !(point.x < min_point.x - epsilon || point.x > max_point.x + epsilon ||
             point.y < min_point.y - epsilon || point.y > max_point.y + epsilon ||
             point.z < min_point.z - epsilon || point.z > max_point.z + epsilon);
}

/**
 * @brief Search for a point within a cell column (all K layers)
 * @return true if found, false otherwise
 */
static bool
find_in_column(const Grid &grid,
               const xyz::Point &point,
               int i,
               int j,
               int &previous_k,
               int &found_i,
               int &found_j,
               int &found_k,
               const bool active_only,
               const py::detail::unchecked_reference<int, 3> &actnumsv)
{
    // Make sure previous_k is within bounds
    previous_k = std::clamp(previous_k, 0, static_cast<int>(grid.nlay - 1));

    // Search outward from previous_k in both directions
    for (int offset = 0; offset < grid.nlay; ++offset) {
        // Try upward
        int k_up = previous_k - offset;
        if (k_up >= 0 && k_up < grid.nlay) {
            // Only check if cell is active (when required)
            if (!active_only || (active_only && actnumsv(i, j, k_up) > 0)) {
                auto cell_corners = get_cell_corners_from_ijk(grid, i, j, k_up);
                if (is_point_in_cell(point, cell_corners)) {
                    found_i = i;
                    found_j = j;
                    found_k = k_up;
                    previous_k = k_up;  // Update for next search
                    return true;
                }
            }
        }

        // Try downward (skip if same as upward)
        int k_down = previous_k + offset;
        if (k_down >= 0 && k_down < grid.nlay && k_down != k_up) {
            if (!active_only || (active_only && actnumsv(i, j, k_down) > 0)) {
                auto cell_corners = get_cell_corners_from_ijk(grid, i, j, k_down);
                if (is_point_in_cell(point, cell_corners)) {
                    found_i = i;
                    found_j = j;
                    found_k = k_down;
                    previous_k = k_down;  // Update for next search
                    return true;
                }
            }
        }
    }

    return false;  // Point not found in this column
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

    // Get the number of points
    size_t num_points = points.size();

    // Create output arrays for indices (initialized to -1)
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

    // Get grid bounding box once
    auto [min_point, max_point] = get_bounding_box(grid);
    int previous_k = 0;

    // Process each point
    for (size_t idx = 0; idx < num_points; ++idx) {
        const auto &point = points.get_point(idx);

        // Skip points outside grid bounds
        if (!is_point_in_grid_bounds(point, min_point, max_point)) {
            continue;
        }

        // Get potential i,j range for the point
        auto [imin, imax, jmin, jmax] =
          estimate_ij_range(point, top_i, top_j, base_i, base_j, grid.ncol, grid.nrow);

        logger.debug("Point {}: i_min={}, i_max={}, j_min={}, j_max={}", idx, imin,
                     imax, jmin, jmax);

        if (imin == -1) {  // Point outside grid (all NaN values from surfaces)
            previous_k = 0;
            continue;
        }

        // Try to find a proposed column for efficiency
        auto [i_est, j_est] = get_proposed_ij(one_grid, point, imin, imax, jmin, jmax);

        // If we found a proposed column, restrict search to just that column
        if (i_est >= 0) {
            imin = imax = i_est;
            jmin = jmax = j_est;
        }

        // Search for the point in the possible cell columns
        bool found = false;
        for (int i = imin; i <= imax && !found; ++i) {
            for (int j = jmin; j <= jmax && !found; ++j) {
                found = find_in_column(grid, point, i, j, previous_k, i_indices_(idx),
                                       j_indices_(idx), k_indices_(idx), active_only,
                                       actnumsv_);
            }
        }

        // Reset previous_k if no cell found
        if (!found) {
            previous_k = 0;
        }
    }

    logger.debug("Finding grid indices for points in a polygon or pointset - DONE");
    return std::make_tuple(i_indices, j_indices, k_indices);
}

}  // namespace xtgeo::grid3d
