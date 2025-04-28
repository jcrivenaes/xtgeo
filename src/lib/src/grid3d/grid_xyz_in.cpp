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

static std::tuple<int, int>
get_proposed_ij(const Grid &one_grid, const xyz::Point &point)
{

    int i_res = -1;
    int j_res = -1;

    bool found = false;

    for (int i = 0; i < one_grid.ncol; ++i) {
        for (int j = 0; j < one_grid.nrow; ++j) {
            // Get the corners of the current cell
            auto cell_corners = get_cell_corners_from_ijk(one_grid, i, j, 0);

            // Check if the point is inside the cell
            if (is_point_in_cell(point, cell_corners)) {
                i_res = i;
                j_res = j;
                found = true;
                break;
            }
        }
        if (found)
            break;
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
                          const Grid &one_grid)
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("grid3d.get_indices_from_polygon");

    // Get the number of points in the polygon
    size_t num_points = points.size();
    logger.debug("Number of points in polygon: {}", num_points);

    // Create output arrays for indices
    py::array_t<int> i_indices(num_points);
    py::array_t<int> j_indices(num_points);
    py::array_t<int> k_indices(num_points);

    auto i_indices_ = i_indices.mutable_unchecked<1>();
    auto j_indices_ = j_indices.mutable_unchecked<1>();
    auto k_indices_ = k_indices.mutable_unchecked<1>();

    // Initialize all indices to -1 (default for points not found in any cell)
    for (size_t idx = 0; idx < num_points; ++idx) {
        i_indices_(idx) = -1;
        j_indices_(idx) = -1;
        k_indices_(idx) = -1;
    }

    // Loop through each point in the PointSet
    for (size_t idx = 0; idx < num_points; ++idx) {
        const auto &point =
          points.get_point(idx);  // Access the point using get_point()

        bool found = false;

        auto [i, j] = get_proposed_ij(one_grid, point);
        logger.info("Point coordinates: ({}, {}, {})", point.x, point.y, point.z);
        logger.info("Proposed i: {}, j: {}", i, j);

        // Loop through all K cells in the grid

        if (i == -1 || j == -1) {
            continue;
        } else {
            for (int k = 0; k < grid.nlay; ++k) {
                // Get the corners of the current cell
                auto cell_corners = get_cell_corners_from_ijk(grid, i, j, k);

                // Check if the point is inside the cell
                if (is_point_in_cell(point, cell_corners)) {
                    i_indices_(idx) = i;
                    j_indices_(idx) = j;
                    k_indices_(idx) = k;
                    found = true;
                    break;
                }
            }

            // If no cell contains the point, set indices to -1
            if (!found) {
                i_indices_(idx) = -1;
                j_indices_(idx) = -1;
                k_indices_(idx) = -1;
            }
        }
    }

    return std::make_tuple(i_indices, j_indices, k_indices);
}
}  // namespace xtgeo::grid3d
