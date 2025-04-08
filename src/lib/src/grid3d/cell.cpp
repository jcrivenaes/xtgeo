#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Given a cell coordinate (i, j, k), find all corner coordinates as an
 * array with 24 values.
 *
 *      Top  --> i-dir     Base c
 *
 * (6,7,8) (9,10,11) (18,19,20) (21,22,23)
 *    |-------|          |-------|
 *    |       |          |       |
 *    |       |          |       |
 *    |-------|          |-------|
 * (0,1,2) (3,4,5)   (12,13,14) (15,16,17)
 * (i,j,k)
 *
 * @param Grid struct
 * @param i The (i) coordinate
 * @param j The (j) coordinate
 * @param k The (k) coordinate
 * @return CellCorners
 */
CellCorners
get_cell_corners_from_ijk(const Grid &grd,
                          const size_t i,
                          const size_t j,
                          const size_t k)
{
    auto coordsv_ = grd.coordsv.data();
    auto zcornsv_ = grd.zcornsv.data();

    double coords[4][6]{};
    auto num_rows = grd.nrow + 1;
    auto num_layers = grd.nlay + 1;
    auto n = 0;
    // Each cell is defined by 4 pillars
    for (auto x = 0; x < 2; x++) {
        for (auto y = 0; y < 2; y++) {
            for (auto z = 0; z < 6; z++) {
                auto idx = (i + y) * num_rows * 6 + (j + x) * 6 + z;
                coords[n][z] = coordsv_[idx];
            }
            n++;
        }
    }

    double z_coords[8]{};
    auto area = num_rows * num_layers;
    // Get the z value of each corner
    z_coords[0] = zcornsv_[((i + 0) * area + (j + 0) * num_layers + (k + 0)) * 4 + 3];
    z_coords[1] = zcornsv_[((i + 1) * area + (j + 0) * num_layers + (k + 0)) * 4 + 2];
    z_coords[2] = zcornsv_[((i + 0) * area + (j + 1) * num_layers + (k + 0)) * 4 + 1];
    z_coords[3] = zcornsv_[((i + 1) * area + (j + 1) * num_layers + (k + 0)) * 4 + 0];

    z_coords[4] = zcornsv_[((i + 0) * area + (j + 0) * num_layers + (k + 1)) * 4 + 3];
    z_coords[5] = zcornsv_[((i + 1) * area + (j + 0) * num_layers + (k + 1)) * 4 + 2];
    z_coords[6] = zcornsv_[((i + 0) * area + (j + 1) * num_layers + (k + 1)) * 4 + 1];
    z_coords[7] = zcornsv_[((i + 1) * area + (j + 1) * num_layers + (k + 1)) * 4 + 0];

    std::array<double, 24> corners{};
    auto crn_idx = 0;
    auto cz_idx = 0;
    for (auto layer = 0; layer < 2; layer++) {
        for (auto n = 0; n < 4; n++) {
            auto x1 = coords[n][0], y1 = coords[n][1], z1 = coords[n][2];
            auto x2 = coords[n][3], y2 = coords[n][4], z2 = coords[n][5];
            auto t = (z_coords[cz_idx] - z1) / (z2 - z1);
            auto point = numerics::lerp3d(x1, y1, z1, x2, y2, z2, t);
            // If coord lines are collapsed (preserves old behavior)
            if (std::abs(z2 - z1) < numerics::EPSILON) {
                point.x = x1;
                point.y = y1;
            }
            corners[crn_idx++] = point.x;
            corners[crn_idx++] = point.y;
            corners[crn_idx++] = z_coords[cz_idx];
            cz_idx++;
        }
    }
    return CellCorners(corners);
}

/*
 * Get the minimum and maximum values of the corners of a cell.
 * @param CellCorners struct
 * @return std::vector<double>
 */
std::vector<double>
get_corners_minmax(CellCorners &get_cell_corners_from_ijk)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();

    auto corners = get_cell_corners_from_ijk.arrange_corners();

    for (auto i = 0; i < 24; i += 3) {
        if (corners[i] < xmin) {
            xmin = corners[i];
        }
        if (corners[i] > xmax) {
            xmax = corners[i];
        }
        if (corners[i + 1] < ymin) {
            ymin = corners[i + 1];
        }
        if (corners[i + 1] > ymax) {
            ymax = corners[i + 1];
        }
        if (corners[i + 2] < zmin) {
            zmin = corners[i + 2];
        }
        if (corners[i + 2] > zmax) {
            zmax = corners[i + 2];
        }
    }
    std::vector<double> minmax{ xmin, xmax, ymin, ymax, zmin, zmax };
    return minmax;
}  // get_corners_minmax

/*
 * Estimate if a point is inside a cell face top (option != 1) or cell face bottom
 * (option = 1), seen from above, and return True if it is inside, False otherwise.
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

/*
 * Get the depth of a point inside a cell.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param CellCorners struct
 * @param option 0: Use cell top, 1: Use cell bottom
 * @return double
 */

double
get_depth_in_cell(const double x,
                  const double y,
                  const CellCorners &corners,
                  int option = 0)
{
    if (option < 0 || option > 1) {
        throw std::invalid_argument("BUG! Invalid option");
    }

    double depth = numerics::QUIET_NAN;

    if (option == 1) {
        depth = geometry::interpolate_z_4p(x, y, corners.lower_sw, corners.lower_se,
                                           corners.lower_nw, corners.lower_ne);
    } else {
        depth = geometry::interpolate_z_4p(x, y, corners.upper_sw, corners.upper_se,
                                           corners.upper_nw, corners.upper_ne);
    }
    return depth;
}  // get_depth_in_cell

/*
 * Check if a cell is convex by examining all corner points.
 * A cell is convex if all internal angles are less than 180 degrees.
 *
 * @param corners The cell corners
 * @return bool True if the cell is convex, false otherwise
 */
bool
is_cell_convex(const CellCorners &corners)
{
    // Create matrices for top and bottom faces
    Eigen::Matrix<double, 4, 3> top, bottom;

    // Fill matrices with corner coordinates
    top << corners.upper_sw.x, corners.upper_sw.y, corners.upper_sw.z,
      corners.upper_se.x, corners.upper_se.y, corners.upper_se.z, corners.upper_ne.x,
      corners.upper_ne.y, corners.upper_ne.z, corners.upper_nw.x, corners.upper_nw.y,
      corners.upper_nw.z;

    bottom << corners.lower_sw.x, corners.lower_sw.y, corners.lower_sw.z,
      corners.lower_se.x, corners.lower_se.y, corners.lower_se.z, corners.lower_ne.x,
      corners.lower_ne.y, corners.lower_ne.z, corners.lower_nw.x, corners.lower_nw.y,
      corners.lower_nw.z;

    auto check_face = [](const Eigen::Matrix<double, 4, 3> &face) -> bool {
        double sign = 0.0;
        for (int i = 0; i < 4; ++i) {
            // Get three consecutive vertices
            Eigen::Vector3d p1 = face.row(i);
            Eigen::Vector3d p2 = face.row((i + 1) % 4);
            Eigen::Vector3d p3 = face.row((i + 2) % 4);

            // Calculate vectors
            Eigen::Vector3d v1 = p2 - p1;
            Eigen::Vector3d v2 = p3 - p2;

            // Calculate cross product
            Eigen::Vector3d cross = v1.cross(v2);

            // Check if the sign changes
            if (std::abs(cross.z()) > numerics::EPSILON) {
                if (sign == 0.0) {
                    sign = cross.z();
                } else if (sign * cross.z() < 0) {
                    return false;
                }
            }
        }
        return true;
    };

    // Check both faces
    bool top_convex = check_face(top);
    bool bottom_convex = check_face(bottom);

    // For vertical edges, we'll just check they don't collapse
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d top_point = top.row(i);
        Eigen::Vector3d bottom_point = bottom.row(i);
        Eigen::Vector3d edge = bottom_point - top_point;

        if (edge.norm() < numerics::EPSILON) {
            return false;  // Collapsed edge
        }
    }

    return top_convex && bottom_convex;
}

// Function to determine if the coordinate system is right-handed or left-handed
static bool
is_right_handed(const Eigen::Vector3d &first,
                const Eigen::Vector3d &second,
                const Eigen::Vector3d &zaxis)
{
    // Calculate the cross product of X and Y
    Eigen::Vector3d cross_product = first.cross(second);

    // Calculate the dot product with Z
    double dot_product = cross_product.dot(zaxis);

    // If the dot product is positive, the system is right-handed
    return dot_product > 0;
}
static bool
is_cell_right_handed(const CellCorners &cell)
{
    // Define local axes using cell corners
    Eigen::Vector3d x_axis =
      cell.upper_se.to_eigen() - cell.upper_sw.to_eigen();  // X-axis or first
    Eigen::Vector3d y_axis =
      cell.upper_nw.to_eigen() - cell.upper_sw.to_eigen();  // Y-axis or second
    Eigen::Vector3d z_axis =
      cell.lower_sw.to_eigen() - cell.upper_sw.to_eigen();  // Z-axis

    // Call is_right_handed to determine the handedness
    return is_right_handed(x_axis, y_axis, z_axis);
}

// Function to check if the point p is "inside" the plane defined by points a, b, and c
static bool
is_inside_plane(const xyz::Point &p,
                const xyz::Point &a,
                const xyz::Point &b,
                const xyz::Point &c,
                const bool is_righthanded)
{
    Eigen::Vector3d point_p = p.to_eigen();
    Eigen::Vector3d point_a = a.to_eigen();
    Eigen::Vector3d point_b = b.to_eigen();
    Eigen::Vector3d point_c = c.to_eigen();

    Eigen::Vector3d normal = (point_b - point_a).cross(point_c - point_a);
    double sign = normal.dot(point_p - point_a);

    // Reverse the check due to left-handed system with Z positive down
    if (is_righthanded) {
        return sign >= 0;
    } else {
        return sign <= 0;
    }
}

// Function to check if a point is inside the hexahedron defined by the cell corners
bool
is_point_inside_cell(const CellCorners &cell, const xyz::Point &point)
{
    bool is_righthanded = is_cell_right_handed(cell);
    return is_inside_plane(point, cell.upper_sw, cell.upper_se, cell.upper_ne,
                           is_righthanded) &&
           is_inside_plane(point, cell.lower_se, cell.lower_sw, cell.lower_nw,
                           is_righthanded) &&
           is_inside_plane(point, cell.upper_sw, cell.lower_sw, cell.lower_se,
                           is_righthanded) &&
           is_inside_plane(point, cell.upper_se, cell.lower_se, cell.lower_ne,
                           is_righthanded) &&
           is_inside_plane(point, cell.upper_ne, cell.lower_ne, cell.upper_nw,
                           is_righthanded) &&
           is_inside_plane(point, cell.upper_sw, cell.upper_nw, cell.lower_nw,
                           is_righthanded);
}

bool
is_line_segment_inside_cell(const CellCorners &cell,
                            const xyz::Point &p1,
                            const xyz::Point &p2)
{
    // Check if either endpoint is inside the cell
    if (is_point_inside_cell(cell, p1) || is_point_inside_cell(cell, p2)) {
        return true;  // At least one endpoint is inside
    }

    // Define the faces of the cell as quadrilaterals
    std::array<std::array<xyz::Point, 4>, 6> cell_faces = { {
      { cell.upper_sw, cell.upper_se, cell.upper_ne, cell.upper_nw },  // Top face
      { cell.lower_sw, cell.lower_nw, cell.lower_ne, cell.lower_se },  // Bottom face
      { cell.upper_sw, cell.lower_sw, cell.lower_se, cell.upper_se },  // Front face
      { cell.upper_se, cell.lower_se, cell.lower_ne, cell.upper_ne },  // Right face
      { cell.upper_ne, cell.lower_ne, cell.lower_nw, cell.upper_nw },  // Back face
      { cell.upper_sw, cell.upper_nw, cell.lower_nw, cell.lower_sw }   // Left face
    } };

    // Check if the line segment intersects any of the cell's faces
    int i = 1;
    for (const auto &face : cell_faces) {
        // Convert the face and line segment to Eigen::Vector3d for compatibility
        printf("Face %d: %f %f %f --- %f %f %f --- %f %f %f --- %f %f %f\n", i,
               face[0].x, face[0].y, face[0].z, face[1].x, face[1].y, face[1].z,
               face[2].x, face[2].y, face[2].z, face[3].x, face[3].y, face[3].z);

        std::array<Eigen::Vector3d, 4> quad = {
            { face[0].to_eigen(), face[1].to_eigen(), face[2].to_eigen(),
              face[3].to_eigen() }
        };

        // Check if the line segment intersects the quadrilateral
        // Don't think handedness matters here
        if (geometry::does_line_segment_intersect_quad_internal(p1.to_eigen(),
                                                                p2.to_eigen(), quad)) {
            printf("Line segment intersects face %d\n", i);
            return true;  // The line segment intersects a face
        }
        i++;
    }

    // If no part of the line segment is inside the cell, return false
    return false;
}
}  // namespace xtgeo::grid3d
