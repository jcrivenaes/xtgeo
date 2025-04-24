#include <limits>   // Required for std::numeric_limits
#include <numeric>  // Required for std::accumulate
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using grid3d::CellCorners;

// Helper function to calculate the cross product
inline xyz::Point
cross_product(const xyz::Point &a, const xyz::Point &b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

// Helper function to calculate the dot product
inline double
dot_product(const xyz::Point &a, const xyz::Point &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Helper function for vector subtraction
inline xyz::Point
subtract(const xyz::Point &a, const xyz::Point &b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

// Helper function for vector addition
inline xyz::Point
add(const xyz::Point &a, const xyz::Point &b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

// Helper function for scalar multiplication
inline xyz::Point
scale(const xyz::Point &a, double s)
{
    return { a.x * s, a.y * s, a.z * s };
}

/**
 * @brief Faster test for convexity of a hexahedron involving point-plane
 *        distance checks.
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
static bool
is_hexahedron_non_convex_test1(const HexahedronCorners &corners)
{
    // Create an array of vertices for easier access
    std::array<xyz::Point, 8> vertices = { corners.upper_sw, corners.upper_se,
                                           corners.upper_ne, corners.upper_nw,
                                           corners.lower_sw, corners.lower_se,
                                           corners.lower_ne, corners.lower_nw };

    // Define faces with consistent outward-pointing normal winding order
    const std::array<std::array<int, 4>, 6> faces = { {
      { 4, 7, 6, 5 },  // bottom face
      { 0, 1, 2, 3 },  // top face
      { 4, 5, 1, 0 },  // front face
      { 5, 6, 2, 1 },  // right face
      { 6, 7, 3, 2 },  // back face
      { 7, 4, 0, 3 }   // left face
    } };

    const double POINT_PLANE_TOLERANCE = 1e-9;  // Tolerance for point-plane distance

    // Iterate over each face
    for (const auto &face : faces) {
        const xyz::Point &p0 = vertices[face[0]];
        const xyz::Point &p1 = vertices[face[1]];
        const xyz::Point &p2 = vertices[face[2]];

        // Calculate the normal of the face using the cross product
        xyz::Point edge1 = subtract(p1, p0);
        xyz::Point edge2 = subtract(p2, p0);
        xyz::Point normal = cross_product(edge1, edge2);

        // Check the sign of the dot product for all other vertices
        double reference_sign = 0.0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            // Skip vertices that are part of the current face
            if (i == face[0] || i == face[1] || i == face[2] || i == face[3]) {
                continue;
            }

            // Calculate the vector from the face to the vertex
            xyz::Point vec = subtract(vertices[i], p0);

            // Calculate the dot product with the face normal
            double dot = dot_product(vec, normal);

            // Determine the sign of the dot product
            if (std::abs(dot) > POINT_PLANE_TOLERANCE) {
                double sign = (dot > 0) ? 1.0 : -1.0;

                // If the reference sign is not set, initialize it
                if (reference_sign == 0.0) {
                    reference_sign = sign;
                }
                // If the sign differs from the reference, the cell is non-convex
                else if (sign != reference_sign) {
                    return true;  // Non-convex
                }
            }
        }
    }

    // If all vertices are on the same side of all face planes, the cell is convex
    return false;
}

/**
 * Determines if a hexahedral cell is non-convex.
 * Checks for non-planar faces and whether the centroid lies inside all face planes.
 *
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
static bool
is_hexahedron_non_convex_test2(const HexahedronCorners &corners)
{
    // Create more accessible array of vertices
    std::array<xyz::Point, 8> vertices = { corners.upper_sw, corners.upper_se,
                                           corners.upper_ne, corners.upper_nw,
                                           corners.lower_sw, corners.lower_se,
                                           corners.lower_ne, corners.lower_nw };

    // Define faces with consistent outward-pointing normal winding order (assuming
    // standard node numbering) (e.g., using the right-hand rule)
    const std::array<std::array<int, 4>, 6> faces = { {
      { 4, 7, 6, 5 },  // bottom face (points down)
      { 0, 1, 2, 3 },  // top face (points up)
      { 4, 5, 1, 0 },  // front face (points -y)
      { 5, 6, 2, 1 },  // right face (points +x)
      { 6, 7, 3, 2 },  // back face (points +y)
      { 7, 4, 0, 3 }   // left face (points -x)
    } };

    const double PLANAR_TOLERANCE_VOL =
      1e-9;  // Tolerance for scalar triple product (volume)
    const double POINT_PLANE_TOLERANCE = 1e-9;  // Tolerance for point-plane distance

    // 1. Check if any face is non-planar using scalar triple product
    for (const auto &face : faces) {
        const xyz::Point &a = vertices[face[0]];
        const xyz::Point &b = vertices[face[1]];
        const xyz::Point &c = vertices[face[2]];
        const xyz::Point &d = vertices[face[3]];

        xyz::Point ab = subtract(b, a);
        xyz::Point ac = subtract(c, a);
        xyz::Point ad = subtract(d, a);

        // Volume of tetrahedron formed by A, B, C, D
        double volume = dot_product(ab, cross_product(ac, ad));

        // If volume is significantly non-zero, the face is non-planar
        // Note: Need to consider the scale of the cell coordinates for the tolerance.
        // A relative tolerance might be better, but requires calculating face area/edge
        // lengths. Using a small absolute tolerance for now.
        if (std::abs(volume) > PLANAR_TOLERANCE_VOL) {
            // Optional: Add logging here if needed
            // xtgeo::log_trace("Non-planar face detected (volume: {})", volume);
            return true;  // Face is non-planar, cell is non-convex
        }
    }

    // 2. Check if the centroid lies on the inner side of all face planes
    // Calculate centroid
    xyz::Point centroid = { 0.0, 0.0, 0.0 };
    for (const auto &v : vertices) {
        centroid = add(centroid, v);
    }
    centroid = scale(centroid, 1.0 / 8.0);

    for (const auto &face : faces) {
        const xyz::Point &p0 = vertices[face[0]];
        const xyz::Point &p1 = vertices[face[1]];
        // Use p3 for normal calculation based on winding order {0, 1, 2, 3} -> (p1-p0)
        // x (p3-p0)
        const xyz::Point &p3 = vertices[face[3]];

        // Calculate face normal (consistent winding order assumed in `faces` array)
        xyz::Point v1 = subtract(p1, p0);
        xyz::Point v2 = subtract(p3, p0);
        xyz::Point normal = cross_product(v1, v2);

        // Check if the centroid is on the correct side of the plane defined by p0 and
        // normal (p - p0) . N <= 0 means p is on the side opposite to N direction (or
        // on the plane)
        double dist = dot_product(subtract(centroid, p0), normal);

        // If dist is positive (beyond tolerance), the centroid is "outside" this face
        // plane.
        if (dist > POINT_PLANE_TOLERANCE) {
            // Optional: Add logging here if needed
            // xtgeo::log_trace("Centroid outside face plane (dist: {})", dist);
            return true;  // Centroid is outside, cell is non-convex
        }
    }

    // Cell passed all tests, it's convex
    return false;
}

/**
 * @brief Check if a hexahedron is non-convex.
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
bool
is_hexahedron_non_convex(const HexahedronCorners &corners)
{
    // Check if the hexahedron is non-convex using the first test
    if (is_hexahedron_non_convex_test1(corners)) {
        return true;
    }

    // If the first test fails, check using the second test
    return is_hexahedron_non_convex_test2(corners);
}  // is_hexahedron_non_convex

/*
 * Get the minimum and maximum values of the corners of a hexahedron.
 * @param CellCorners struct
 * @return std::vector<double>
 */
std::vector<double>
get_hexahedron_minmax(const HexahedronCorners &cell_corners)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();

    // List of all corners
    std::array<xyz::Point, 8> corners = {
        cell_corners.upper_sw, cell_corners.upper_se, cell_corners.upper_ne,
        cell_corners.upper_nw, cell_corners.lower_sw, cell_corners.lower_se,
        cell_corners.lower_ne, cell_corners.lower_nw
    };

    // Iterate over all corners to find min/max values
    for (const auto &corner : corners) {
        if (corner.x < xmin)
            xmin = corner.x;
        if (corner.x > xmax)
            xmax = corner.x;
        if (corner.y < ymin)
            ymin = corner.y;
        if (corner.y > ymax)
            ymax = corner.y;
        if (corner.z < zmin)
            zmin = corner.z;
        if (corner.z > zmax)
            zmax = corner.z;
    }

    return { xmin, xmax, ymin, ymax, zmin, zmax };
}

/**
 * @brief Get the bounding box for a cell, a wrapper for get_corners_minmax.
 * @param CellCorners struct
 * @return std::tuple<xyz::Point, xyz::Point> {min_point, max_point}
 */
std::tuple<xyz::Point, xyz::Point>
get_hexahedron_bounding_box(const HexahedronCorners &corners)
{
    auto minmax = get_hexahedron_minmax(corners);
    auto min_point = xyz::Point(minmax[0], minmax[2], minmax[4]);
    auto max_point = xyz::Point(minmax[1], minmax[3], minmax[5]);
    return std::make_tuple(min_point, max_point);
}  // get_hexahedron_bounding_box

}  // namespace xtgeo::geometry
