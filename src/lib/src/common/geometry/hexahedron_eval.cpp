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
 * Determines if a hexahedral cell is non-convex.
 * Checks for non-planar faces and whether the centroid lies inside all face planes.
 *
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
bool
is_hexahedron_non_convex(const grid3d::CellCorners &corners)
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

}  // namespace xtgeo::geometry
