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

// #include <xtgeo/geometry.hpp>
// #include <xtgeo/grid3d.hpp>
// #include <xtgeo/logging.hpp>
// #include <xtgeo/numerics.hpp>
// #include <xtgeo/types.hpp>

// namespace xtgeo::geometry {

// using grid3d::CellCorners;

// // /**
// //  * Determines if a hexahedral cell is non-convex by checking if any face is
// //  non-planar
// //  * or if any internal planes (diagonals) intersect the cell's boundary.
// //  *
// //  * @param corners The 8 corners of the hexahedron
// //  * @return true if the cell is non-convex, false if it is convex
// //  */
// bool
// is_hexahedron_non_convex(const grid3d::CellCorners &corners)
// {
//     // Create more accessible array of vertices
//     std::array<xyz::Point, 8> vertices = { corners.upper_sw, corners.upper_se,
//                                            corners.upper_ne, corners.upper_nw,
//                                            corners.lower_sw, corners.lower_se,
//                                            corners.lower_ne, corners.lower_nw };

//     // 1. Check if any face is non-planar
//     // A planar quadrilateral has zero volume when split into two triangles
//     const std::array<std::array<int, 4>, 6> faces = { {
//       { 0, 1, 2, 3 },  // top face
//       { 4, 5, 6, 7 },  // bottom face
//       { 0, 1, 5, 4 },  // front face
//       { 1, 2, 6, 5 },  // right face
//       { 2, 3, 7, 6 },  // back face
//       { 3, 0, 4, 7 }   // left face
//     } };

//     const double COPLANAR_TOLERANCE = 1e-8;

//     // Check each face for planarity
//     for (const auto &face : faces) {
//         // Get the 4 corners of this face
//         const xyz::Point &a = vertices[face[0]];
//         const xyz::Point &b = vertices[face[1]];
//         const xyz::Point &c = vertices[face[2]];
//         const xyz::Point &d = vertices[face[3]];

//         // Calculate normal vectors for two triangulations of the face
//         // Triangle 1: ABC
//         xyz::Point v1 = { b.x - a.x, b.y - a.y, b.z - a.z };
//         xyz::Point v2 = { c.x - a.x, c.y - a.y, c.z - a.z };
//         xyz::Point n1 = { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
//                           v1.x * v2.y - v1.y * v2.x };

//         // Triangle 2: ACD
//         xyz::Point v3 = { d.x - a.x, d.y - a.y, d.z - a.z };
//         xyz::Point n2 = { v2.y * v3.z - v2.z * v3.y, v2.z * v3.x - v2.x * v3.z,
//                           v2.x * v3.y - v2.y * v3.x };

//         // Normalize normal vectors
//         double len1 = std::sqrt(n1.x * n1.x + n1.y * n1.y + n1.z * n1.z);
//         double len2 = std::sqrt(n2.x * n2.x + n2.y * n2.y + n2.z * n2.z);

//         if (len1 > COPLANAR_TOLERANCE && len2 > COPLANAR_TOLERANCE) {
//             n1 = { n1.x / len1, n1.y / len1, n1.z / len1 };
//             n2 = { n2.x / len2, n2.y / len2, n2.z / len2 };

//             // Check if normals are significantly different (non-planar face)
//             double dot_product = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
//             if (std::abs(dot_product) <
//                 0.99) {       // Allow small deviation (about 8 degrees)
//                 return true;  // Face is non-planar, cell is non-convex
//             }
//         }
//     }

//     // 2. Check for self-intersection with diagonal planes
//     // This is more complex but essential for cells like your test case with the
//     // collapsed corner

//     // Test diagonal planes through the cell
//     const std::array<std::array<int, 4>, 6> diagonals = { {
//       { 0, 2, 6, 4 },  // Diagonal plane SW-NE
//       { 1, 3, 7, 5 },  // Diagonal plane SE-NW
//       { 0, 3, 6, 5 },  // Cross diagonal 1
//       { 1, 2, 7, 4 },  // Cross diagonal 2
//       { 0, 1, 7, 6 },  // Cross diagonal 3
//       { 2, 3, 5, 4 }   // Cross diagonal 4
//     } };

//     // For each diagonal plane, check if any edges of the cell intersect it
//     // (excluding the edges that form the plane)
//     const std::array<std::array<int, 2>, 12> edges = { {
//       { 0, 1 },
//       { 1, 2 },
//       { 2, 3 },
//       { 3, 0 },  // Top face edges
//       { 4, 5 },
//       { 5, 6 },
//       { 6, 7 },
//       { 7, 4 },  // Bottom face edges
//       { 0, 4 },
//       { 1, 5 },
//       { 2, 6 },
//       { 3, 7 }  // Vertical edges
//     } };

//     for (const auto &diag : diagonals) {
//         // Form a plane from the first three points
//         const xyz::Point &p0 = vertices[diag[0]];
//         const xyz::Point &p1 = vertices[diag[1]];
//         const xyz::Point &p2 = vertices[diag[2]];

//         // Calculate plane normal
//         xyz::Point v1 = { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
//         xyz::Point v2 = { p2.x - p0.x, p2.y - p0.y, p2.z - p0.z };
//         xyz::Point normal = { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
//                               v1.x * v2.y - v1.y * v2.x };

//         // D value in plane equation Ax + By + Cz + D = 0
//         double d = -(normal.x * p0.x + normal.y * p0.y + normal.z * p0.z);

//         // Check each edge for intersection with this plane
//         for (const auto &edge : edges) {
//             // Skip edges that are part of the diagonal plane
//             if (std::find(diag.begin(), diag.end(), edge[0]) != diag.end() &&
//                 std::find(diag.begin(), diag.end(), edge[1]) != diag.end()) {
//                 continue;
//             }

//             const xyz::Point &e1 = vertices[edge[0]];
//             const xyz::Point &e2 = vertices[edge[1]];

//             // Check if the edge endpoints are on opposite sides of the plane
//             double d1 = normal.x * e1.x + normal.y * e1.y + normal.z * e1.z + d;
//             double d2 = normal.x * e2.x + normal.y * e2.y + normal.z * e2.z + d;

//             // If signs are opposite, the plane intersects this edge
//             if (d1 * d2 < 0) {
//                 return true;  // Cell is non-convex
//             }
//         }
//     }

//     // Cell passed all tests, it's convex
//     return false;
// }

// }  // namespace geometry
