#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using grid3d::CellCorners;
using xyz::Point;

/*
 * Function to determine if a ray intersects a triangle in 3D space.
 * Uses Möller–Trumbore intersection algorithm. This implementation is robust against
 * both left-handed and right-handed coordinate systems.
 *
 * @param origin Origin point of the ray
 * @param direction Direction of the ray (doesn't need to be normalized)
 * @param v0 First vertex of the triangle
 * @param v1 Second vertex of the triangle
 * @param v2 Third vertex of the triangle
 * @return bool True if the ray intersects the triangle, false otherwise
 */
static bool
ray_intersects_triangle(const xyz::Point &origin,
                        const xyz::Point &direction,
                        const xyz::Point &v0,
                        const xyz::Point &v1,
                        const xyz::Point &v2)
{
    const double EPSILON = 1e-10;

    // Calculate edge vectors
    xyz::Point edge1 = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
    xyz::Point edge2 = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };

    // Calculate the cross product (pvec = direction × edge2)
    xyz::Point pvec = { direction.y * edge2.z - direction.z * edge2.y,
                        direction.z * edge2.x - direction.x * edge2.z,
                        direction.x * edge2.y - direction.y * edge2.x };

    // Calculate determinant (dot product of edge1 and pvec)
    double det = edge1.x * pvec.x + edge1.y * pvec.y + edge1.z * pvec.z;

    // If determinant is near zero, ray lies in plane of triangle or ray is parallel to
    // plane Use absolute value to handle both left and right-handed coordinate systems
    if (fabs(det) < EPSILON)
        return false;

    double inv_det = 1.0 / det;

    // Calculate vector from v0 to ray origin
    xyz::Point tvec = { origin.x - v0.x, origin.y - v0.y, origin.z - v0.z };

    // Calculate u parameter and test bounds
    double u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    // Calculate qvec (qvec = tvec × edge1)
    xyz::Point qvec = { tvec.y * edge1.z - tvec.z * edge1.y,
                        tvec.z * edge1.x - tvec.x * edge1.z,
                        tvec.x * edge1.y - tvec.y * edge1.x };

    // Calculate v parameter and test bounds
    double v =
      (direction.x * qvec.x + direction.y * qvec.y + direction.z * qvec.z) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    // Calculate t parameter - ray intersects triangle
    double t = (edge2.x * qvec.x + edge2.y * qvec.y + edge2.z * qvec.z) * inv_det;

    // Only consider intersections in front of the ray
    return (t > EPSILON);
}  // ray_intersects_triangle

// helper function to use ray casting to check if a point is inside a "normal" cell
static bool
is_point_in_hexahedron_using_raycasting(const xyz::Point &point,
                                        const CellCorners &corners)
{

    // Get all corners in a more accessible format, and negate Z so it will be
    // right-handed
    std::array<xyz::Point, 8> cell_corners = {
        xyz::Point{ corners.upper_sw.x, corners.upper_sw.y, -corners.upper_sw.z },
        xyz::Point{ corners.upper_se.x, corners.upper_se.y, -corners.upper_se.z },
        xyz::Point{ corners.upper_ne.x, corners.upper_ne.y, -corners.upper_ne.z },
        xyz::Point{ corners.upper_nw.x, corners.upper_nw.y, -corners.upper_nw.z },
        xyz::Point{ corners.lower_sw.x, corners.lower_sw.y, -corners.lower_sw.z },
        xyz::Point{ corners.lower_se.x, corners.lower_se.y, -corners.lower_se.z },
        xyz::Point{ corners.lower_ne.x, corners.lower_ne.y, -corners.lower_ne.z },
        xyz::Point{ corners.lower_nw.x, corners.lower_nw.y, -corners.lower_nw.z }
    };

    // change Point also to right-handed
    xyz::Point point_rh = { point.x, point.y, -point.z };

    // Define the 6 faces of the hexahedron (each face is 4 corners)
    // Face indices: 0=top, 1=bottom, 2=front, 3=right, 4=back, 5=left
    std::array<std::array<int, 4>, 6> faces = { {
      { 0, 1, 2, 3 },  // top face (upper)
      { 4, 5, 6, 7 },  // bottom face (lower)
      { 0, 1, 5, 4 },  // front face
      { 1, 2, 6, 5 },  // right face
      { 2, 3, 7, 6 },  // back face
      { 3, 0, 4, 7 }   // left face
    } };

    // Ray casting: count intersections along a ray in +X direction
    int intersections = 0;

    // Check each face for intersection with the ray from point to +X infinity
    for (const auto &face : faces) {
        // Get the 4 corners of this face
        xyz::Point a = cell_corners[face[0]];
        xyz::Point b = cell_corners[face[1]];
        xyz::Point c = cell_corners[face[2]];
        xyz::Point d = cell_corners[face[3]];

        // For a ray in +X direction, we need AT LEAST ONE corner has X > point.x
        // the right) We don't need points on the left for the face to possibly
        // intersect the ray
        bool has_point_right = (a.x > point_rh.x || b.x > point_rh.x ||
                                c.x > point_rh.x || d.x > point_rh.x);

        // Check if face's YZ range contains point's YZ coordinates
        bool y_in_range = (std::min({ a.y, b.y, c.y, d.y }) <= point_rh.y) &&
                          (std::max({ a.y, b.y, c.y, d.y }) >= point_rh.y);
        bool z_in_range = (std::min({ a.z, b.z, c.z, d.z }) <= point_rh.z) &&
                          (std::max({ a.z, b.z, c.z, d.z }) >= point_rh.z);

        // To potentially intersect with ray, we need:
        // 1. At least one point to the right of the ray origin
        // 2. The point's YZ position is within the face's YZ extents
        bool may_intersect = has_point_right && y_in_range && z_in_range;

        // Special case: if face is in a plane parallel to ray (all X values equal)
        if (a.x == b.x && b.x == c.x && c.x == d.x && a.x > point_rh.x) {
            // If all points are at the same X coordinate AND that X is greater than
            // the point's X, we may have an intersection
            may_intersect = y_in_range && z_in_range;
        }

        // If the face may intersect, do detailed intersection test
        if (may_intersect) {
            // For planar quadrilateral faces, we should count only one
            // intersection
            // regardless of how we split it into triangles
            bool intersected = false;

            // Test if either triangle is intersected
            if (geometry::ray_intersects_triangle(point_rh, { 1, 0, 0 }, a, b, c) ||
                geometry::ray_intersects_triangle(point_rh, { 1, 0, 0 }, a, c, d)) {
                intersections++;
                intersected = true;
            }
        }
    }
    // If the number of intersections is odd, the point is inside
    return (intersections % 2) == 1;
}

// =====================================================================================
// Tetrahedrical calculations for a point inside a hexahedron
// =====================================================================================
/**
 * @brief Special method for difficult cases

 */

static bool
is_point_in_hexahedron_using_centroid_tetrahedrons(const xyz::Point &point,
                                                   const CellCorners &corners)
{
    // Get all corners in a more accessible format (right-handed)
    std::array<xyz::Point, 8> vertices = {
        xyz::Point{ corners.upper_sw.x, corners.upper_sw.y, -corners.upper_sw.z },
        xyz::Point{ corners.upper_se.x, corners.upper_se.y, -corners.upper_se.z },
        xyz::Point{ corners.upper_ne.x, corners.upper_ne.y, -corners.upper_ne.z },
        xyz::Point{ corners.upper_nw.x, corners.upper_nw.y, -corners.upper_nw.z },
        xyz::Point{ corners.lower_sw.x, corners.lower_sw.y, -corners.lower_sw.z },
        xyz::Point{ corners.lower_se.x, corners.lower_se.y, -corners.lower_se.z },
        xyz::Point{ corners.lower_ne.x, corners.lower_ne.y, -corners.lower_ne.z },
        xyz::Point{ corners.lower_nw.x, corners.lower_nw.y, -corners.lower_nw.z }
    };

    // Convert point to right-handed
    xyz::Point point_rh = { point.x, point.y, -point.z };

    // Calculate the centroid of the hexahedron
    xyz::Point centroid = { 0.0, 0.0, 0.0 };
    for (const auto &v : vertices) {
        centroid.x += v.x;
        centroid.y += v.y;
        centroid.z += v.z;
    }
    centroid.x /= 8.0;
    centroid.y /= 8.0;
    centroid.z /= 8.0;

    // Use the CENTROID_TETRAHEDRON_SCHEME defined in geometry.hpp
    for (int scheme = 0; scheme < 2; ++scheme) {
        for (size_t i = 0; i < 8; ++i) {
            const auto &tetra = CENTROID_TETRAHEDRON_SCHEME[scheme][i];
            // Note: We're explicitly handling the -1 case (which means use centroid)
            const xyz::Point &v0 = vertices[tetra[0]];
            const xyz::Point &v1 = vertices[tetra[1]];
            const xyz::Point &v2 = vertices[tetra[2]];

            // The fourth vertex is always the centroid, as specified by -1
            if (is_point_in_tetrahedron(point_rh, v0, v1, v2, centroid)) {
                return true;
            }
        }
    }

    return false;
}
/**
 * Determines if a point is inside a hexahedron (8-vertex cell).
 *
 * @param point The point to test
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 */
static bool
is_point_in_hexahedron_using_tetrahedrons(const xyz::Point &point,
                                          const CellCorners &corners)
{

    // Get all corners in a more accessible format,
    //   and negate Z so it will be right - handed
    std::array<xyz::Point, 8> vertices = {
        xyz::Point{ corners.upper_sw.x, corners.upper_sw.y, -corners.upper_sw.z },
        xyz::Point{ corners.upper_se.x, corners.upper_se.y, -corners.upper_se.z },
        xyz::Point{ corners.upper_ne.x, corners.upper_ne.y, -corners.upper_ne.z },
        xyz::Point{ corners.upper_nw.x, corners.upper_nw.y, -corners.upper_nw.z },
        xyz::Point{ corners.lower_sw.x, corners.lower_sw.y, -corners.lower_sw.z },
        xyz::Point{ corners.lower_se.x, corners.lower_se.y, -corners.lower_se.z },
        xyz::Point{ corners.lower_ne.x, corners.lower_ne.y, -corners.lower_ne.z },
        xyz::Point{ corners.lower_nw.x, corners.lower_nw.y, -corners.lower_nw.z }
    };

    // change Point also to right-handed
    xyz::Point point_rh = { point.x, point.y, -point.z };

    size_t score = 0;
    // Loop through all defined tetrahedron decomposition schemes

    for (int scheme = 0; scheme < 4; ++scheme) {
        for (size_t i = 0; i < 6; ++i) {
            const auto &tetra_indices = TETRAHEDRON_SCHEMES[scheme][i];
            if (is_point_in_tetrahedron(
                  point_rh, vertices[tetra_indices[0]], vertices[tetra_indices[1]],
                  vertices[tetra_indices[2]], vertices[tetra_indices[3]])) {
                // If the point is found in any tetrahedron of any scheme, return
                // true
                score += 1;
            }
        }
    }

    if (score > 2) {
        // If the point is found in more than 2 tetrahedra, it is likely inside the
        // hexahedron
        return true;
    }
    if (score == 2) {
        // In doubth here if the point is inside or outside the hexahedron, try the
        // centroid method
        return is_point_in_hexahedron_using_centroid_tetrahedrons(point, corners);
    }
    return false;
}  // is_point_in_hexahedron_using_tetrahedrons

/**
 * @brief A central function where one can select appropriate method
 */
bool
is_point_in_hexahedron(const xyz::Point &point,
                       const CellCorners &corners,
                       const std::string &method)
{

    // Quick rejection test using bounding box; this is independent of the method
    auto [min_point, max_point] = grid3d::get_cell_bounding_box(corners);
    double epsilon =
      1e-8 * std::max({ max_point.x - min_point.x, max_point.y - min_point.y,
                        max_point.z - min_point.z });

    // Use an epsilon for the bounding box check to handle numerical precision
    if (point.x < min_point.x - epsilon || point.x > max_point.x + epsilon ||
        point.y < min_point.y - epsilon || point.y > max_point.y + epsilon ||
        point.z < min_point.z - epsilon || point.z > max_point.z + epsilon) {
        return false;
    }

    if (method == "ray_casting") {
        return is_point_in_hexahedron_using_raycasting(point, corners);
    } else if (method == "tetrahedrons") {
        return is_point_in_hexahedron_using_tetrahedrons(point, corners);
    } else if (method == "centroid_tetrahedrons") {
        return is_point_in_hexahedron_using_centroid_tetrahedrons(point, corners);
    } else {
        throw std::invalid_argument("Invalid method for point-in-cell test");
    }
}  // is_point_in_tetrahedron

}  // namespace geometry
