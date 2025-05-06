#include <cmath>   // Required for fabs
#include <limits>  // Required for numeric_limits
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using grid3d::CellCorners;
using xyz::Point;

// Helper function for vector subtraction
static xyz::Point
subtract(const xyz::Point &a, const xyz::Point &b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

// Helper function for cross product
static xyz::Point
cross_product(const xyz::Point &a, const xyz::Point &b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

// Helper function for dot product
static double
dot_product(const xyz::Point &a, const xyz::Point &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Helper function to calculate magnitude squared (avoid sqrt)
static double
magnitude_squared(const xyz::Point &v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/**
 * @brief Determines if a point is inside or on the boundary of a hexahedron using plane
 * side tests. This method triangulates each face and checks if the point lies on the
 * inner side of all 12 resulting triangles. It uses the centroid to consistently orient
 * face normals outwards. Handles potential numerical precision issues and degenerate
 * triangles.
 *
 * @param point The point to test.
 * @param vertices The 8 vertices of the hexahedron in standard order:
 *        0-3: upper face (sw, se, ne, nw)
 *        4-7: lower face (sw, se, ne, nw)
 * @return true if the point is inside or on the boundary, false otherwise.
 */
static bool
is_point_in_hexahedron_using_planes(const xyz::Point &point,
                                    const std::array<xyz::Point, 8> &vertices,
                                    const Point &min_pt,
                                    const Point &max_pt)
{
    double diagonal =
      std::sqrt(std::pow(max_pt.x - min_pt.x, 2) + std::pow(max_pt.y - min_pt.y, 2) +
                std::pow(max_pt.z - min_pt.z, 2));
    // If diagonal is zero (degenerate hexahedron), handle appropriately
    if (diagonal < std::numeric_limits<double>::epsilon()) {
        // All vertices are coincident. Point is inside only if it matches the vertex.
        return magnitude_squared(subtract(point, vertices[0])) <
               std::numeric_limits<double>::epsilon();
    }
    const double EPSILON = 1e-9 * diagonal;  // Relative tolerance

    // Define the 6 faces of the hexahedron by vertex indices
    // Ensure consistent winding order for outward normals (e.g., counter-clockwise when
    // viewed from outside)
    //      3----2
    //     /|   /|
    //    0----1 |   Upper face (z+)
    //    | 7--|-6
    //    |/   |/
    //    4----5     Lower face (z-)
    //
    const std::array<std::array<int, 4>, 6> faces = { {
      { 0, 1, 2, 3 },  // Top (viewed from +Z)
      { 4, 7, 6, 5 },  // Bottom (viewed from -Z) - Corrected
      { 0, 4, 5, 1 },  // Front (viewed from +Y) - Corrected
      { 1, 5, 6, 2 },  // Right (viewed from +X) - Corrected
      { 2, 6, 7, 3 },  // Back (viewed from -Y) - Corrected
      { 3, 7, 4, 0 }   // Left (viewed from -X) - Corrected
    } };

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

    // Check the point against the planes defined by the triangles of each face
    for (const auto &face_indices : faces) {
        // Split the quadrilateral face into two triangles (using the first vertex)
        const std::array<std::array<int, 3>, 2> triangles = {
            { { face_indices[0], face_indices[1], face_indices[2] },
              { face_indices[0], face_indices[2], face_indices[3] } }
        };

        for (const auto &triangle_indices : triangles) {
            const xyz::Point &p0 = vertices[triangle_indices[0]];
            const xyz::Point &p1 = vertices[triangle_indices[1]];
            const xyz::Point &p2 = vertices[triangle_indices[2]];

            // Calculate the triangle normal
            xyz::Point edge1 = subtract(p1, p0);
            xyz::Point edge2 = subtract(p2, p0);
            xyz::Point normal = cross_product(edge1, edge2);

            // Check for degenerate triangles (zero area). Normal magnitude squared is
            // proportional to area squared. Use a tolerance slightly larger than
            // machine epsilon for squared values.
            if (magnitude_squared(normal) < EPSILON * EPSILON * 1e-6) {
                // Skip degenerate triangles. If a point lies exactly on a degenerate
                // face, other non-degenerate faces should still correctly classify it.
                continue;
            }

            // Orient the normal outwards using the centroid.
            // Check if the centroid is on the same side as the normal direction
            // relative to the plane origin p0.
            xyz::Point p0_to_centroid = subtract(centroid, p0);
            if (dot_product(normal, p0_to_centroid) > 0) {
                // Normal is pointing inwards relative to the centroid, flip it
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }

            // Check if the test point is on the outer side of the plane defined by the
            // triangle. Calculate signed distance: dot(normal, point - p0)
            xyz::Point p0_to_point = subtract(point, p0);
            double dist = dot_product(normal, p0_to_point);

            // If point is clearly outside (positive distance beyond tolerance, given
            // outward normal)
            if (dist > EPSILON) {
                return false;  // Point is outside this face plane
            }
        }
    }

    // If the point was not outside any face plane (i.e., dist <= EPSILON for all), it's
    // inside or on the boundary.
    return true;
}

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

// =====================================================================================
// Ray casting calculations for a point inside a hexahedron
// =====================================================================================
/**
 * @brief Using ray casting method to determine if a point is inside a hexahedron.
 */
static bool
is_point_in_hexahedron_using_raycasting(const xyz::Point &point_rh,
                                        const std::array<xyz::Point, 8> &vertices)
{

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
        xyz::Point a = vertices[face[0]];
        xyz::Point b = vertices[face[1]];
        xyz::Point c = vertices[face[2]];
        xyz::Point d = vertices[face[3]];

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
is_point_in_non_convex_hexahedron(const xyz::Point &point,
                                  const std::array<xyz::Point, 8> &vertices)
{

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

    // Dynamically decompose the hexahedron into tetrahedrons
    const std::array<std::array<int, 4>, 12> dynamic_tetrahedrons = {
        // Top and bottom faces
        std::array<int, 4>{ 0, 1, 2, 4 }, std::array<int, 4>{ 0, 2, 3, 4 },
        std::array<int, 4>{ 4, 5, 6, 7 }, std::array<int, 4>{ 4, 6, 7, 0 },
        // Front and back faces
        std::array<int, 4>{ 0, 1, 5, 4 }, std::array<int, 4>{ 1, 2, 6, 5 },
        std::array<int, 4>{ 2, 3, 7, 6 }, std::array<int, 4>{ 3, 0, 4, 7 },
        // Centroid-based tetrahedrons
        std::array<int, 4>{ 0, 1, 2, 8 }, std::array<int, 4>{ 2, 3, 0, 8 },
        std::array<int, 4>{ 4, 5, 6, 8 }, std::array<int, 4>{ 6, 7, 4, 8 }
    };

    // Check if the point is inside any tetrahedron
    for (const auto &tetra : dynamic_tetrahedrons) {
        if (is_point_in_tetrahedron(point, vertices[tetra[0]], vertices[tetra[1]],
                                    vertices[tetra[2]], centroid)) {
            return true;
        }
    }

    return false;
}
/**
 * @brief Determines if a point is inside a hexahedron using signed volume tests.
 * @param point The point to test.
 * @param vertices The 8 vertices of the hexahedron.
 * @return true if the point is inside or on the boundary of the hexahedron, false
 * otherwise.
 */
static bool
is_point_in_hexahedron_using_signed_volume(const xyz::Point &point,
                                           const std::array<xyz::Point, 8> &vertices)
{
    // Helper function to calculate the signed volume of a tetrahedron
    auto signed_volume = [](const xyz::Point &p1, const xyz::Point &p2,
                            const xyz::Point &p3, const xyz::Point &p4) -> double {
        return (1.0 / 6.0) *
               ((p2.x - p1.x) *
                  ((p3.y - p1.y) * (p4.z - p1.z) - (p3.z - p1.z) * (p4.y - p1.y)) -
                (p2.y - p1.y) *
                  ((p3.x - p1.x) * (p4.z - p1.z) - (p3.z - p1.z) * (p4.x - p1.x)) +
                (p2.z - p1.z) *
                  ((p3.x - p1.x) * (p4.y - p1.y) - (p3.y - p1.y) * (p4.x - p1.x)));
    };

    // Decompose the hexahedron into 6 tetrahedrons
    const std::array<std::array<int, 4>, 6> tetrahedrons = { {
      { 0, 1, 2, 4 },  // Top face and lower_sw
      { 1, 2, 3, 5 },  // Top face and lower_se
      { 2, 3, 0, 6 },  // Top face and lower_ne
      { 3, 0, 1, 7 },  // Top face and lower_nw
      { 4, 5, 6, 7 },  // Bottom face
      { 0, 1, 2, 3 }   // Top face
    } };

    // Calculate the signed volume of the hexahedron
    double hexahedron_volume = 0.0;
    for (const auto &tetra : tetrahedrons) {
        hexahedron_volume +=
          std::abs(signed_volume(vertices[tetra[0]], vertices[tetra[1]],
                                 vertices[tetra[2]], vertices[tetra[3]]));
    }

    // Calculate the sum of signed volumes of tetrahedrons formed with the point
    double point_volume_sum = 0.0;
    for (const auto &tetra : tetrahedrons) {
        point_volume_sum += std::abs(signed_volume(
          point, vertices[tetra[1]], vertices[tetra[2]], vertices[tetra[3]]));
    }

    // If the sum of the point volumes equals the hexahedron volume, the point is inside
    const double EPSILON = 1e-8;
    return std::abs(point_volume_sum - hexahedron_volume) < EPSILON;
}

static xyz::Point
calculate_normal(const xyz::Point &p1, const xyz::Point &p2, const xyz::Point &p3)
{
    xyz::Point u = { p2.x - p1.x, p2.y - p1.y, p2.z - p1.z };
    xyz::Point v = { p3.x - p1.x, p3.y - p1.y, p3.z - p1.z };
    return { u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x };
}

static double
calculate_planarity(const xyz::Point &p1,
                    const xyz::Point &p2,
                    const xyz::Point &p3,
                    const xyz::Point &p4)
{
    xyz::Point normal1 = calculate_normal(p1, p2, p3);
    xyz::Point normal2 = calculate_normal(p1, p3, p4);
    double dot_product =
      normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z;
    double magnitude1 =
      std::sqrt(normal1.x * normal1.x + normal1.y * normal1.y + normal1.z * normal1.z);
    double magnitude2 =
      std::sqrt(normal2.x * normal2.x + normal2.y * normal2.y + normal2.z * normal2.z);
    return std::abs(dot_product /
                    (magnitude1 * magnitude2));  // Cosine of the angle between normals
}
static double
calculate_dip_direction(const xyz::Point &p1,
                        const xyz::Point &p2,
                        const xyz::Point &p3,
                        const xyz::Point &p4)
{
    double avg_z = (p1.z + p2.z + p3.z + p4.z) / 4.0;
    return avg_z;  // Higher avg_z indicates an upward slope
}

/**
 * @brief Local function to select best scheme for tetrahedrons base on planarity
 * and elevation differences.
 */
static int
select_best_scheme(const std::array<xyz::Point, 8> &vertices)
{
    // Calculate elevation differences
    double diag0_2_top = std::abs(vertices[0].z - vertices[2].z);
    double diag1_3_top = std::abs(vertices[1].z - vertices[3].z);
    double diag4_6_base = std::abs(vertices[4].z - vertices[6].z);
    double diag5_7_base = std::abs(vertices[5].z - vertices[7].z);

    // Calculate planarity scores
    double planarity_top_diag0_2 =
      calculate_planarity(vertices[0], vertices[1], vertices[2], vertices[3]);
    double planarity_top_diag1_3 =
      calculate_planarity(vertices[0], vertices[1], vertices[3], vertices[2]);
    double planarity_base_diag4_6 =
      calculate_planarity(vertices[4], vertices[5], vertices[6], vertices[7]);
    double planarity_base_diag5_7 =
      calculate_planarity(vertices[4], vertices[5], vertices[7], vertices[6]);

    // Calculate dip directions
    double dip_top =
      calculate_dip_direction(vertices[0], vertices[1], vertices[2], vertices[3]);
    double dip_base =
      calculate_dip_direction(vertices[4], vertices[5], vertices[6], vertices[7]);

    // Evaluate each scheme
    double scheme0_score = diag0_2_top + diag4_6_base + (1.0 - planarity_top_diag0_2) +
                           (1.0 - planarity_base_diag4_6);
    double scheme1_score = diag0_2_top + diag5_7_base + (1.0 - planarity_top_diag0_2) +
                           (1.0 - planarity_base_diag5_7);
    double scheme2_score = diag1_3_top + diag4_6_base + (1.0 - planarity_top_diag1_3) +
                           (1.0 - planarity_base_diag4_6);
    double scheme3_score = diag1_3_top + diag5_7_base + (1.0 - planarity_top_diag1_3) +
                           (1.0 - planarity_base_diag5_7);

    // Find the scheme with the minimum score
    double scores[4] = { scheme0_score, scheme1_score, scheme2_score, scheme3_score };
    int best_scheme = 0;
    for (int i = 1; i < 4; ++i) {
        if (scores[i] < scores[best_scheme]) {
            best_scheme = i;
        }
    }

    return best_scheme;
}

/**
 * Determines if a point is inside a hexahedron (8-vertex cell).
 *
 * @param point The point to test
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 */
static bool
is_point_in_hexahedron_using_tetrahedrons(const xyz::Point &point_rh,
                                          const std::array<xyz::Point, 8> &vertices)
{

    // Select the best scheme
    int best_scheme = select_best_scheme(vertices);

    // Use the selected scheme to check if the point is inside
    for (size_t i = 0; i < 6; ++i) {
        const auto &tetra = TETRAHEDRON_SCHEMES[best_scheme][i];
        if (is_point_in_tetrahedron(point_rh, vertices[tetra[0]], vertices[tetra[1]],
                                    vertices[tetra[2]], vertices[tetra[3]])) {
            return true;
        }
    }

    return false;
}
/**
 * @brief A central function where one can select appropriate method
 * for point-in-cell test.
 * @param point The point to test, negated Z compared to python
 * @param corners The 8 corners of the hexahedron
 * @param method The method to use for the test
 * @return true if the point is inside the hexahedron, false otherwise
 * @throws std::invalid_argument if the method is not recognized
 */
bool
is_point_in_hexahedron(const xyz::Point &point,
                       const HexahedronCorners &hexahedron_corners,
                       const std::string &method)
{

    // Quick rejection test using bounding box; this is independent of the method
    auto [min_point, max_point] = get_hexahedron_bounding_box(hexahedron_corners);

    double epsilon =
      1e-8 * std::max({ max_point.x - min_point.x, max_point.y - min_point.y,
                        max_point.z - min_point.z });

    // Use an epsilon for the bounding box check to handle numerical precision
    if (point.x < min_point.x - epsilon || point.x > max_point.x + epsilon ||
        point.y < min_point.y - epsilon || point.y > max_point.y + epsilon ||
        point.z < min_point.z - epsilon || point.z > max_point.z + epsilon) {
        return false;
    }

    std::array<xyz::Point, 8> vertices = {
        hexahedron_corners.upper_sw, hexahedron_corners.upper_se,
        hexahedron_corners.upper_ne, hexahedron_corners.upper_nw,
        hexahedron_corners.lower_sw, hexahedron_corners.lower_se,
        hexahedron_corners.lower_ne, hexahedron_corners.lower_nw
    };

    if (method == "ray_casting") {
        return is_point_in_hexahedron_using_raycasting(point, vertices);
    } else if (method == "tetrahedrons") {
        return is_point_in_hexahedron_using_tetrahedrons(point, vertices);
    } else if (method == "non_convex") {
        return is_point_in_non_convex_hexahedron(point, vertices);
    } else if (method == "using_planes") {
        return is_point_in_hexahedron_using_planes(point, vertices, min_point,
                                                   max_point);
    } else if (method == "score_based") {
        // evaluate using 4 methods and return the majority, except for cells that are
        // clearly concave in projected plane, or very thin
        if (is_hexahedron_thin(hexahedron_corners, 1e-6)) {
            // Check if the cell is thin, using planes
            return is_point_in_hexahedron_using_planes(point, vertices, min_point,
                                                       max_point);
        }

        if (is_hexahedron_concave_projected(hexahedron_corners)) {
            // Check if the cell is non-convex, using special method
            return is_point_in_non_convex_hexahedron(point, vertices);
        }
        int count_ray = 0;
        int count_tetra = 0;
        int count_non_convex = 0;
        count_ray += is_point_in_hexahedron_using_raycasting(point, vertices);
        count_tetra += is_point_in_hexahedron_using_tetrahedrons(point, vertices);
        count_non_convex += is_point_in_non_convex_hexahedron(point, vertices);
        int count_sum = count_ray + count_tetra + count_non_convex;
        if (count_sum == 1 || count_sum == 2) {
            int count_sign = is_point_in_hexahedron_using_planes(point, vertices,
                                                                 min_point, max_point);
            count_sum += count_sign;
        }
        return count_sum >= 2;
    } else {
        throw std::invalid_argument("Invalid method for point-in-cell test");
    }
}  // is_point_in_hexahedron

}  // namespace geometry
