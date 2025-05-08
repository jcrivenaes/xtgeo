#include <algorithm>  // For std::min and std::max
#include <array>      // For std::array
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using grid3d::CellCorners;
using xyz::Point;
/**
 * Calculate the dot product of two vectors.
 */
static double
dot_product(const xyz::Point &a, const xyz::Point &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * Check if a point is inside a triangle in 2D space.
 */
static bool
is_point_in_triangle_2d(const xyz::Point &p,
                        const xyz::Point &a,
                        const xyz::Point &b,
                        const xyz::Point &c)
{
    // Calculate barycentric coordinates in 2D
    double area =
      0.5 * (-b.y * c.x + a.y * (-b.x + c.x) + a.x * (b.y - c.y) + b.x * c.y);
    double s = 1.0 / (2.0 * area) *
               (a.y * c.x - a.x * c.y + (c.y - a.y) * p.x + (a.x - c.x) * p.y);
    double t = 1.0 / (2.0 * area) *
               (a.x * b.y - a.y * b.x + (a.y - b.y) * p.x + (b.x - a.x) * p.y);

    // Check if point is inside the triangle
    return s >= 0 && t >= 0 && (s + t) <= 1;
}

/**
 * Calculate the normal vector of a plane defined by three points.
 */
static xyz::Point
calculate_normal(const xyz::Point &a, const xyz::Point &b, const xyz::Point &c)
{
    // Calculate vectors
    xyz::Point ab = { b.x - a.x, b.y - a.y, b.z - a.z };
    xyz::Point ac = { c.x - a.x, c.y - a.y, c.z - a.z };

    // Compute the cross product
    return { ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z,
             ab.x * ac.y - ab.y * ac.x };
}

static xyz::Point
subtract(const xyz::Point &a, const xyz::Point &b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

static double
magnitude_squared(const xyz::Point &v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
/**
 * Calculate the bounding box of a tetrahedron defined by four points.
 * Returns a pair of points representing the minimum and maximum corners of the bounding
 * box.
 */
static std::pair<xyz::Point, xyz::Point>
get_tetrahedron_bounding_box(const xyz::Point &a,
                             const xyz::Point &b,
                             const xyz::Point &c,
                             const xyz::Point &d)
{
    // Initialize min and max points
    xyz::Point min_pt = { std::min({ a.x, b.x, c.x, d.x }),
                          std::min({ a.y, b.y, c.y, d.y }),
                          std::min({ a.z, b.z, c.z, d.z }) };

    xyz::Point max_pt = { std::max({ a.x, b.x, c.x, d.x }),
                          std::max({ a.y, b.y, c.y, d.y }),
                          std::max({ a.z, b.z, c.z, d.z }) };

    return { min_pt, max_pt };
}
/**
 * Calculate the signed volume of a tetrahedron.
 * Positive if vertices are in counterclockwise order when viewed from the first vertex.
 */
static double
signed_tetrahedron_volume(const xyz::Point &a,
                          const xyz::Point &b,
                          const xyz::Point &c,
                          const xyz::Point &d)
{
    // Calculate vectors from a to other points
    xyz::Point ab = { b.x - a.x, b.y - a.y, b.z - a.z };
    xyz::Point ac = { c.x - a.x, c.y - a.y, c.z - a.z };
    xyz::Point ad = { d.x - a.x, d.y - a.y, d.z - a.z };

    // Calculate the scalar triple product (ab · (ac × ad)) / 6
    double cross_x = ac.y * ad.z - ac.z * ad.y;
    double cross_y = ac.z * ad.x - ac.x * ad.z;
    double cross_z = ac.x * ad.y - ac.y * ad.x;

    return (ab.x * cross_x + ab.y * cross_y + ab.z * cross_z) / 6.0;
}

static bool
is_tetrahedron_degenerate(const xyz::Point &a,
                          const xyz::Point &b,
                          const xyz::Point &c,
                          const xyz::Point &d,
                          double tolerance = 1e-12)
{
    double volume = std::abs(signed_tetrahedron_volume(a, b, c, d));
    return volume < tolerance;
}
/**
 * Helper function to calculate relative tolerance based on volumes of tetrahedra.
 * Currently inactive, but kept for reference
 */
static double
calculate_max_volume(const std::array<double, 5> &volumes)
{
    // Find the maximum absolute volume
    double max_volume = 0.0;
    for (const auto &volume : volumes) {
        max_volume = std::max(max_volume, std::abs(volume));
    }

    return max_volume;
}

static double
calculate_bbox_diagonal(const xyz::Point &min_pt, const xyz::Point &max_pt)
{
    // Calculate the diagonal length of the bounding box
    double diagonal =
      std::sqrt(std::pow(max_pt.x - min_pt.x, 2) + std::pow(max_pt.y - min_pt.y, 2) +
                std::pow(max_pt.z - min_pt.z, 2));

    return diagonal;
}

/**
 * Determines if a point is inside a tetrahedron using signed volume method.
 */
static bool
is_point_in_tetrahedron_signedsum(const xyz::Point &p,
                                  const xyz::Point &a,
                                  const xyz::Point &b,
                                  const xyz::Point &c,
                                  const xyz::Point &d,
                                  const double epsilon)
{

    // Calculate the signed volume of the tetrahedron
    double v0 = signed_tetrahedron_volume(a, b, c, d);

    // Calculate the signed volumes of the sub-tetrahedra
    double v1 = signed_tetrahedron_volume(p, b, c, d);
    double v2 = signed_tetrahedron_volume(a, p, c, d);
    double v3 = signed_tetrahedron_volume(a, b, p, d);
    double v4 = signed_tetrahedron_volume(a, b, c, p);

    // Check for degenerate tetrahedron
    if (std::abs(v0) < epsilon) {
        // logger.debug("Degenerate tetrahedron detected (v0 = {})", v0);

        // Degenerate tetrahedron: Check if the point lies in the plane of the
        // tetrahedron
        xyz::Point normal = calculate_normal(a, b, c);
        double plane_distance = dot_product(normal, subtract(p, a));

        // If the point is not in the plane, it's outside
        if (std::abs(plane_distance) > epsilon) {
            return false;
        }

        // Perform a 2D point-in-polygon test in the plane
        return is_point_in_triangle_2d(p, a, b, c) ||
               is_point_in_triangle_2d(p, a, c, d);
    }

    // Use consistent signs regardless of coordinate system handedness
    double sign = v0 > 0 ? 1.0 : -1.0;

    // Point is inside if all sub-volumes have the same sign as the main volume
    return ((v1 * sign >= -epsilon) && (v2 * sign >= -epsilon) &&
            (v3 * sign >= -epsilon) && (v4 * sign >= -epsilon));
}

/**
 * @brief Determines if a point is inside or on the edge of a tetrahedron using
 * barycentric coordinates.
 * @param p The point to check.
 * @param a, b, c, d The vertices of the tetrahedron.
 * @return bool True if the point is inside or on the edge, false otherwise.
 */
static bool
is_point_in_tetrahedron_barycentric(const xyz::Point &p,
                                    const xyz::Point &a,
                                    const xyz::Point &b,
                                    const xyz::Point &c,
                                    const xyz::Point &d,
                                    const double epsilon)
{
    // Helper function to compute the determinant of a 3x3 matrix
    auto determinant = [](const xyz::Point &u, const xyz::Point &v,
                          const xyz::Point &w) -> double {
        return u.x * (v.y * w.z - v.z * w.y) - u.y * (v.x * w.z - v.z * w.x) +
               u.z * (v.x * w.y - v.y * w.x);
    };
    // Compute vectors
    xyz::Point ap = { p.x - a.x, p.y - a.y, p.z - a.z };
    xyz::Point ab = { b.x - a.x, b.y - a.y, b.z - a.z };
    xyz::Point ac = { c.x - a.x, c.y - a.y, c.z - a.z };
    xyz::Point ad = { d.x - a.x, d.y - a.y, d.z - a.z };

    // Compute the volume of the tetrahedron (main determinant)
    double det_main = determinant(ab, ac, ad);

    // Compute barycentric coordinates
    double alpha = determinant(ap, ac, ad) / det_main;  // Weight for vertex a
    double beta = determinant(ab, ap, ad) / det_main;   // Weight for vertex b
    double gamma = determinant(ab, ac, ap) / det_main;  // Weight for vertex c
    double delta = 1.0 - alpha - beta - gamma;          // Weight for vertex d

    // Check if the point is inside or on the edge
    return (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon &&
            delta >= -epsilon);
}

bool
is_point_in_tetrahedron(const xyz::Point &point,
                        const xyz::Point &a,
                        const xyz::Point &b,
                        const xyz::Point &c,
                        const xyz::Point &d,
                        const double tolerance_scaler)
{

    auto [min_pt, max_pt] = get_tetrahedron_bounding_box(a, b, c, d);

    // Quick rejection test using bounding box
    if (point.x < min_pt.x || point.x > max_pt.x || point.y < min_pt.y ||
        point.y > max_pt.y || point.z < min_pt.z || point.z > max_pt.z) {
        return false;
    }

    // Calculate the diagonal length of the bounding box
    double diagonal = calculate_bbox_diagonal(min_pt, max_pt);

    // Calculate the practical epsilon based on the diagonal length, multiplied with a
    // practical factor of 1e6 (since numerics::EPSILON is very small, i.e. ~1e-16)
    double epsilon = numerics::EPSILON * tolerance_scaler * diagonal * 1e6;

    // If diagonal is zero (degenerate tetrahedron), handle appropriately
    if (diagonal < epsilon) {
        // All vertices are coincident. Point is inside only if it matches the vertex.
        return magnitude_squared(subtract(point, a)) < epsilon;
    }

    if (is_tetrahedron_degenerate(a, b, c, d)) {
        return is_point_in_tetrahedron_signedsum(point, a, b, c, d, epsilon);
    } else {
        return is_point_in_tetrahedron_barycentric(point, a, b, c, d, epsilon);
    }
}
}  // namespace xtgeo::geometry
