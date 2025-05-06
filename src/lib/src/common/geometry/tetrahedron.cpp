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

/**
 * Helper function to calculate relative tolerance based on RELATIVE_EPSILON.
 */
static double
calculate_relative_tolerance(const std::array<double, 5> &volumes,
                             double relative_epsilon)
{
    // Find the maximum absolute volume
    double max_volume = 0.0;
    for (const auto &volume : volumes) {
        max_volume = std::max(max_volume, std::abs(volume));
    }

    // Compute the relative tolerance
    return relative_epsilon * max_volume;
}

/**
 * Determines if a point is inside a tetrahedron using barycentric coordinates.
 * This version is robust against both left-handed and right-handed coordinate systems.
 */
static bool
is_point_in_tetrahedron_signedsum(const xyz::Point &p,
                                  const xyz::Point &a,
                                  const xyz::Point &b,
                                  const xyz::Point &c,
                                  const xyz::Point &d)
{

    // auto &logger =
    //   xtgeo::logging::LoggerManager::get("geometry::is_point_in_tetrahedron");

    const double RELATIVE_EPSILON = 1e-12;

    // Calculate the signed volume of the tetrahedron
    double v0 = signed_tetrahedron_volume(a, b, c, d);

    // Calculate the signed volumes of the sub-tetrahedra
    double v1 = signed_tetrahedron_volume(p, b, c, d);
    double v2 = signed_tetrahedron_volume(a, p, c, d);
    double v3 = signed_tetrahedron_volume(a, b, p, d);
    double v4 = signed_tetrahedron_volume(a, b, c, p);

    // logger.debug("Tetrahedron volumes: v0 = {}, v1 = {}, v2 = {}, v3 = {}, v4 = {}",
    // v0,
    //              v1, v2, v3, v4);

    // Calculate the relative tolerance
    std::array<double, 5> volumes = { v0, v1, v2, v3, v4 };
    double relative_tolerance = calculate_relative_tolerance(volumes, RELATIVE_EPSILON);

    printf("------------------ Relative tolerance: %f\n", relative_tolerance);

    // Check for degenerate tetrahedron
    if (std::abs(v0) < relative_tolerance) {
        // logger.debug("Degenerate tetrahedron detected (v0 = {})", v0);

        // Degenerate tetrahedron: Check if the point lies in the plane of the
        // tetrahedron
        xyz::Point normal = calculate_normal(a, b, c);
        double plane_distance = dot_product(normal, subtract(p, a));

        // If the point is not in the plane, it's outside
        if (std::abs(plane_distance) > relative_tolerance) {
            return false;
        }

        // Perform a 2D point-in-polygon test in the plane
        return is_point_in_triangle_2d(p, a, b, c) ||
               is_point_in_triangle_2d(p, a, c, d);
    }

    // Use consistent signs regardless of coordinate system handedness
    double sign = v0 > 0 ? 1.0 : -1.0;

    // Point is inside if all sub-volumes have the same sign as the main volume
    return ((v1 * sign >= -relative_tolerance) && (v2 * sign >= -relative_tolerance) &&
            (v3 * sign >= -relative_tolerance) && (v4 * sign >= -relative_tolerance));
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
                                    const xyz::Point &d)
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
    const double EPSILON = 1e-12;  // Tolerance for numerical precision
    return (alpha >= -EPSILON && beta >= -EPSILON && gamma >= -EPSILON &&
            delta >= -EPSILON);
}

bool
is_point_in_tetrahedron(const xyz::Point &point,
                        const xyz::Point &a,
                        const xyz::Point &b,
                        const xyz::Point &c,
                        const xyz::Point &d,
                        const std::string &method)
{
    // Check if the method is valid
    if (method != "barycentric" && method != "signedsum") {
        throw std::invalid_argument("Invalid method: " + method);
    }

    // Select the appropriate method
    if (method == "barycentric") {
        return is_point_in_tetrahedron_barycentric(point, a, b, c, d);
    } else {
        return is_point_in_tetrahedron_signedsum(point, a, b, c, d);
    }
}
}  // namespace xtgeo::geometry
