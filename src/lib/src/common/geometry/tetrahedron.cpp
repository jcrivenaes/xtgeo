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
bool
is_point_in_tetrahedron(const xyz::Point &p,
                        const xyz::Point &a,
                        const xyz::Point &b,
                        const xyz::Point &c,
                        const xyz::Point &d)
{

    // auto &logger =
    //   xtgeo::logging::LoggerManager::get("geometry::is_point_in_tetrahedron");

    const double RELATIVE_EPSILON = 1e-8;

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

}  // namespace xtgeo::geometry
