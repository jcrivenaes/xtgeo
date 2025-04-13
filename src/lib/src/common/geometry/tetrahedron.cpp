#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using grid3d::CellCorners;
using xyz::Point;

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
    const double EPSILON = 1e-10;

    // Calculate the barycentric coordinates using signed volumes
    double v0 = signed_tetrahedron_volume(a, b, c, d);

    // Check for degenerate tetrahedron
    if (std::abs(v0) < EPSILON) {
        return false;
    }

    double v1 = signed_tetrahedron_volume(p, b, c, d);
    double v2 = signed_tetrahedron_volume(a, p, c, d);
    double v3 = signed_tetrahedron_volume(a, b, p, d);
    double v4 = signed_tetrahedron_volume(a, b, c, p);

    // Important: Use consistent signs regardless of coordinate system handedness
    // We only care that all sub-volumes have the same sign as the full volume
    // This works for both left-handed and right-handed systems
    double sign = v0 > 0 ? 1.0 : -1.0;

    // Point is inside if all sub-volumes have the same sign as the main volume
    return ((v1 * sign >= -EPSILON) && (v2 * sign >= -EPSILON) &&
            (v3 * sign >= -EPSILON) && (v4 * sign >= -EPSILON));
}

}  // namespace geometry
