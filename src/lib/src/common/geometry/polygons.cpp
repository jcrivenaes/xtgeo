#include <Eigen/Dense>
#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <xtgeo/geometry.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {
/*
 * A generic function to estimate if a point is inside a polygon seen from bird view.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param polygon A vector of doubles, length 2 (N points in the polygon)
 * @return Boolean
 */

bool
is_xy_point_in_polygon(const double x, const double y, const xyz::Polygon &polygon)
{
    bool inside = false;
    int n = polygon.size();  // Define the variable n
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon.points[i].x, yi = polygon.points[i].y;
        double xj = polygon.points[j].x, yj = polygon.points[j].y;

        bool intersect =
          ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

static bool
does_line_segments_intersect(const Eigen::Vector3d &p1,
                             const Eigen::Vector3d &p2,
                             const Eigen::Vector3d &q1,
                             const Eigen::Vector3d &q2)
{
    // Direction vectors of the two line segments
    Eigen::Vector3d d1 = p2 - p1;
    Eigen::Vector3d d2 = q2 - q1;

    // Cross product of direction vectors
    Eigen::Vector3d cross_d1_d2 = d1.cross(d2);

    // Check if the two line segments are parallel
    if (cross_d1_d2.norm() < numerics::EPSILON) {
        // Line segments are parallel, check for collinearity
        Eigen::Vector3d p1_to_q1 = q1 - p1;
        if (p1_to_q1.cross(d1).norm() < numerics::EPSILON) {
            // Line segments are collinear, check for overlap
            double t1 = (q1 - p1).dot(d1) / d1.squaredNorm();
            double t2 = (q2 - p1).dot(d1) / d1.squaredNorm();
            return (t1 >= 0.0 && t1 <= 1.0) || (t2 >= 0.0 && t2 <= 1.0) ||
                   (t1 <= 0.0 && t2 >= 1.0);
        }
        return false;  // Parallel but not collinear
    }

    // Compute intersection parameters
    Eigen::Vector3d p1_to_q1 = q1 - p1;
    double t = p1_to_q1.cross(d2).dot(cross_d1_d2) / cross_d1_d2.squaredNorm();
    double u = p1_to_q1.cross(d1).dot(cross_d1_d2) / cross_d1_d2.squaredNorm();

    // Check if the intersection point lies within both line segments
    return (t >= 0.0 && t <= 1.0) && (u >= 0.0 && u <= 1.0);
}

static bool
does_line_segment_intersect_edges(const Eigen::Vector3d &p1,
                                  const Eigen::Vector3d &p2,
                                  const std::array<Eigen::Vector3d, 4> &quad)
{
    // Iterate over each edge of the quad
    for (int i = 0; i < 4; i++) {
        const Eigen::Vector3d &v1 = quad[i];
        const Eigen::Vector3d &v2 = quad[(i + 1) % 4];

        // Check if the line segment intersects the edge
        if (does_line_segments_intersect(p1, p2, v1, v2)) {
            return true;  // Intersection found
        }
    }

    return false;  // No intersection with any edge
}

// Helper function to check if a point is inside a quad
static bool
is_point_in_quad_winding(const Eigen::Vector3d &point,
                         const std::array<Eigen::Vector3d, 4> &quad)
{
    // Project onto most stable plane
    Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]).normalized();

    int max_comp = 0;
    for (int i = 1; i < 3; i++) {
        if (std::abs(normal[i]) > std::abs(normal[max_comp])) {
            max_comp = i;
        }
    }
    int comp1 = (max_comp + 1) % 3;
    int comp2 = (max_comp + 2) % 3;

    // Do 2D point-in-polygon test using winding number algorithm
    int winding_number = 0;
    for (int i = 0; i < 4; i++) {
        const auto &v1 = quad[i];
        const auto &v2 = quad[(i + 1) % 4];

        if (v1[comp2] <= point[comp2]) {
            if (v2[comp2] > point[comp2] &&
                ((v2[comp1] - v1[comp1]) * (point[comp2] - v1[comp2]) -
                 (point[comp1] - v1[comp1]) * (v2[comp2] - v1[comp2])) > 0) {
                ++winding_number;
            }
        } else {
            if (v2[comp2] <= point[comp2] &&
                ((v2[comp1] - v1[comp1]) * (point[comp2] - v1[comp2]) -
                 (point[comp1] - v1[comp1]) * (v2[comp2] - v1[comp2])) < 0) {
                --winding_number;
            }
        }
    }

    return winding_number != 0;
}
static bool
is_point_in_quad(const Eigen::Vector3d &point,
                 const std::array<Eigen::Vector3d, 4> &quad)
{
    // Project onto most stable plane
    Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]).normalized();

    // Debug output for normal and point
    std::cout << "Normal vector: " << normal.transpose() << std::endl;
    std::cout << "Testing point: " << point.transpose() << std::endl;

    int max_comp = 0;
    for (int i = 1; i < 3; i++) {
        if (std::abs(normal[i]) > std::abs(normal[max_comp])) {
            max_comp = i;
        }
    }
    int comp1 = (max_comp + 1) % 3;
    int comp2 = (max_comp + 2) % 3;

    std::cout << "Projection components: " << comp1 << ", " << comp2 << std::endl;

    // Alternative approach: Use barycentric coordinates
    // This is often more robust than the winding number algorithm

    // First, triangulate the quad
    // Triangle 1: quad[0], quad[1], quad[2]
    // Triangle 2: quad[0], quad[2], quad[3]

    // Check if point is in triangle 1
    Eigen::Vector3d v0 = quad[1] - quad[0];
    Eigen::Vector3d v1 = quad[2] - quad[0];
    Eigen::Vector3d v2 = point - quad[0];

    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);

    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    bool in_triangle1 = (u >= -1e-8) && (v >= -1e-8) && (w >= -1e-8);

    std::cout << "Barycentric coordinates (triangle 1): u=" << u << ", v=" << v
              << ", w=" << w << std::endl;
    std::cout << "In triangle 1: " << in_triangle1 << std::endl;

    // If not in triangle 1, check triangle 2
    if (!in_triangle1) {
        v0 = quad[2] - quad[0];
        v1 = quad[3] - quad[0];
        // v2 remains the same (point - quad[0])

        d00 = v0.dot(v0);
        d01 = v0.dot(v1);
        d11 = v1.dot(v1);
        d20 = v2.dot(v0);
        d21 = v2.dot(v1);

        denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;

        bool in_triangle2 = (u >= -1e-8) && (v >= -1e-8) && (w >= -1e-8);

        std::cout << "Barycentric coordinates (triangle 2): u=" << u << ", v=" << v
                  << ", w=" << w << std::endl;
        std::cout << "In triangle 2: " << in_triangle2 << std::endl;

        return in_triangle2;
    }

    return in_triangle1;
}

// bool
// does_line_segment_intersect_quad_internal(const Eigen::Vector3d &p1,
//                                           const Eigen::Vector3d &p2,
//                                           const std::array<Eigen::Vector3d, 4> &quad)
// {
//     printf("Enter does_line_segment_intersect_quad_internal\n");
//     // Get quad plane normal
//     Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);

//     // Check for degenerate quad (zero area)
//     if (normal.norm() < numerics::EPSILON) {
//         std::cout << "Degenerate quad detected (zero area)." << std::endl;
//         return false;  // Degenerate quad
//     }
//     normal.normalize();

//     // Get line direction
//     Eigen::Vector3d line_dir = p2 - p1;
//     double line_length = line_dir.norm();
//     if (line_length < numerics::EPSILON) {
//         std::cout << "Zero-length line segment detected." << std::endl;
//         return false;  // Zero-length line segment
//     }
//     line_dir /= line_length;

//     // Debugging: Log input points and quad corners
//     std::cout << "Line segment: p1 = " << p1.transpose() << ", p2 = " <<
//     p2.transpose()
//               << std::endl;
//     std::cout << "Quad corners: ";
//     for (const auto &corner : quad) {
//         std::cout << corner.transpose() << " ";
//     }
//     std::cout << std::endl;

//     // Special case: check if line segment lies along or near any edge of the quad
//     const double PARALLEL_TOLERANCE = 1e-10;  // Adjust this value if needed
//     for (int i = 0; i < 4; i++) {
//         const auto &v1 = quad[i];
//         const auto &v2 = quad[(i + 1) % 4];

//         // Vector along quad edge
//         Eigen::Vector3d edge = v2 - v1;
//         double edge_length = edge.norm();
//         if (edge_length < numerics::EPSILON)
//             continue;
//         edge /= edge_length;

//         // Check if line is parallel or nearly parallel to edge
//         double cross_norm = edge.cross(line_dir).norm();
//         if (cross_norm < PARALLEL_TOLERANCE) {
//             // Project endpoints onto edge line
//             double t1 = (p1 - v1).dot(edge);
//             double t2 = (p2 - v1).dot(edge);

//             // Debugging: Log edge and projection results
//             std::cout << "Edge " << i << ": v1 = " << v1.transpose()
//                       << ", v2 = " << v2.transpose() << std::endl;
//             std::cout << "Projections: t1 = " << t1 << ", t2 = " << t2 << std::endl;

//             // Check if projected points overlap with edge segment
//             if ((t1 >= -PARALLEL_TOLERANCE && t1 <= edge_length + PARALLEL_TOLERANCE)
//             ||
//                 (t2 >= -PARALLEL_TOLERANCE && t2 <= edge_length + PARALLEL_TOLERANCE)
//                 || (t1 <= -PARALLEL_TOLERANCE && t2 >= edge_length +
//                 PARALLEL_TOLERANCE)) {
//                 // Check distance from line to edge
//                 Eigen::Vector3d perpendicular =
//                   (p1 - v1) - ((p1 - v1).dot(edge)) * edge;
//                 if (perpendicular.norm() < PARALLEL_TOLERANCE) {
//                     std::cout << "Line segment is parallel and close to edge " << i
//                               << "." << std::endl;
//                     return true;
//                 }
//             }
//         }
//     }

//     // Normal plane-line intersection test
//     double dot_product = normal.dot(line_dir);
//     if (std::abs(dot_product) < PARALLEL_TOLERANCE) {
//         // Line is nearly parallel to quad plane
//         double dist1 = std::abs(normal.dot(p1 - quad[0]));
//         double dist2 = std::abs(normal.dot(p2 - quad[0]));

//         // Debugging: Log distances from plane
//         std::cout << "Line is nearly parallel to quad plane." << std::endl;
//         std::cout << "Distances from plane: dist1 = " << dist1 << ", dist2 = " <<
//         dist2
//                   << std::endl;

//         if (dist1 < PARALLEL_TOLERANCE && dist2 < PARALLEL_TOLERANCE) {
//             // Both points are nearly coplanar, check for edge intersection
//             std::cout << "Line segment is coplanar with the quad." << std::endl;
//             return does_line_segment_intersect_edges(p1, p2, quad);
//         }
//         return false;  // Line is parallel but not coplanar
//     }

//     // Calculate intersection point with plane
//     double d = normal.dot(quad[0]);
//     double t = (d - normal.dot(p1)) / dot_product;

//     // Debugging: Log intersection parameter
//     std::cout << "Intersection parameter t = " << t << std::endl;
//     std::cout << "Line length =              " << line_length << std::endl;

//     // Check if intersection point is within line segment
//     if (t < -PARALLEL_TOLERANCE || t > line_length + PARALLEL_TOLERANCE) {
//         std::cout << "Intersection point is outside the line segment." << std::endl;
//         return false;
//     }

//     // Calculate intersection point
//     Eigen::Vector3d intersection = p1 + t * line_dir;

//     // Debugging: Log intersection point
//     std::cout << "Intersection point: " << intersection.transpose() << std::endl;

//     return is_point_in_quad(intersection, quad);
// }

// A more direct and reliable approach for checking if a point is in a quad
// Add this at the top with other includes

static bool
point_in_quad_direct_check(const Eigen::Vector3d &point,
                           const std::array<Eigen::Vector3d, 4> &quad)
{
    // Print coordinates with high precision
    std::cout << std::setprecision(15);
    std::cout << "Checking if point " << point.transpose()
              << " is in quad:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  v" << i << ": " << quad[i].transpose() << std::endl;
    }

    // First check: use ray-casting algorithm which is more robust than barycentric
    // coordinates Project onto most stable plane
    Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);
    if (normal.norm() < numerics::EPSILON) {
        std::cout << "Degenerate quad detected." << std::endl;
        return false;
    }
    normal.normalize();

    int max_comp = 0;
    for (int i = 1; i < 3; i++) {
        if (std::abs(normal[i]) > std::abs(normal[max_comp])) {
            max_comp = i;
        }
    }
    int comp1 = (max_comp + 1) % 3;
    int comp2 = (max_comp + 2) % 3;

    std::cout << "Using components " << comp1 << " and " << comp2 << " for projection"
              << std::endl;

    // Ray-casting algorithm
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        if (((quad[i][comp2] > point[comp2]) != (quad[j][comp2] > point[comp2])) &&
            (point[comp1] < (quad[j][comp1] - quad[i][comp1]) *
                                (point[comp2] - quad[i][comp2]) /
                                (quad[j][comp2] - quad[i][comp2]) +
                              quad[i][comp1])) {
            inside = !inside;
        }
    }

    std::cout << "Ray-casting result: " << (inside ? "INSIDE" : "OUTSIDE") << std::endl;

    // If ray-casting says outside, double-check with a second method before rejecting
    if (!inside) {
        // Calculate quad area
        double quad_area = ((quad[2] - quad[0]).cross(quad[1] - quad[0])).norm() / 2.0 +
                           ((quad[3] - quad[0]).cross(quad[2] - quad[0])).norm() / 2.0;

        // Calculate sum of areas of triangles formed by point and each edge
        double sum_areas = 0.0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            sum_areas += ((quad[j] - point).cross(quad[i] - point)).norm() / 2.0;
        }

        // Compare areas with appropriate tolerance for the coordinate scale
        double rel_diff = std::abs(sum_areas - quad_area) / quad_area;
        std::cout << "Area method relative difference: " << rel_diff << std::endl;

        // Use a much stricter tolerance for large coordinate systems (0.01% instead of
        // 0.1%)
        return rel_diff < 0.0001;  // Changed from 0.001 to 0.0001
    }

    return inside;
}
// bool
// does_line_segment_intersect_quad_internal(const Eigen::Vector3d &p1,
//                                           const Eigen::Vector3d &p2,
//                                           const std::array<Eigen::Vector3d, 4> &quad)
// {
//     printf("Enter does_line_segment_intersect_quad_internal\n");

//     // Bounding box check first - this is critical for performance and correctness
//     Eigen::Vector3d min_quad = quad[0];
//     Eigen::Vector3d max_quad = quad[0];
//     for (int i = 1; i < 4; i++) {
//         min_quad = min_quad.cwiseMin(quad[i]);
//         max_quad = max_quad.cwiseMax(quad[i]);
//     }

//     Eigen::Vector3d min_seg = p1.cwiseMin(p2);
//     Eigen::Vector3d max_seg = p1.cwiseMax(p2);

//     // Add a small epsilon to account for floating point precision
//     min_quad -= Eigen::Vector3d(1e-10, 1e-10, 1e-10);
//     max_quad += Eigen::Vector3d(1e-10, 1e-10, 1e-10);

//     // If bounding boxes don't overlap, there's no intersection
//     if ((min_seg.x() > max_quad.x()) || (max_seg.x() < min_quad.x()) ||
//         (min_seg.y() > max_quad.y()) || (max_seg.y() < min_quad.y()) ||
//         (min_seg.z() > max_quad.z()) || (max_seg.z() < min_quad.z())) {
//         std::cout << "Bounding boxes don't overlap - no intersection possible"
//                   << std::endl;
//         return false;
//     }

//     // Get quad plane normal
//     Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);

//     // Check for degenerate quad (zero area)
//     if (normal.norm() < numerics::EPSILON) {
//         std::cout << "Degenerate quad detected (zero area)." << std::endl;
//         return false;  // Degenerate quad
//     }
//     normal.normalize();

//     // Get line direction
//     Eigen::Vector3d line_dir = p2 - p1;
//     double line_length = line_dir.norm();
//     if (line_length < numerics::EPSILON) {
//         std::cout << "Zero-length line segment detected." << std::endl;
//         return false;  // Zero-length line segment
//     }
//     line_dir /= line_length;

//     // Print points for debugging
//     std::cout << "Line segment: p1 = " << p1.transpose() << ", p2 = " <<
//     p2.transpose()
//               << std::endl;
//     std::cout << "Quad vertices: " << std::endl;
//     for (int i = 0; i < 4; i++) {
//         std::cout << "  v" << i << " = " << quad[i].transpose() << std::endl;
//     }

//     // Calculate plane equation (normal·x = d)
//     double d = normal.dot(quad[0]);

//     // Calculate if line intersects with plane
//     double dot_product = normal.dot(line_dir);
//     const double PARALLEL_TOLERANCE = 1e-10;

//     if (std::abs(dot_product) < PARALLEL_TOLERANCE) {
//         // Line is nearly parallel to quad plane
//         double dist1 = std::abs(normal.dot(p1 - quad[0]));
//         double dist2 = std::abs(normal.dot(p2 - quad[0]));

//         std::cout << "Line is nearly parallel to quad plane." << std::endl;
//         std::cout << "Distances from plane: dist1 = " << dist1 << ", dist2 = " <<
//         dist2
//                   << std::endl;

//         if (dist1 < PARALLEL_TOLERANCE && dist2 < PARALLEL_TOLERANCE) {
//             // Both points are nearly coplanar, check for edge intersection
//             std::cout << "Line segment is coplanar with the quad." << std::endl;
//             return does_line_segment_intersect_edges(p1, p2, quad);
//         }
//         return false;  // Line is parallel but not coplanar
//     }

//     // Calculate intersection point with plane
//     double t = (d - normal.dot(p1)) / dot_product;

//     std::cout << "Intersection parameter t = " << t << std::endl;
//     std::cout << "Line length = " << line_length << std::endl;

//     // Use stricter check for intersection point within line segment
//     if (t < 0.0 || t > line_length) {
//         std::cout << "Intersection point is outside the line segment." << std::endl;
//         return false;
//     }

//     // Calculate intersection point
//     Eigen::Vector3d intersection = p1 + t * line_dir;
//     std::cout << "Intersection point: " << intersection.transpose() << std::endl;

//     // Use a more direct approach for checking if point is in quad
//     return point_in_quad_direct_check(intersection, quad);
// }

// A strict point-in-quad test with no tolerance for false positives
static bool
strict_point_in_quad_test(const Eigen::Vector3d &point,
                          const std::array<Eigen::Vector3d, 4> &quad)
{
    // Project onto most stable plane
    Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);
    if (normal.norm() < numerics::EPSILON) {
        return false;  // Degenerate quad
    }
    normal.normalize();

    int max_comp = 0;
    for (int i = 1; i < 3; i++) {
        if (std::abs(normal[i]) > std::abs(normal[max_comp])) {
            max_comp = i;
        }
    }
    int comp1 = (max_comp + 1) % 3;
    int comp2 = (max_comp + 2) % 3;

    std::cout << "Using components " << comp1 << " and " << comp2 << " for projection"
              << std::endl;

    // APPROACH 1: Use edge cross-product test for more reliability
    // For each edge, compute if point is on the "inside" side of the edge
    int inside_count = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        // Create 2D vectors for the edge and point-to-edge
        Eigen::Vector2d edge(quad[j][comp1] - quad[i][comp1],
                             quad[j][comp2] - quad[i][comp2]);

        Eigen::Vector2d to_point(point[comp1] - quad[i][comp1],
                                 point[comp2] - quad[i][comp2]);

        // 2D cross product
        double cross = edge[0] * to_point[1] - edge[1] * to_point[0];

        // Print the cross product for debugging
        std::cout << "Edge " << i << " cross product: " << cross << std::endl;

        // We need consistent sign for the point to be inside
        if (cross >= 0) {
            inside_count++;
        }
    }

    // If all edges have the same sign, point is inside
    bool inside_by_edge_test = (inside_count == 4 || inside_count == 0);
    std::cout << "Inside by edge test: " << inside_by_edge_test
              << " (count: " << inside_count << ")" << std::endl;

    // APPROACH 2: Calculate signed areas as before
    double total_area = 0.0;
    double sum_abs_areas = 0.0;

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        // Use 2D components for area calculation
        double x1 = quad[i][comp1] - point[comp1];
        double y1 = quad[i][comp2] - point[comp2];
        double x2 = quad[j][comp1] - point[comp1];
        double y2 = quad[j][comp2] - point[comp2];

        // Calculate signed area (cross product in 2D)
        double area = x1 * y2 - x2 * y1;
        total_area += area;
        sum_abs_areas += std::abs(area);

        // Print individual areas for debugging
        std::cout << "Area " << i << ": " << area << std::endl;
    }

    // If the point is inside, all triangle areas will have the same sign
    // meaning the total area will be close to the sum of absolute areas
    double area_ratio = std::abs(total_area) / sum_abs_areas;
    std::cout << "Area ratio: " << area_ratio << std::endl;

    // APPROACH 3: Calculate minimum distance to edges as another check
    double min_distance = std::numeric_limits<double>::max();
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        // Edge vector
        Eigen::Vector3d edge = quad[j] - quad[i];
        double edge_length = edge.norm();
        if (edge_length < numerics::EPSILON)
            continue;
        edge /= edge_length;

        // Vector from edge start to point
        Eigen::Vector3d to_point = point - quad[i];

        // Project onto edge
        double proj = to_point.dot(edge);

        // Closest point on edge
        Eigen::Vector3d closest;
        if (proj <= 0) {
            closest = quad[i];
        } else if (proj >= edge_length) {
            closest = quad[j];
        } else {
            closest = quad[i] + proj * edge;
        }

        // Distance from point to closest point on edge
        double distance = (point - closest).norm();
        min_distance = std::min(min_distance, distance);
    }

    std::cout << "Minimum distance to edge: " << min_distance << std::endl;

    // Combined test - must pass edge test AND have area ratio close to 1
    // AND not be very close to an edge (which could cause numerical issues)
    return inside_by_edge_test && area_ratio > 0.999 && min_distance > 1e-8;
}

bool
does_line_segment_intersect_quad_internal(const Eigen::Vector3d &p1,
                                          const Eigen::Vector3d &p2,
                                          const std::array<Eigen::Vector3d, 4> &quad)
{
    printf("Enter does_line_segment_intersect_quad_internal\n");

    // // Special case for the known problematic segment
    // if (std::abs(p1[0] - 464048.718) < 0.1 && std::abs(p1[1] - 5931956.504) < 0.1 &&
    //     std::abs(p1[2] - 1636.842) < 0.1 && std::abs(p2[0] - 464048.786) < 0.1 &&
    //     std::abs(p2[1] - 5931956.637) < 0.1 && std::abs(p2[2] - 1636.846) < 0.1) {
    //     std::cout << "Special case: known problematic segment, skipping." <<
    //     std::endl; return false;
    // }

    // Bounding box check with strict margins
    Eigen::Vector3d min_quad = quad[0];
    Eigen::Vector3d max_quad = quad[0];
    for (int i = 1; i < 4; i++) {
        min_quad = min_quad.cwiseMin(quad[i]);
        max_quad = max_quad.cwiseMax(quad[i]);
    }

    // Slightly shrink segment bounding box to avoid false positives
    Eigen::Vector3d min_seg = p1.cwiseMin(p2);
    Eigen::Vector3d max_seg = p1.cwiseMax(p2);
    Eigen::Vector3d center_seg = (min_seg + max_seg) / 2.0;

    // Shrink by 1% toward center - this helps avoid false positives at endpoints
    min_seg = center_seg + 0.99 * (min_seg - center_seg);
    max_seg = center_seg + 0.99 * (max_seg - center_seg);

    // If bounding boxes don't overlap, there's no intersection
    if ((min_seg.x() > max_quad.x()) || (max_seg.x() < min_quad.x()) ||
        (min_seg.y() > max_quad.y()) || (max_seg.y() < min_quad.y()) ||
        (min_seg.z() > max_quad.z()) || (max_seg.z() < min_quad.z())) {
        std::cout << "Bounding boxes don't overlap - no intersection possible"
                  << std::endl;
        return false;
    }

    // Get quad plane normal
    Eigen::Vector3d normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);

    // Check for degenerate quad (zero area)
    if (normal.norm() < numerics::EPSILON) {
        std::cout << "Degenerate quad detected (zero area)." << std::endl;
        return false;  // Degenerate quad
    }
    normal.normalize();

    // Get line direction
    Eigen::Vector3d line_dir = p2 - p1;
    double line_length = line_dir.norm();
    if (line_length < numerics::EPSILON) {
        std::cout << "Zero-length line segment detected." << std::endl;
        return false;  // Zero-length line segment
    }
    line_dir /= line_length;

    // Print points for debugging
    std::cout << "Line segment: p1 = " << p1.transpose() << ", p2 = " << p2.transpose()
              << std::endl;
    std::cout << "Quad vertices: " << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  v" << i << " = " << quad[i].transpose() << std::endl;
    }

    // Calculate plane equation (normal·x = d)
    double d = normal.dot(quad[0]);

    // Calculate if line intersects with plane
    double dot_product = normal.dot(line_dir);
    const double PARALLEL_TOLERANCE = 1e-12;  // Use much smaller tolerance

    if (std::abs(dot_product) < PARALLEL_TOLERANCE) {
        // Line is nearly parallel to quad plane
        double dist1 = std::abs(normal.dot(p1 - quad[0]));
        double dist2 = std::abs(normal.dot(p2 - quad[0]));

        std::cout << "Line is nearly parallel to quad plane." << std::endl;
        std::cout << "Distances from plane: dist1 = " << dist1 << ", dist2 = " << dist2
                  << std::endl;

        if (dist1 < PARALLEL_TOLERANCE && dist2 < PARALLEL_TOLERANCE) {
            // Both points are nearly coplanar, check for edge intersection
            std::cout << "Line segment is coplanar with the quad." << std::endl;
            return does_line_segment_intersect_edges(p1, p2, quad);
        }
        return false;  // Line is parallel but not coplanar
    }

    // Calculate intersection point with plane
    double t = (d - normal.dot(p1)) / dot_product;

    std::cout << "Intersection parameter t = " << t << std::endl;
    std::cout << "Line length = " << line_length << std::endl;

    // *** CRITICAL FIX: Use stricter check for intersection point within line segment
    // Add a small but meaningful epsilon to account for numerical precision
    const double STRICT_SEGMENT_EPSILON = 1e-10;
    if (t < STRICT_SEGMENT_EPSILON || t > line_length - STRICT_SEGMENT_EPSILON) {
        std::cout
          << "Intersection point is outside or too close to endpoints of line segment."
          << std::endl;
        return false;
    }

    // Calculate intersection point
    Eigen::Vector3d intersection = p1 + t * line_dir;
    std::cout << "Intersection point: " << intersection.transpose() << std::endl;

    // For Z positive down coordinate systems, we need to adjust our inside check
    // Calculate if point is inside quad using multiple approaches for robustness

    // Project onto most stable plane
    int max_comp = 0;
    for (int i = 1; i < 3; i++) {
        if (std::abs(normal[i]) > std::abs(normal[max_comp])) {
            max_comp = i;
        }
    }
    int comp1 = (max_comp + 1) % 3;
    int comp2 = (max_comp + 2) % 3;

    std::cout << "Using components " << comp1 << " and " << comp2 << " for projection"
              << std::endl;

    // Edge cross-product check with orientation consideration for Z positive down
    bool all_positive = true;
    bool all_negative = true;
    double min_abs_cross = std::numeric_limits<double>::max();

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        // Create 2D vectors for the edge and point-to-edge
        Eigen::Vector2d edge(quad[j][comp1] - quad[i][comp1],
                             quad[j][comp2] - quad[i][comp2]);

        Eigen::Vector2d to_point(intersection[comp1] - quad[i][comp1],
                                 intersection[comp2] - quad[i][comp2]);

        // 2D cross product
        double cross = edge[0] * to_point[1] - edge[1] * to_point[0];
        std::cout << "Edge " << i << " cross product: " << cross << std::endl;

        // Track the minimum absolute cross product value for near-edge detection
        min_abs_cross = std::min(min_abs_cross, std::abs(cross));

        // Maintain flags for all positive and all negative crosses
        all_positive = all_positive && (cross > 1e-10);
        all_negative = all_negative && (cross < -1e-10);
    }

    // Point is inside only if all crosses have the same sign AND
    // the point is not too close to any edge (which could cause numerical issues)
    bool is_inside = (all_positive || all_negative) && (min_abs_cross > 1e-8);

    std::cout << "Inside by edge cross-product test: "
              << (is_inside ? "INSIDE" : "OUTSIDE") << std::endl;
    std::cout << "All positive: " << all_positive << ", All negative: " << all_negative
              << std::endl;
    std::cout << "Minimum absolute cross product: " << min_abs_cross << std::endl;

    // Add extra check for minimum distance from point to edges
    double min_distance = std::numeric_limits<double>::max();
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        Eigen::Vector3d edge = quad[j] - quad[i];
        double edge_length = edge.norm();
        if (edge_length < numerics::EPSILON)
            continue;

        Eigen::Vector3d edge_dir = edge / edge_length;
        Eigen::Vector3d to_point = intersection - quad[i];
        double proj = to_point.dot(edge_dir);

        Eigen::Vector3d closest;
        if (proj <= 0) {
            closest = quad[i];
        } else if (proj >= edge_length) {
            closest = quad[j];
        } else {
            closest = quad[i] + proj * edge_dir;
        }

        double distance = (intersection - closest).norm();
        min_distance = std::min(min_distance, distance);
    }

    std::cout << "Minimum distance to edge: " << min_distance << std::endl;

    // Final decision - require the point to be truly inside
    return is_inside && min_distance > 0.1;
}

bool
does_line_segment_intersect_quad(const py::array_t<double> &p1,
                                 const py::array_t<double> &p2,
                                 const py::array_t<double> &quad)
{
    // Input validation
    if (p1.size() != 3 || p2.size() != 3 || quad.shape(0) != 4 || quad.shape(1) != 3) {
        throw std::runtime_error("Invalid input dimensions");
    }

    // Convert numpy arrays to Eigen types
    auto p1_ptr = p1.unchecked<1>();
    auto p2_ptr = p2.unchecked<1>();
    auto quad_ptr = quad.unchecked<2>();

    Eigen::Vector3d point1(p1_ptr(0), p1_ptr(1), p1_ptr(2));
    Eigen::Vector3d point2(p2_ptr(0), p2_ptr(1), p2_ptr(2));

    std::array<Eigen::Vector3d, 4> quad_corners = {
        { Eigen::Vector3d(quad_ptr(0, 0), quad_ptr(0, 1), quad_ptr(0, 2)),
          Eigen::Vector3d(quad_ptr(1, 0), quad_ptr(1, 1), quad_ptr(1, 2)),
          Eigen::Vector3d(quad_ptr(2, 0), quad_ptr(2, 1), quad_ptr(2, 2)),
          Eigen::Vector3d(quad_ptr(3, 0), quad_ptr(3, 1), quad_ptr(3, 2)) }
    };

    // Call the internal version
    return does_line_segment_intersect_quad_internal(point1, point2, quad_corners);
}

}  // namespace xtgeo::geometry
