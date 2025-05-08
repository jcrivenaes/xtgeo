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

bool
is_hexahedron_severely_distorted(const xtgeo::geometry::HexahedronCorners &corners)
{
    // Thresholds for distortion checks
    constexpr double MIN_VOLUME_THRESHOLD = 1e-6;
    constexpr double PLANARITY_TOLERANCE = 12.0;
    constexpr double ASPECT_RATIO_THRESHOLD = 100.0;
    constexpr double DIHEDRAL_ANGLE_TOLERANCE = 30.0;  // In degrees

    // Helper function to calculate the length of an edge
    auto edge_length = [](const xyz::Point &p1, const xyz::Point &p2) {
        return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2) +
                         std::pow(p2.z - p1.z, 2));
    };

    // Check aspect ratios
    std::array<double, 12> edge_lengths = {
        edge_length(corners.upper_sw, corners.upper_se),
        edge_length(corners.upper_se, corners.upper_ne),
        edge_length(corners.upper_ne, corners.upper_nw),
        edge_length(corners.upper_nw, corners.upper_sw),
        edge_length(corners.lower_sw, corners.lower_se),
        edge_length(corners.lower_se, corners.lower_ne),
        edge_length(corners.lower_ne, corners.lower_nw),
        edge_length(corners.lower_nw, corners.lower_sw),
        edge_length(corners.upper_sw, corners.lower_sw),
        edge_length(corners.upper_se, corners.lower_se),
        edge_length(corners.upper_ne, corners.lower_ne),
        edge_length(corners.upper_nw, corners.lower_nw),
    };

    double max_edge = *std::max_element(edge_lengths.begin(), edge_lengths.end());
    double min_edge = *std::min_element(edge_lengths.begin(), edge_lengths.end());

    if (min_edge <= 0.0) {
        return true;  // Severely distorted due to zero or negative edge length
    }
    if (max_edge / min_edge > ASPECT_RATIO_THRESHOLD) {
        printf("Aspect ratio: %f\n", max_edge / min_edge);
        return true;  // Severely distorted due to aspect ratio
    }

    // Check face planarity
    auto check_face_planarity = [](const std::array<xyz::Point, 4> &face) {
        auto edge1 = xyz::Point(face[1].x - face[0].x, face[1].y - face[0].y,
                                face[1].z - face[0].z);
        auto edge2 = xyz::Point(face[2].x - face[0].x, face[2].y - face[0].y,
                                face[2].z - face[0].z);
        auto normal = xyz::Point(edge1.y * edge2.z - edge1.z * edge2.y,
                                 edge1.z * edge2.x - edge1.x * edge2.z,
                                 edge1.x * edge2.y - edge1.y * edge2.x);
        double magnitude =
          std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (magnitude <= 1e-9) {
            return false;
        }
        normal.x /= magnitude;
        normal.y /= magnitude;
        normal.z /= magnitude;

        for (const auto &point : face) {
            auto vec =
              xyz::Point(point.x - face[0].x, point.y - face[0].y, point.z - face[0].z);
            double dot_product = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;
            if (std::abs(dot_product) > PLANARITY_TOLERANCE) {
                return false;  // Face is not planar
            }
        }
        return true;
    };

    std::array<std::array<xyz::Point, 4>, 6> faces = { {
      { corners.upper_sw, corners.upper_se, corners.upper_ne, corners.upper_nw },
      { corners.lower_sw, corners.lower_se, corners.lower_ne, corners.lower_nw },
      { corners.upper_sw, corners.upper_se, corners.lower_se, corners.lower_sw },
      { corners.upper_se, corners.upper_ne, corners.lower_ne, corners.lower_se },
      { corners.upper_ne, corners.upper_nw, corners.lower_nw, corners.lower_ne },
      { corners.upper_nw, corners.upper_sw, corners.lower_sw, corners.lower_nw },
    } };

    for (const auto &face : faces) {
        if (!check_face_planarity(face)) {
            printf("Face planarity check failed\n");
            return true;  // Severely distorted due to non-planar face
        }
    }

    // Check dihedral angles
    auto calculate_angle = [](const xyz::Point &normal1, const xyz::Point &normal2) {
        double dot =
          normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z;
        double magnitude1 = std::sqrt(normal1.x * normal1.x + normal1.y * normal1.y +
                                      normal1.z * normal1.z);
        double magnitude2 = std::sqrt(normal2.x * normal2.x + normal2.y * normal2.y +
                                      normal2.z * normal2.z);
        return std::acos(dot / (magnitude1 * magnitude2)) * 180.0 /
               M_PI;  // Convert to degrees
    };

    for (size_t i = 0; i < faces.size(); ++i) {
        for (size_t j = i + 1; j < faces.size(); ++j) {
            // Calculate normals for both faces
            auto normal1 = cross_product(subtract(faces[i][1], faces[i][0]),
                                         subtract(faces[i][2], faces[i][0]));
            auto normal2 = cross_product(subtract(faces[j][1], faces[j][0]),
                                         subtract(faces[j][2], faces[j][0]));

            double angle = calculate_angle(normal1, normal2);
            if (std::abs(angle - 90.0) > DIHEDRAL_ANGLE_TOLERANCE) {
                return true;  // Severely distorted due to dihedral angle deviation
            }
        }
    }

    // Check volume
    double volume = hexahedron_volume(corners, 0.001);
    if (volume < MIN_VOLUME_THRESHOLD) {
        return true;  // Severely distorted due to near-zero volume
    }

    return false;  // Hexahedron is not severely distorted
}

/**
 * @brief Check if a hexahedron cell is thin based on the ratio of thickness to area.
 * @param corners The 8 corners of the hexahedron
 * @param threshold The threshold for the thickness-to-area ratio
 * @return bool Returns true if the cell is thin, false otherwise
 */
bool
is_hexahedron_thin(const HexahedronCorners &corners, const double threshold)
{
    // Helper function to calculate the area of a quadrilateral face
    auto calculate_area = [](const xyz::Point &p1, const xyz::Point &p2,
                             const xyz::Point &p3, const xyz::Point &p4) -> double {
        auto cross = [](const xyz::Point &a, const xyz::Point &b) {
            return xyz::Point{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                               a.x * b.y - a.y * b.x };
        };

        auto subtract = [](const xyz::Point &a, const xyz::Point &b) {
            return xyz::Point{ a.x - b.x, a.y - b.y, a.z - b.z };
        };

        // Divide the quadrilateral into two triangles and calculate their areas
        xyz::Point v1 = subtract(p2, p1);
        xyz::Point v2 = subtract(p3, p1);
        xyz::Point v3 = subtract(p4, p1);

        double area1 =
          0.5 * std::sqrt(std::pow(cross(v1, v2).x, 2) + std::pow(cross(v1, v2).y, 2) +
                          std::pow(cross(v1, v2).z, 2));
        double area2 =
          0.5 * std::sqrt(std::pow(cross(v2, v3).x, 2) + std::pow(cross(v2, v3).y, 2) +
                          std::pow(cross(v2, v3).z, 2));

        return area1 + area2;
    };

    // Calculate the area of the top and bottom faces
    double top_area = calculate_area(corners.upper_sw, corners.upper_se,
                                     corners.upper_ne, corners.upper_nw);
    double bottom_area = calculate_area(corners.lower_sw, corners.lower_se,
                                        corners.lower_ne, corners.lower_nw);

    // Use the average of the top and bottom areas
    double average_area = (top_area + bottom_area) / 2.0;

    // Calculate the thickness (difference in Z-coordinates between upper and lower
    // faces)
    double thickness = 0.25 * (std::abs(corners.upper_sw.z - corners.lower_sw.z) +
                               std::abs(corners.upper_se.z - corners.lower_se.z) +
                               std::abs(corners.upper_ne.z - corners.lower_ne.z) +
                               std::abs(corners.upper_nw.z - corners.lower_nw.z));

    if (thickness <= numerics::TOLERANCE) {
        return true;  // Cell is considered thin if thickness or area is too small
    }
    if (average_area <= numerics::TOLERANCE) {
        return false;  // Cell is probably not thin if area is too small
    }

    // Check if the thickness-to-area ratio is below the threshold
    return (thickness / average_area) < threshold;
}

/**
 * @brief Detect if a hexahedron is concave when viewed from above (projected onto the
 * XY plane). A cell is concave if one corner is within the triangle formed by the other
 * corners at the top and/or base.
 *
 * @param corners The 8 corners of the hexahedron
 * @return bool Returns true if the cell is concave, false if it is convex
 */
bool
is_hexahedron_concave_projected(const HexahedronCorners &corners)
{
    // Extract the X and Y coordinates of the corners for top and base
    std::array<std::array<double, 2>, 4> xp, yp;
    xp[0] = { corners.upper_sw.x, corners.lower_sw.x };
    xp[1] = { corners.upper_se.x, corners.lower_se.x };
    xp[2] = { corners.upper_ne.x, corners.lower_ne.x };
    xp[3] = { corners.upper_nw.x, corners.lower_nw.x };

    yp[0] = { corners.upper_sw.y, corners.lower_sw.y };
    yp[1] = { corners.upper_se.y, corners.lower_se.y };
    yp[2] = { corners.upper_ne.y, corners.lower_ne.y };
    yp[3] = { corners.upper_nw.y, corners.lower_nw.y };

    // Check for concavity at both the top and base
    for (int ntop = 0; ntop < 2; ++ntop) {
        for (int nchk = 0; nchk < 4; ++nchk) {
            // Form a triangle with the other three corners
            std::vector<xyz::Point> triangle_points;
            for (int n = 0; n < 4; ++n) {
                if (n != nchk) {
                    triangle_points.push_back({ xp[n][ntop], yp[n][ntop], 0.0 });
                }
            }

            // Construct the Polygon directly with its points
            xyz::Polygon triangle(triangle_points);

            // Check if the current corner is inside the triangle
            if (is_xy_point_in_polygon(xp[nchk][ntop], yp[nchk][ntop], triangle)) {
                return true;  // The cell is concave
            }
        }
    }

    return false;  // The cell is convex
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
