#ifndef XTGEO_GEOMETRY_HPP_
#define XTGEO_GEOMETRY_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <vector>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::geometry {

// =====================================================================================
// TETRAHEDRONS
// =====================================================================================

constexpr int TETRAHEDRON_VERTICES[4][6][4] = {
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 5
      { 3, 7, 4, 5 },
      { 0, 4, 7, 5 },
      { 0, 3, 1, 5 },
      // upper left common vertex 6
      { 0, 4, 7, 6 },
      { 3, 7, 4, 6 },
      { 0, 3, 2, 6 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    {
      // upper right common vertex 7
      { 1, 5, 6, 7 },
      { 2, 6, 5, 7 },
      { 1, 2, 3, 7 },
      // lower left common vertex 4
      { 1, 5, 6, 4 },
      { 2, 6, 5, 4 },
      { 1, 2, 0, 4 },
    },

    // Another combination...
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 1
      { 3, 7, 0, 1 },
      { 0, 4, 3, 1 },
      { 4, 7, 5, 1 },
      // upper left common vertex 2
      { 0, 4, 3, 2 },
      { 3, 7, 0, 2 },
      { 4, 7, 6, 2 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    { // upper right common vertex 3
      { 1, 5, 2, 3 },
      { 2, 6, 1, 3 },
      { 5, 6, 7, 3 },
      // lower left common vertex 0
      { 1, 5, 2, 0 },
      { 2, 6, 1, 0 },
      { 5, 6, 4, 0 } }
};

// schemes used for tetrahedron decomposition when the cell is re-arranged to
// counter clock order.
constexpr int TETRAHEDRON_SCHEMES[4][6][4] = {
    // Scheme 0: Diagonal 0-2 at top and 4-6 at base
    {
      { 0, 1, 2, 6 },  // Top face: diagonal 0-2
      { 0, 2, 3, 6 },  // Top face: diagonal 0-2
      { 4, 5, 6, 0 },  // Base face: diagonal 4-6
      { 4, 6, 7, 0 },  // Base face: diagonal 4-6
      { 0, 1, 5, 6 },  // Front face
      { 3, 7, 6, 2 }   // Back face
    },
    // Scheme 1: Diagonal 0-2 at top and 5-7 at base
    {
      { 0, 1, 2, 6 },  // Top face: diagonal 0-2
      { 0, 2, 3, 6 },  // Top face: diagonal 0-2
      { 4, 5, 7, 3 },  // Base face: diagonal 5-7
      { 5, 6, 7, 3 },  // Base face: diagonal 5-7
      { 0, 1, 5, 6 },  // Front face
      { 3, 7, 6, 2 }   // Back face
    },
    // Scheme 2: Diagonal 1-3 at top and 4-6 at base
    {
      { 0, 1, 3, 7 },  // Top face: diagonal 1-3
      { 1, 2, 3, 7 },  // Top face: diagonal 1-3
      { 4, 5, 6, 0 },  // Base face: diagonal 4-6
      { 4, 6, 7, 0 },  // Base face: diagonal 4-6
      { 0, 1, 5, 7 },  // Front face
      { 2, 3, 7, 6 }   // Back face
    },
    // Scheme 3: Diagonal 1-3 at top and 5-7 at base
    {
      { 0, 1, 3, 7 },  // Top face: diagonal 1-3
      { 1, 2, 3, 7 },  // Top face: diagonal 1-3
      { 4, 5, 7, 3 },  // Base face: diagonal 5-7
      { 5, 6, 7, 3 },  // Base face: diagonal 5-7
      { 0, 1, 5, 7 },  // Front face
      { 2, 3, 7, 6 }   // Back face
    }
};
// Centroid-based decomposition (8 tetrahedra)
// Each tetrahedron connects a triangular face to the centroid
// The -1 in the fourth position indicates to use the centroid

constexpr int CENTROID_TETRAHEDRON_SCHEME[2][8][4] = {
    {
      { 0, 1, 3, -1 },  // top face: upper_sw, upper_se, upper_nw, centroid
      { 1, 2, 3, -1 },  // top face: upper_se, upper_ne, upper_nw, centroid
      { 4, 5, 7, -1 },  // bottom face: lower_sw, lower_se, lower_nw, centroid
      { 5, 6, 7, -1 },  // bottom face: lower_se, lower_ne, lower_nw, centroid
      { 0, 1, 5, -1 },  // front face: upper_sw, upper_se, lower_se, centroid
      { 0, 4, 5, -1 },  // front face: upper_sw, lower_sw, lower_se, centroid
      { 1, 2, 6, -1 },  // right face: upper_se, upper_ne, lower_ne, centroid
      { 1, 5, 6, -1 }   // right face: upper_se, lower_se, lower_ne, centroid
    },
    {
      { 0, 1, 2, -1 },  // top face: upper_sw, upper_se, upper_ne, centroid
      { 0, 2, 3, -1 },  // top face: upper_sw, upper_ne, upper_nw, centroid
      { 4, 5, 6, -1 },  // bottom face: lower_sw, lower_se, lower_ne, centroid
      { 4, 6, 7, -1 },  // bottom face: lower_sw, lower_ne, lower_nw, centroid
      { 0, 1, 5, -1 },  // front face: upper_sw, upper_se, lower_se, centroid
      { 0, 4, 5, -1 },  // front face: upper_sw, lower_sw, lower_se, centroid
      { 2, 3, 7, -1 },  // back face: upper_ne, upper_nw, lower_nw centroid
      { 2, 6, 7, -1 }   // back face: upper_ne lower_ne lower_nw centroid
    }
};
bool
is_point_in_tetrahedron(const xyz::Point &point,
                        const xyz::Point &v0,
                        const xyz::Point &v1,
                        const xyz::Point &v2,
                        const xyz::Point &v3,
                        const std::string &method = "barycentric");

// =====================================================================================
// POLYGONS (TRIANGLES, QUADRILATERALS, ...)
// =====================================================================================

inline double
triangle_area(const xyz::Point &p1, const xyz::Point &p2, const xyz::Point &p3)
{
    return 0.5 *
           std::abs(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
}

inline double
quadrilateral_area(const xyz::Point &p1,
                   const xyz::Point &p2,
                   const xyz::Point &p3,
                   const xyz::Point &p4)
{
    // Note points are in clockwise order or counter-clockwise order
    return triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4);
}

bool
is_xy_point_in_polygon(const double x, const double y, const xyz::Polygon &polygon);

bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const xyz::Point &p1,
                             const xyz::Point &p2,
                             const xyz::Point &p3,
                             const xyz::Point &p4,
                             const double tolerance = numerics::TOLERANCE);
double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const xyz::Point &p1,
                         const xyz::Point &p2,
                         const xyz::Point &p3,
                         const xyz::Point &p4,
                         const double tolerance = numerics::TOLERANCE);

double
interpolate_z_4p(const double x,
                 const double y,
                 const xyz::Point &p1,
                 const xyz::Point &p2,
                 const xyz::Point &p3,
                 const xyz::Point &p4,
                 const double tolerance = numerics::TOLERANCE);

std::array<double, 8>
find_rect_corners_from_center(const double x,
                              const double y,
                              const double xinc,
                              const double yinc,
                              const double rot);

// =====================================================================================
// HEXAHEDRON
// =====================================================================================

inline double
hexahedron_dz(const HexahedronCorners &corners)
{
    double dzsum = 0.0;
    dzsum += std::abs(corners.upper_sw.z - corners.lower_sw.z);
    dzsum += std::abs(corners.upper_se.z - corners.lower_se.z);
    dzsum += std::abs(corners.upper_nw.z - corners.lower_nw.z);
    dzsum += std::abs(corners.upper_ne.z - corners.lower_ne.z);
    return dzsum / 4.0;
}

double
hexahedron_volume(const HexahedronCorners &corners, const int precision);
// overload for CellCorners
double
hexahedron_volume(const grid3d::CellCorners &corners, const int precision);

bool
is_point_in_hexahedron(const xyz::Point &point,
                       const HexahedronCorners &corners,
                       const std::string &method);
bool
is_hexahedron_non_convex(const HexahedronCorners &corners);

bool
is_hexahedron_severely_distorted(const xtgeo::geometry::HexahedronCorners &corners);

bool
is_hexahedron_thin(const HexahedronCorners &corners, const double threshold = 0.05);

bool
is_hexahedron_concave_projected(const HexahedronCorners &corners);

std::vector<double>
get_hexahedron_minmax(const HexahedronCorners &corners);

std::tuple<xyz::Point, xyz::Point>
get_hexahedron_bounding_box(const HexahedronCorners &corners);

// =====================================================================================
// PYTHON BINDINGS
// =====================================================================================
inline void
init(py::module &m)
{
    auto m_geometry = m.def_submodule("geometry", "Internal geometric functions");

    py::class_<HexahedronCorners>(m_geometry, "HexahedronCorners")
      // a constructor that takes 8 xyz::Point objects
      .def(py::init<xyz::Point, xyz::Point, xyz::Point, xyz::Point, xyz::Point,
                    xyz::Point, xyz::Point, xyz::Point>())
      // a constructor that takes a one-dimensional array of 24 elements
      // Note that HexahedronCorners differs from CellCorners (slightly)
      .def(py::init<const py::array_t<double> &>())

      .def_readonly("upper_sw", &HexahedronCorners::upper_sw)
      .def_readonly("upper_se", &HexahedronCorners::upper_se)
      .def_readonly("upper_ne", &HexahedronCorners::upper_ne)
      .def_readonly("upper_nw", &HexahedronCorners::upper_nw)
      .def_readonly("lower_sw", &HexahedronCorners::lower_sw)
      .def_readonly("lower_se", &HexahedronCorners::lower_se)
      .def_readonly("lower_ne", &HexahedronCorners::lower_ne)
      .def_readonly("lower_nw", &HexahedronCorners::lower_nw)

      ;

    m_geometry.def(
      "hexahedron_volume",
      [](const grid3d::CellCorners &corners,
         int precision) {  // overload for CellCorners
          return hexahedron_volume(corners, precision);
      },
      "Estimate the volume of a hexahedron i.e. a cornerpoint cell using "
      "CornerPoints.");
    m_geometry.def("is_xy_point_in_polygon", &is_xy_point_in_polygon,
                   "Return True if a XY point is inside a polygon seen from above, "
                   "False otherwise.");
    m_geometry.def("is_xy_point_in_quadrilateral", &is_xy_point_in_quadrilateral,
                   "Return True if a XY point is inside a quadrilateral seen from , "
                   "above. False otherwise.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
    m_geometry.def("interpolate_z_4p_regular", &interpolate_z_4p_regular,
                   "Interpolate Z when having 4 corners in a regular XY space, "
                   "typically a regular surface.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
    m_geometry.def("interpolate_z_4p", &interpolate_z_4p,
                   "Interpolate Z when having 4 corners in a non regular XY space, "
                   "like the top of a 3D grid cell.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
    m_geometry.def("is_point_in_hexahedron", &is_point_in_hexahedron,
                   "Determine if a point XYZ is inside a hexahedron, with method");
    m_geometry.def("is_hexahedron_non_convex", &is_hexahedron_non_convex,
                   "Determine if a hexahedron is non-convex");
    m_geometry.def("is_hexahedron_severely_distorted",
                   &is_hexahedron_severely_distorted,
                   "Determine if a hexahedron is severely distorted");
    m_geometry.def("is_hexahedron_thin", &is_hexahedron_thin,
                   "Determine if a hexahedron is thin", py::arg("corners"),
                   py::arg("threshold") = 0.05);
    m_geometry.def("is_hexahedron_concave_projected", &is_hexahedron_concave_projected,
                   "Determine if a hexahedron is concave projected");
}
}  // namespace xtgeo::geometry

#endif  // XTGEO_GEOMETRY_HPP_
