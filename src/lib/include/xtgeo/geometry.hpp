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

enum class PointInHexahedronMethod
{
    RayCasting,
    Tetrahedrons,
    UsingPlanes,
    Legacy,
    Isoparametric,
    Optimized  // combing approriate methods
};

// =====================================================================================
// TETRAHEDRONS
// =====================================================================================

double
signed_tetrahedron_volume(const xyz::Point &a,
                          const xyz::Point &b,
                          const xyz::Point &c,
                          const xyz::Point &d);

int
is_point_in_tetrahedron(const xyz::Point &point,
                        const xyz::Point &v0,
                        const xyz::Point &v1,
                        const xyz::Point &v2,
                        const xyz::Point &v3);

bool
is_point_in_tetrahedron_legacy(const xyz::Point &point,
                               const xyz::Point &v0,
                               const xyz::Point &v1,
                               const xyz::Point &v2,
                               const xyz::Point &v3);
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
hexahedron_volume_legacy(const HexahedronCorners &corners, const int precision);

// overload for CellCorners
double
hexahedron_volume_legacy(const grid3d::CellCorners &corners, const int precision);

double
hexahedron_volume(const HexahedronCorners &corners);

// overload for CellCorners
double
hexahedron_volume(const grid3d::CellCorners &corners);

bool
is_point_in_hexahedron_raycasting(const xyz::Point &point,
                                  const HexahedronCorners &corners);
bool
is_point_in_hexahedron_usingplanes(const xyz::Point &point,
                                   const HexahedronCorners &corners);
int
is_point_in_hexahedron_tetrahedrons_legacy(const xyz::Point &point,
                                           const HexahedronCorners &corners);

bool
is_point_in_hexahedron_tetrahedrons_by_scheme(const xyz::Point &point,
                                              const HexahedronCorners &corners);

int
is_point_in_hexahedron_isoparametric(const xyz::Point &point,
                                     const HexahedronCorners &corners);

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

bool
is_point_in_hexahedron_bounding_box(const xyz::Point &point,
                                    const HexahedronCorners &hexahedron_corners);
bool
is_point_in_hexahedron_bounding_box_minmax_pt(const xyz::Point &point,
                                              const xyz::Point &min_pt,
                                              const xyz::Point &max_pt);

// =====================================================================================
// PYTHON BINDINGS
// =====================================================================================
inline void
init(py::module &m)
{
    auto m_geometry = m.def_submodule("geometry", "Internal geometric functions");

    py::enum_<PointInHexahedronMethod>(m_geometry, "PointInHexahedronMethod")
      .value("RayCasting", PointInHexahedronMethod::RayCasting)
      .value("Tetrahedrons", PointInHexahedronMethod::Tetrahedrons)
      .value("UsingPlanes", PointInHexahedronMethod::UsingPlanes)
      .value("Legacy", PointInHexahedronMethod::Legacy)
      .value("Isoparametric", PointInHexahedronMethod::Isoparametric)
      .value("Optimized", PointInHexahedronMethod::Optimized)
      .export_values();  // Makes the enum values accessible as attributes of the enum

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
      [](const grid3d::CellCorners &corners) {  // overload for CellCorners
          return hexahedron_volume(corners);
      },
      "Estimate the volume of a hexahedron i.e. a cornerpoint cell using "
      "CornerPoints.");
    m_geometry.def(
      "hexahedron_volume_legacy",
      [](const grid3d::CellCorners &corners, const int precision) {
          return hexahedron_volume_legacy(corners, precision);
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
