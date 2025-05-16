#include <pybind11/stl.h>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>

namespace xtgeo::geometry {

// The one is based on CellCorners numbering which is different for HexahedronCorners
// numbering!
// Vertex ordering:
// 0: upper_sw, 1: upper_se, 2: upper_nw, 3: upper_ne
// 4: lower_sw, 5: lower_se, 6: lower_nw, 7: lower_ne

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
// Five-tetrahedron decomposition schemes for hexahedra using HexahedronCorners indexing
// Vertex ordering for HexahedronCorners:
// 0: upper_sw, 1: upper_se, 2: upper_ne, 3: upper_nw
// 4: lower_sw, 5: lower_se, 6: lower_ne, 7: lower_nw
constexpr int HEXAHEDRON_TETRAHEDRON_VERTICES[4][6][4] = {
    // cell top/base hinge is splittet 0 - 2 / 4 - 6
    {
      // lower right common vertex 5
      { 2, 6, 5, 4 },
      { 0, 4, 6, 5 },
      { 0, 2, 1, 5 },
      // upper left common vertex 7
      { 0, 4, 7, 6 },
      { 2, 6, 4, 7 },
      { 0, 2, 7, 3 },
    },

    // cell top/base hinge is splittet 1-3 / 5-7
    {
      // upper right common vertex 6
      { 1, 5, 7, 6 },
      { 3, 7, 6, 5 },
      { 1, 3, 2, 6 },
      // lower left common vertex 4
      { 1, 5, 4, 7 },
      { 3, 7, 5, 4 },
      { 1, 3, 4, 0 },
    },

    // Another combination...
    // cell top/base hinge is splittet 0 - 2 / 4 - 6
    {
      // lower right common vertex 1
      { 2, 6, 1, 0 },
      { 0, 4, 2, 1 },
      { 4, 6, 1, 5 },
      // upper left common vertex 3
      { 0, 4, 3, 2 },
      { 2, 6, 0, 3 },
      { 4, 6, 7, 3 },
    },

    // cell top/base hinge is splittet 1-3 / 5-7
    { // upper right common vertex 2
      { 1, 5, 3, 2 },
      { 3, 7, 2, 1 },
      { 5, 7, 2, 6 },
      // lower left common vertex 0
      { 1, 5, 0, 3 },
      { 3, 7, 1, 0 },
      { 5, 7, 4, 0 } }
};

using grid3d::CellCorners;

/**
 * @brief Legacy computation; from C/SWIG codebase in xtgeo
 *
 * Estimate the volume of a hexahedron i.e. a cornerpoint cell. This is a
 * nonunique entity, but it is approximated by computing two different ways of
 * top/base splitting and average those.
 *
 *    6          7
 *     2        3
 *      |------|   Example of split along
 *      |    / |   0 - 3 at top and 4 - 7
 *      |   /  |   at base. Alternative is
 *      | /    |   1 - 2 at top and 5 - 6
 *      |------|   at base
 *     0        1
 *    4          5
 *
 * Note however... this fails if the cell is concave and convace cells need
 * special attention
 *
 * @param corners A CellCorners instance
 * @param precision The precision to calculate the volume to (int, counting number of
 * schemas)
 * @return The volume of the hexahadron
 */
double
hexahedron_volume_legacy(const HexahedronCorners &corners, const int precision)
{
    // Avoid cells that collapsed in some way
    if (hexahedron_dz(corners) < numerics::EPSILON) {
        return 0.0;
    }

    std::array<std::array<double, 3>, 8> crn = {
        { { corners.upper_sw.x, corners.upper_sw.y, corners.upper_sw.z },
          { corners.upper_se.x, corners.upper_se.y, corners.upper_se.z },
          { corners.upper_ne.x, corners.upper_ne.y, corners.upper_ne.z },
          { corners.upper_nw.x, corners.upper_nw.y, corners.upper_nw.z },
          { corners.lower_sw.x, corners.lower_sw.y, corners.lower_sw.z },
          { corners.lower_se.x, corners.lower_se.y, corners.lower_se.z },
          { corners.lower_ne.x, corners.lower_ne.y, corners.lower_ne.z },
          { corners.lower_nw.x, corners.lower_nw.y, corners.lower_nw.z } }
    };

    double vol = 0.0;
    double tetrahedron[12]{};
    for (auto i = 1; i <= precision; i++) {
        double tetra_vol = 0.0;
        for (auto j = 0; j < 6; j++) {
            auto idx = 0;
            for (auto k = 0; k < 4; k++) {
                auto tetra_idx = HEXAHEDRON_TETRAHEDRON_VERTICES[i - 1][j][k];
                tetrahedron[idx++] = crn[tetra_idx][0];
                tetrahedron[idx++] = crn[tetra_idx][1];
                tetrahedron[idx++] = crn[tetra_idx][2];
            }
            tetra_vol += x_tetrahedron_volume(tetrahedron, 12);
        }
        vol = (vol * (i - 1) + tetra_vol) / i;
    }
    return vol;
}

// Helper function to get a corner by index
static const xyz::Point &
get_corner_by_index(const HexahedronCorners &corners, int index)
{
    switch (index) {
        case 0:
            return corners.upper_sw;
        case 1:
            return corners.upper_se;
        case 2:
            return corners.upper_ne;
        case 3:
            return corners.upper_nw;
        case 4:
            return corners.lower_sw;
        case 5:
            return corners.lower_se;
        case 6:
            return corners.lower_ne;
        case 7:
            return corners.lower_nw;
        default:
            throw std::out_of_range("Invalid corner index");
    }
}
/**
 * @brief Calculate the volume of a hexahedron. Robust for both convex and concave
 * cells.
 *
 * @param corners The corners of the hexahedron.
 * @return double The volume of the hexahedron.
 */

static double
hexahedron_volume_per_scheme(const HexahedronCorners &corners, int scheme)
{
    // Standard node ordering:
    // 0: upper_sw, 1: upper_se, 2: upper_ne, 3: upper_nw
    // 4: lower_sw, 5: lower_se, 6: lower_ne, 7: lower_nw
    // Avoid cells that collapsed in some way
    if (hexahedron_dz(corners) < numerics::EPSILON) {
        return 0.0;
    }

    double vol = 0.0;
    for (int i = 0; i < 6; ++i) {
        const auto &tetra = HEXAHEDRON_TETRAHEDRON_VERTICES[scheme][i];
        double tetra_vol =
          signed_tetrahedron_volume(get_corner_by_index(corners, tetra[0]),
                                    get_corner_by_index(corners, tetra[1]),
                                    get_corner_by_index(corners, tetra[2]),
                                    get_corner_by_index(corners, tetra[3]));
        vol += tetra_vol;  // using signed, to detect concave cells
    }
    return std::abs(vol);
}

/**
 * @brief Calculate the volume of a hexahedron. Robust for both convex and concave
 * cells.
 *
 * @param corners The corners of the hexahedron.
 * @return double The volume of the hexahedron.
 */

double
hexahedron_volume(const HexahedronCorners &corners)
{

    double volume0 = hexahedron_volume_per_scheme(corners, 0);
    double volume1 = hexahedron_volume_per_scheme(corners, 1);
    double volume2 = hexahedron_volume_per_scheme(corners, 2);
    double volume3 = hexahedron_volume_per_scheme(corners, 3);

    // Return the average of the volumes from the four schemes
    return (volume0 + volume1 + volume2 + volume3) / 4.0;
}

// Overload for CellCorners
double
hexahedron_volume(const CellCorners &cell_corners)
{
    // Convert CellCorners to HexahedronCorners
    HexahedronCorners hexa_corners = cell_corners.to_hexahedron_corners();

    // Call the HexahedronCorners version
    return hexahedron_volume(hexa_corners);
}

// Overload for CellCorners for legacy volume
double
hexahedron_volume_legacy(const CellCorners &cell_corners, const int precision)
{
    // Convert CellCorners to HexahedronCorners
    HexahedronCorners hexa_corners = cell_corners.to_hexahedron_corners();

    // Call the HexahedronCorners version
    return hexahedron_volume_legacy(hexa_corners, precision);
}

}  // namespace xtgeo::geometry
