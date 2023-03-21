# -*- coding: utf-8 -*-
"""Module/class for a triangle surface (i.e. unstructured surface) with XTGeo.

Triangle surfaces are made in a similar manner as the Roxar API, but XTGeo allows
that each vertex has additional attributes. Hence the vertices are stored in a Pandas
dataframe on form:

X_UTME    Y_UTMN     Z_TVDSS   ATTRIBUTE1   ATTRIBUTE2 ...
456228.2  7677772.2  2328.7    20.0         9992.9               << vertex 0
456238.2  7679272.2  2322.7    24.0         9932.4               << vertex 1
456268.2  7674472.2  2123.7    30.0         9922.2               << vertex 2
...

The 3 first columns are mandatory, and the third column may typically be renamed
(cg. treatment of Points and Polygons)

Then there is a numpy array if integers defining the triangle connections, where the
numbers refers to vertex index:

[[0 1 2]
 [2 4 6]
  ...
]

Usage::

   >>> import xtgeo
   >>> mysurf = xtgeo.trianglesurface_from_file("surface.xtg")

or::

   >>> mysurf = xtgeo.trianglesurface_from_roxar(project, 'TopX', 'DepthSurface')

"""
# import functools
# import io
# import math
# import numbers
# import pathlib
# import warnings
# from collections import OrderedDict
# from copy import deepcopy
# from types import FunctionType
from typing import Optional

# import deprecation
import numpy as np
import pandas as pd

import xtgeo

# import xtgeo.common.sys as xtgeosys
# from xtgeo.common.constants import VERYLARGENEGATIVE, VERYLARGEPOSITIVE

# from . import (
#     _regsurf_cube,
#     _regsurf_cube_window,
#     _regsurf_export,
#     _regsurf_grid3d,
#     _regsurf_gridding,
#     _regsurf_import,
#     _regsurf_oper,
#     _regsurf_roxapi,
#     _regsurf_utils,
# )

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class TriangleSurface:
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        triangles: Optional[np.ndarray] = None,
    ):
        """Instatating the TriangleSurface class.

        dataframe: A pandas dataframe defining the vertices (points)
        triangles: A numpy array with shape (n, 3) defining the triangles

        """
        self._df = dataframe
        self._triangles = triangles

    @classmethod
    def _read_whatever_file(cls, mfile):
        ...

    # ==================================================================================
    # Public properties
    # ==================================================================================

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, indata):
        self._validate_dataframe(indata)

    # ==================================================================================
    # Public methods
    # ==================================================================================

    # ==================================================================================
    # Private methods
    # ==================================================================================
