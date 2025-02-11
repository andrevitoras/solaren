#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

from numpy import dot, cos, sin, array, arcsin, sqrt, pi, zeros, arctan, linspace
from scipy.interpolate import UnivariateSpline, splprep, splev
from niopy.geometric_transforms import dst, ang_h, ang_p, V, nrm, R, mid_point
from pathlib import Path

from pysoltrace import PySolTrace
from pysoltrace.geometries import flat_element
from soltracepy import Element, OpticalSurface
from soltracepy.auxiliary import create_csi_file


class PlaneCurve:
    """
    This class stands to generally represent plane curves, i.e., geometry contours that are fully defined in a plane,
    the x-y plane in this case.

    It is important to highlight that x-y units are considered to be in millimeters ('mm'). Therefore, attributes and
    methods are defined/constructed regarding this.
    """

    def __init__(self,
                 curve_pts: array,
                 curve_center: array):

        """
        :param curve_pts: the curve x-y points, as [x,y] or [x,0,z] point-arrays.
        :param curve_center: the curve center, as an [x,y] or [x,0,z] point-array.
        """

        # Ensure that points of the plane curve are in ascending order at the x coordinate ######
        # This is needed to generate splines interpolations for the curve
        pts = deepcopy(curve_pts)
        if pts[0][0] > pts[-1][0]:
            pts = pts[::-1]
        #########################################################################################

        # Taking the values of the curve points of the x and y coordinates ######################
        # Curve points could be in the form [x, y] or [x, 0, y]. The code works for both cases.
        self.x = pts.T[0]
        self.y = pts.T[-1]
        #########################################################################################

        # Creating an attribute for the curve points in the ascending order and as [x, y] array points #####
        self.curve_pts = zeros(shape=(self.x.shape[0], 2))
        self.curve_pts.T[:] = self.x, self.y
        ####################################################################################################

        # # An attribute for the center of the curve #######################################################
        # Old version ##################################
        # It was only applied to a PrimaryMirror object
        # (see scopy.linear_fresnel.PrimaryMirror)
        # if curve_center is None:
        #     n_m = int((len(pts) + 1) / 2)
        #     hc = pts[n_m]
        # else:
        #     hc = curve_center
        #################################################

        # New version for the curve center
        # It must be inputted as the class argument
        self.center = array([curve_center[0], curve_center[-1]])
        #####################################################

        # Attributes for the translated x and y coordinates of the curve ###################################
        # That is, the points relative position from the curve center: translated point-arrays
        # self.x_t = self.x - self.center[0]
        # self.y_t = self.y - self.center[-1]

        self.x_t, self.y_t = (self.curve_pts - self.center).T
        ####################################################################################################

    def as_spline(self, centered=False, rep=False):
        """
        A method to return the PlaneCurve object as a cubic as cubic spline.

        It can return the centered curve or not. It can also return the spline in a form where 'x' and 'y' are functions
        of a parameter 'u' which ranges from 0 to 1.
        See scipy.interpolate 'splprep', 'splev', and 'UnivariateSpline' documentations.

        :param centered: A boolean sign to indicate to return the spline regarding the curve centered points.
        :param rep: A boolean sign to return the parametrized representation of the cubic spline.

        :return: The plane curve as a cubic spline.
        """

        if centered:
            x, y = self.x_t, self.y_t
        else:
            x, y = self.x, self.y

        if not rep:
            spl = UnivariateSpline(x=x, y=y, s=0)
        else:
            spl = splprep([x, y], s=0)

        return spl

    def slope2surface(self, rep=True):
        """
        A method which returns the slope to the curve surface.
        The slope is the angle that the tangent vector makes with the positive direction of the x-axis.

        Since the curve is defined as a set of discretized point-arrays, the slope are then returned for each one of
        these points and in the same order (i.e., slope[0] is angle regarding to self.curve_pts[0]).

        :return: An array of angles, in radians.
        """
        if rep:
            tck, u = self.as_spline(rep=True)
            dx_du, dy_du = splev(u, tck, der=1)
            slope = arctan(dy_du / dx_du)
        else:
            dy_dx = self.as_spline(rep=False).derivative(n=1)
            slope = arctan(dy_dx(self.x))

        return slope

    def normals2surface(self):
        """
        A method which returns the normal vectors to the curve surface.
        A normal vector is perpendicular to the tangent vector to the surface (a positive 90 degrees rotation).

        Since the curve is defined as a set of discretized point-arrays, the normal vectors are then returned for each
        one of such points and in the same order (i.e., normals[0] is the direction perpendicular to
        the curve at the point self.curve_pts[0])

        :return: An array of [x,y] array-vectors.
        """

        # # Old version ##################################################
        # normals = zeros(shape=(self.x.shape[0], 2))
        # spl = self.as_spline()
        # d1 = spl.derivative()
        #
        # normals[:] = [nrm(V(arctan(d1(p)))) for p in self.x]
        # normals = R(pi / 2).dot(normals.T).T
        # ################################################################

        # New version ####################################################
        # ToDo: Check if this works!!!!!
        # Generalization to include curves that do not fit in the mathematical definition of a function y = y(x).
        # That is, x1 will only and only yield a unique value: y1 = y(x1). E.g., curve points of a circle.

        slope = self.slope2surface()
        normals = array([nrm(V(beta)) for beta in slope])
        normals = R(pi/2).dot(normals.T).T
        ##################################################################

        return normals

    def to_soltrace(self, name: str, length: float, optic: OpticalSurface):
        """
        This method returns the PlaneCurve object as a set of flat Elements objects. That is, it approximates two
        consecutive contour points as a straight line and construct a flat Element between them.

        :param name: the variable to give for each Element, inputted in the 'comment' argument.
        :param length: the length of the Elements, in millimeters.
        :param optic: the optical property associated with the Elements, as an OpticalSurface object.

        :return: A list of Elements objects.
        """

        elements = []
        for i in range(len(self.curve_pts) - 1):

            # Selecting the two consecutive points in the PlaneCurve #######
            # that will define the flat segment
            pt_a = self.curve_pts[i]
            pt_b = self.curve_pts[i + 1]
            ################################################################

            # Calculates the ecs_origin and the aim-point of the flat element #####
            origin = mid_point(p=pt_a, q=pt_b)
            aim_pt = origin + 100 * R(-pi / 2).dot(pt_b - pt_a) * dst(p=pt_a, q=pt_b)
            ###################################################################

            # Converts to meters and to a [x, 0, z] arrays #########
            origin = array([origin[0], 0, origin[-1]]) / 1000
            aim_pt = array([aim_pt[0], 0, aim_pt[-1]]) / 1000
            ########################################################

            # Calculates the width of the flat Element ####
            width = round(dst(p=pt_a, q=pt_b) / 1000, 4)
            ###############################################

            # Defining the aperture of the Element ########
            aperture = list([0] * 9)
            aperture[0:3] = 'r', width, round(length / 1000, 4)
            ###############################################

            # Defining the surface of the Element #########
            surface = list([0] * 9)
            surface[0] = 'f'
            ###############################################

            # Appending the current Element to the elements list ############
            elements += [Element(name=f"{name}_{i}",
                                 ecs_origin=origin.round(4), ecs_aim_pt=aim_pt.round(4), z_rot=0,
                                 aperture=aperture, surface=surface,
                                 optic=optic,
                                 reflect=True, enable=True)]
            #################################################################

        return elements

    def spline2soltrace(self, file_path: Path, file_name: str) -> Path:
        """
        This method considers the SolTrace interpretation of an Element as a rotationally symmetric cubic spline. Then,
        it creates a 'csi' file to be read by SolTrace and returns the full path of such a file.

        :param file_path: The file path of where the 'csi' file should be created.
        :param file_name: The variable_name of the 'csi' file to be created.

        :return: The file full path
        """

        ###########################################################################
        # Constructs the as_spline ##################################################
        # Calculates the first derivative values at both edges knots.
        # At the first and last points, the parameters are 0 and 1
        tck, u = self.as_spline(centered=False, rep=True)
        dx_du, dy_du = splev([0, 1], tck, der=1)
        dy_dx = dy_du / dx_du
        df_1, df_n = dy_dx
        ########################################################################

        # New implementation ################################################################################
        # In 13-Mar-2023
        full_file_path = create_csi_file(curve_pts=self.curve_pts/1000, knots_derivatives=(df_1, df_n),
                                         file_path=file_path, file_name=file_name)
        #####################################################################################################

        # # Old implementation #########################################################################################
        # # Creates the surface cubic as_spline file ######################################################
        # # Creating the as_spline file path.
        # # A 'csi' extension file for SolTrace to correctly read it.
        # full_file_path = Path(file_path, f"{file_name}.csi")
        #
        # # creating file and writing the data into it.
        # file = open(full_file_path, 'w')
        # file.write(f"{len(x)}\n")  # the first line must contain the number of points which defines the surface
        # for i in range(len(x)):
        #     # write in the file the point coordinates values in meters
        #     file.write(f"{x[i]}\t{y[i]}\n")
        #
        # # the last line should contain the first derivatives at both edge knots.
        # file.write(f"{df_1}\t{df_n}")  # writes the first derivatives at both edges knots
        # file.close()  # closes the file
        # ##############################################################################################################
        # ##############################################################################################################

        return full_file_path

    def to_pysoltrace(self, length: float,
                      parent_stage: PySolTrace.Stage, id_number: int,
                      optic: PySolTrace.Optics) -> list:

        elements = []
        for i in range(len(self.curve_pts) - 1):
            # Selecting the two consecutive points in the PlaneCurve #######
            # that will define the flat segment
            pt_a = self.curve_pts[i]
            pt_b = self.curve_pts[i + 1]
            ################################################################

            # Calculates the ecs_origin and the aim-point of the flat element #####
            origin = mid_point(p=pt_a, q=pt_b)
            aim_pt = origin + 100 * R(-pi / 2).dot(pt_b - pt_a) * dst(p=pt_a, q=pt_b)
            ###################################################################

            # Converts to meters and to a [x, 0, z] arrays #########
            origin = array([origin[0], 0, origin[-1]]) / 1000
            aim_pt = array([aim_pt[0], 0, aim_pt[-1]]) / 1000
            ########################################################

            # Calculates the width of the flat Element ####
            width = dst(p=pt_a, q=pt_b) / 1000
            ###############################################

            elements += [flat_element(width=width, length=length/1000,
                                      ecs_origin=origin, ecs_aim=aim_pt,
                                      parent_stage=parent_stage, id_number=id_number+i,
                                      optic=optic)]

        return elements

    def export2rhino(self, script):
        """
        This method writes the lines of code for the Rhino CAD software understand the PlaneCurve object as a curve.

        :param script: The script file in which the lines of code should be written in.
        """

        script.write(f'-_InterpCrv\n')
        for pt in self.curve_pts:
            script.write(f'{pt[0]},{pt[1]}\n')
        script.write('-_Enter\n')


########################################################################################################################
# Auxiliary functions ##################################################################################################

def concatenate_curves(base_curve: array, next_curve: array, precision=0.00001):

    k = 0
    while dst(next_curve[k][0], base_curve[-1][0]) <= precision:
        k = k + 1

    concatenated_curve = array(base_curve.tolist() + next_curve[k:].tolist())

    # concatenated_curve = array(base_curve.tolist() + next_curve.tolist())

    return concatenated_curve


########################################################################################################################
########################################################################################################################


########################################################################################################################
# Plane curve functions from Julio Chaves book #########################################################################
"""
Chaves, J. Introduction to Nonimaging Optics. New York, 2nd Edition: CRC Press; 2016.
"""


def par(alpha: float, f: array, p: array):

    """
    This function returns a parametric function of a parabola tilted by an angle 'alpha' to the horizontal,
    with focus 'f' and that goes through a point 'p'.

    :param alpha: the parabola's tilt angle from the horizontal, in radians.
    :param f: the parabola's focal point, an array.
    :param p: a point which the parabola goes through, an array.

    :return: a parametric function (a Python callable).
    """

    def fc(x):
        num = dst(p, f) - dot(p - f, V(alpha))
        den = 1 - cos(x)
        return (num / den) * V(x + alpha) + f

    return fc


def eli(f: array, g: array, p: array):

    """
    :param f: one of the ellipse's focus.
    :param g: one of the ellipse's focus.
    :param p: a point which the ellipse goes through.

    :return: a parametric function of an ellipse with focus at 'f' and 'g' and that goes through a point 'p'.
    """
    # ToDo: check if it works.
    alpha = ang_h(g - f)
    k = dst(f, p) + dst(p, g)
    d_fg = dst(f, g)
    num = k ** 2 - d_fg ** 2

    def fc(x):
        den = 2 * (k - d_fg * cos(x))

        return f + (num / den) * V(x + alpha)

    return fc


def hyp(f: array, g: array, p: array):
    """
    This function returns a parametric function of a hyperbola with foci 'f' and 'g'
    and passing through point 'p'

    :param f: Hyperbola focus, an array point
    :param g: Hyperbola focus, an array point
    :param p: A point the that the hyperbola passes through

    :return: A parametric function
    """

    alpha = ang_h(g - f)
    k = abs(dst(f, p) - dst(p, g))
    d_fg = dst(f, g)
    num = k ** 2 - d_fg ** 2

    def fc(y):
        den = 2 * (k - d_fg * cos(y))

        return f + (num / den) * V(y + alpha)

    return fc


def winv(p: array, f: array, r: float):
    """
    This function returns a parametric function of a winding involute to a circle centered at point 'f' with
    absorber_radius 'r', and that goes through point 'p'.

    :param p: A point that the involute passes through.
    :param f: Center point of the circle.
    :param r: Radius of the circle

    :return: A parametric function
    """

    d_pf = dst(p, f)
    phi_p = ang_h(p - f) + arcsin(r / d_pf)
    k = sqrt(d_pf ** 2 - r ** 2) + r * phi_p

    def fc(x):
        return r * array([sin(x), -cos(x)]) + (k - r * x) * V(x) + f

    return fc


def uinv(p: array, f: array, r: float):
    """
    This function returns a parametric function of an unwinding involute to a circle with center at 'f' with
    absorber_radius 'r', and that goes through point 'p'.

    :param p: A point that the involute passes through.
    :param f: Center point of the circle.
    :param r: Radius of the circle.

    :return: A parametric function.
    """

    d_pf = dst(p, f)

    phi_p = ang_h(p - f) - arcsin(r / d_pf)
    phi_p = 2 * pi + phi_p if phi_p < 0 else phi_p
    k = sqrt(d_pf ** 2 - r ** 2) - r * phi_p

    def fc(x):
        return r * array([-sin(x), cos(x)]) + (k + r * x) * V(x) + f

    return fc


def wmp(alpha: float, f: array, r: float, p: array):
    """
    This function returns a parametric function of a winding macrofocal parabola tilted by an angle 'alpha'
    to the horizontal, with macrofocus centered at 'f' and absorber_radius 'r', and that goes through point 'p'.

    :param alpha: the macrofocal parabola tilt angle from the horizontal, in radians.
    :param f: the macrofocus center point.
    :param r: the macrofocus absorber_radius.
    :param p: a point the macrofocal parabola passes through.

    :return: a parametric function.
    """

    phi_p = ang_p(p - f, V(alpha)) + arcsin(r / dst(p, f))

    k = sqrt(dst(p, f) ** 2 - r ** 2) * (1 - cos(phi_p)) + r * (1 + phi_p - sin(phi_p))

    def fc(x):
        num = k + r * (sin(x) - 1 - x)
        den = 1 - cos(x)

        return r * array([sin(x + alpha), -cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def ump(alpha: float, f: array, r: float, p: array):
    """
    This function returns a parametric function of an unwinding macrofocal parabola tilted by an angle 'alpha'
    to the horizontal, with macrofocus centered at 'f' and absorber_radius 'r', and that goes through point 'p'.

    :param alpha: the macrofocal parabola tilt angle from the horizontal, in radians.
    :param f: the macrofocus center point.
    :param r: the macrofocus absorber_radius.
    :param p: a point the macrofocal parabola passes through.

    :return: a parametric function.
    """

    phi_p = ang_p(p - f, V(alpha)) - arcsin(r / dst(p, f))
    phi_p = 2 * pi + phi_p if phi_p < 0 else phi_p

    k = sqrt(dst(p, f) ** 2 - r ** 2) * (1 - cos(phi_p)) + r * (1 - phi_p + sin(phi_p))

    def fc(x):
        num = k + r * (x - 1 - sin(x))
        den = 1 - cos(x)

        return r * array([-sin(x + alpha), cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def wme(f: array, r: float, g: array, p: array):
    """
    This function returns a parametric function of a winding macrofocal ellipse with macrofocus centered at 'f'
    with absorber_radius 'r', point focus at 'g' and that goes through point 'p'.

    :param f: the ellipse macrofocus center point.
    :param r: the macrofocus absorber_radius.
    :param g: the ellipse point focus.
    :param p: a point that the macrofocal ellipse passes through.

    :return: a parametric function.
    """

    alpha = ang_h(g - f)
    ff = dst(g, f)

    phi_p = ang_p(p - f, V(alpha)) + arcsin(r / dst(p, f))
    tp = sqrt(dst(p, f) ** 2 - r ** 2)
    k = tp + r * phi_p + sqrt(ff ** 2 + r ** 2 + tp ** 2 - 2 * ff * (tp * cos(phi_p) + r * sin(phi_p)))

    def fc(x):
        num = (k - r * x) ** 2 + 2 * ff * r * sin(x) - ff ** 2 - r ** 2
        den = 2 * (k - r * x - ff * cos(x))

        return r * array([sin(x + alpha), -cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def ume(f: array, r: float, g: array, p: array):
    """
    This function returns a parametric function of an unwinding macrofocal ellipse with macrofocus centered
    at 'f' with absorber_radius 'r', point focus at 'g' and that goes through point 'p'.

    :param f: the ellipse macrofocus center point.
    :param r: the macrofocus absorber_radius.
    :param g: the ellipse point focus.
    :param p: a point that the macrofocal ellipse passes through.

    :return: a parametric function.
    """

    alpha = ang_h(g - f)
    ff = dst(g, f)
    tp = sqrt(dst(p, f) ** 2 - r ** 2)

    phi = ang_p(p - f, V(alpha)) - arcsin(r / dst(p, f))
    phi_p = 2 * pi + phi if phi < 0 else phi

    k = tp - r * phi_p + sqrt(ff ** 2 + r ** 2 + tp ** 2 - 2 * ff * (tp * cos(phi_p) - r * sin(phi_p)))

    def fc(x):
        num = (k + r * x) ** 2 - 2 * ff * r * sin(x) - ff ** 2 - r ** 2
        den = 2 * (k + r * x - ff * cos(x))

        return r * array([-sin(x + alpha), cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc

########################################################################################################################

########################################################################################################################
# Auxiliary functions ##################################################################################################


def parametric_points(f: callable, phi_1: float, phi_2: float, nbr_pts: int):
    return array([f(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])

########################################################################################################################
########################################################################################################################
