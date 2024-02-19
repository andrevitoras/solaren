"""
Created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
"""
from numpy import arccos, sin, pi, tan, array, linspace, deg2rad, cos, zeros

from niopy.geometric_transforms import ang_h, ang_p, R, nrm, mid_point
from niopy.plane_curves import uinv, wmp, ump, concatenate_curves, PlaneCurve, winv, parametric_points, wme, ume
from niopy.geometric_transforms import tgs2tube

from soltracepy import OpticalSurface
from utils import rotate2D


class cpc_type:

    def __init__(self, left_conic: array, left_involute: array, right_involute: array, right_conic: array):

        self.l_conic = left_conic
        self.l_inv = left_involute
        self.r_inv = right_involute
        self.r_conic = right_conic

        self.cusp = self.l_inv[-1]

        self.left_contour = concatenate_curves(base_curve=self.l_conic, next_curve=self.l_inv)
        self.right_contour = concatenate_curves(base_curve=self.r_inv, next_curve=self.r_conic)
        self.contour = concatenate_curves(base_curve=self.left_contour, next_curve=self.right_contour)

        self.aperture_center = mid_point(p=self.contour[0], q=self.contour[-1])

        if (self.contour[0] - self.cusp)[1] >= 0:
            self.up_vector = R(pi/2).dot(nrm(self.contour[-1] - self.contour[0]))
        else:
            self.up_vector = R(-pi/2).dot(nrm(self.contour[-1] - self.contour[0]))

    def as_plane_curve(self, center=None):
        return PlaneCurve(curve_pts=self.contour,
                          curve_center=self.cusp if center is None else center)

    def as_soltrace_element(self, name: str, length: float, optic: OpticalSurface):
        return self.as_plane_curve().to_soltrace(name=name, length=length, optic=optic)


def symmetric_cpc2tube(tube_radius: float,
                       tube_center: array,
                       theta_a: float,
                       degrees=True,
                       nbr_pts=121,
                       upwards=True):

    """
    This functions returns a tuple of arrays that corresponds to the elements of a Compound Parabolic Concentrator (cpc)
    to a tube.

    :param tube_radius: the absorber tube absorber_radius, in mm.
    :param tube_center: the absorber tube center point, in mm.
    :param theta_a: the acceptance half-angle of the cpc optic.
    :param nbr_pts: the number of points to discretize the contours of the cpc_type optic.

    :param degrees: a boolean sign to indicate whether the acceptance angle was inputted in degrees or radians.
    :param upwards: a boolean sign to indicate whether the cpc aperture must be upwards (True) or downwards (False).

    :return: a tuple with the arrays which define the contour of the optics, from the left to the right side.
    """

    # This function first consider the center of the tube as the origin.
    # Then, it translates the coordinate system by the function argument 'tube_center'.
    f = array([0, 0])

    # To ensure that the acceptance half-angle used in the calculations is in radians.
    ta = theta_a * pi / 180 if degrees else theta_a

    # The cusp point. That is, where the left and right involutes touch the tube.
    A = f - array([0, tube_radius])

    # the point defined by the interception of edge-rays.
    X = f + array([0, tube_radius / sin(ta)])

    # Calculates the tangent points at the tube that passes through point X ###########
    # It is necessary in order to better define the angular parameters
    # that define each section of the cpc_type contour
    t1, t2 = tgs2tube(point=X, tube_center=f, tube_radius=tube_radius)
    if t2[0] > t1[0]:
        tl, tr = t1, t2
    else:
        tl, tr = t2, t1
    ####################################################################################

    #######################################################################################################
    # LEFT SIDE DESIGN ###################################################################################
    # The left side involute design ##############################################
    left_inv_axis = array([-1, 0])
    i1 = uinv(p=A, f=f, r=tube_radius)
    phi_1 = ang_p(array([1, 0]), left_inv_axis)
    phi_2 = ang_p(tl - X, left_inv_axis)

    # defining the contour points
    l_in = parametric_points(f=i1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    # the last point of the left involute is the starting point of the left macrofocal parabola
    s3 = l_in[-1]

    # The left macrofocal parabola design
    left_mp_axis = X - tl
    mp1 = wmp(alpha=ang_h(left_mp_axis), f=f, r=tube_radius, p=s3)
    phi_1 = ang_p(tl - X, left_mp_axis)
    phi_2 = ang_p(X - tr, left_mp_axis)
    # l_mp = array([mp1(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    l_mp = parametric_points(f=mp1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)

    #######################################################################################################

    #######################################################################################################
    # RIGHT SIDE DESIGN ###################################################################################
    right_inv_axis = array([1, 0])
    i2 = uinv(p=A, f=f, r=tube_radius)
    phi_1 = ang_p(array([1, 0]), left_inv_axis)
    phi_2 = ang_p(tr - X, right_inv_axis)
    # r_in = array([i2(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    r_in = parametric_points(f=i2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    s4 = r_in[-1]

    # The left macrofocal parabola design
    right_mp_axis = X - tr
    mp2 = ump(alpha=ang_h(right_mp_axis), f=f, r=tube_radius, p=s4)
    phi_1 = ang_p(tr - X, right_mp_axis)
    phi_2 = ang_p(X - tl, right_mp_axis)
    # r_mp = array([mp2(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    r_mp = parametric_points(f=mp2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    #######################################################################################################

    # Check if the cpc_type aperture must be upwards or downwards and rotate the optics correspondingly.
    # It returns the tuple in the order from the left to the right.
    # Obviously the order of the left involute and macrofocal parabola are inverted.
    # This reversion aims to return a continuous contour to plot when concatenating this arrays

    if upwards:
        return l_mp[::-1] + tube_center, l_in[::-1] + tube_center, r_in + tube_center, r_mp + tube_center
    else:
        l_mp = rotate2D(points=l_mp, center=A, tau=pi) + array([0, 2 * tube_radius])
        l_in = rotate2D(points=l_in, center=A, tau=pi) + array([0, 2 * tube_radius])
        r_in = rotate2D(points=r_in, center=A, tau=pi) + array([0, 2 * tube_radius])
        r_mp = rotate2D(points=r_mp, center=A, tau=pi) + array([0, 2 * tube_radius])

        return r_mp[::-1] + tube_center, r_in[::-1] + tube_center, l_in + tube_center, l_mp + tube_center


def symmetric_cec2tube(tube_center: array, tube_radius: float, source_distance: float, source_width: float,
                       nbr_pts=50, upwards=False):

    # This function first consider the center of the tube as the origin ######################
    # Then, it translates the coordinate system by the argument 'tube_center'.
    f = array([0, 0])
    ##########################################################################################

    # The edge-points (left and right) in the Source ####
    f2 = array([-0.5 * source_width, source_distance])
    f1 = array([+0.5 * source_width, source_distance])
    #####################################################

    # The edge-points (left and right) in the receiver #################################################
    # The surface points which the tangent lines passes through the edges of the emitter

    # The receiver edge regarding the left side of the emitter lies in the right of the receiver
    tan_pts = tgs2tube(point=f2, tube_center=f, tube_radius=tube_radius)
    t1 = tan_pts[0] if tan_pts[0][0] > tan_pts[1][0] else tan_pts[1]

    # The receiver edge regarding the right side of the emitter lies in the left of the receiver
    tan_pts = tgs2tube(point=f1, tube_center=f, tube_radius=tube_radius)
    t2 = tan_pts[0] if tan_pts[0][0] < tan_pts[1][0] else tan_pts[1]
    ####################################################################################################

    # The cusp point of the cec optic #############
    # This is the case for a symmetric design
    A = f - array([0, tube_radius])
    ###############################################

    # The left involute design #######################################################
    # involute parametric equation
    i1 = uinv(p=A, f=f, r=tube_radius)

    # involute axis and angular (parametric) extension
    left_inv_axis = array([-1, 0])
    phi_1 = ang_p(array([1, 0]), left_inv_axis)
    phi_2 = ang_p(t2 - f1, left_inv_axis)

    # involute parametric [x, y] points
    l_in = parametric_points(f=i1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)

    # the 'end' of the involute is the 'starting' point of the ellipse
    s3 = l_in[-1]
    ##################################################################################

    # The left macrofocal ellipse design ##########################################
    # macrofocal ellipse parametric equation
    me1 = wme(f=f, r=tube_radius, g=f1, p=s3)

    # macrofocal ellipse axis and angular (parametric) extension
    # macrofocal ellipse axis goes from the macrofocus to the point focus
    left_me_axis = f1 - f
    phi_1 = ang_p(t2 - f1, left_me_axis)
    phi_2 = ang_p(f2 - t1, left_me_axis)

    # macrofocal ellipse parametric [x,y] point
    l_me = parametric_points(f=me1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    ###############################################################################

    # The right involute design #######################################################
    # involute parametric equation
    i2 = uinv(p=A, f=f, r=tube_radius)

    # involute axis and angular (parametric) extension
    right_inv_axis = array([1, 0])
    phi_1 = ang_p(array([1, 0]), left_inv_axis)
    phi_2 = ang_p(t1 - f2, right_inv_axis)

    # involute parametric points
    r_in = parametric_points(f=i2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)

    # the 'end' of the involute is the 'starting' point of the ellipse
    s4 = r_in[-1]
    ###################################################################################

    # The right macrofocal ellipse design ##########################################
    # macrofocal ellipse parametric equation
    me2 = ume(f=f, r=tube_radius, g=f2, p=s4)

    # macrofocal ellipse axis and angular (parametric) extension
    # macrofocal ellipse axis goes from the macrofocus to the point focus
    right_me_axis = f2 - f
    phi_1 = ang_p(t1 - f2, right_me_axis)
    phi_2 = ang_p(f1 - t2, right_me_axis)

    # macrofocal ellipse [x,y] parametric points
    r_me = parametric_points(f=me2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    ################################################################################

    # Check if the cec optic aperture must be upwards or downwards and rotate the optics correspondingly.
    # It returns the tuple in the order from the left to the right.
    # Obviously the order of the left involute and macrofocal parabola are inverted.
    # This reversion aims to return a continuous contour to plot when concatenating this arrays

    if upwards:
        return l_me[::-1] + tube_center, l_in[::-1] + tube_center, r_in + tube_center, r_me + tube_center
    else:
        l_me = rotate2D(points=l_me, center=A, tau=pi) + array([0, 2 * tube_radius])
        l_in = rotate2D(points=l_in, center=A, tau=pi) + array([0, 2 * tube_radius])
        r_in = rotate2D(points=r_in, center=A, tau=pi) + array([0, 2 * tube_radius])
        r_me = rotate2D(points=r_me, center=A, tau=pi) + array([0, 2 * tube_radius])

        return r_me[::-1] + tube_center, r_in[::-1] + tube_center, l_in + tube_center, l_me + tube_center


def symmetric_cpc2evacuated_tube(theta_a: float, tube_center: array, tube_radius: float, cover_radius: float,
                                 nbr_pts=121, degrees=True, upwards=True, dy=0):

    dt = tube_center

    f = array([0, 0])

    # To ensure that the acceptance half-angle used in the calculations is in radians.
    ta = theta_a * pi / 180 if degrees else theta_a

    gap = cover_radius + dy

    A = array([0, -gap])
    t1, t2 = tgs2tube(point=A, tube_center=f, tube_radius=tube_radius)
    if t1[0] > t2[0]:
        t1, t2 = t2, t1

    # the point defined by the interception of edge-rays
    X = array([0, tube_radius / sin(ta)])

    # Calculates the tangent points at the tube that passes through point X ###########
    # It is necessary in order to better define the angular parameters
    # that define each section of the cpc_type contour
    t3, t4 = tgs2tube(point=X, tube_center=f, tube_radius=tube_radius)
    if t4[0] > t3[0]:
        tl, tr = t3, t4
    else:
        tl, tr = t3, t4

    #######################################################################################################
    # LEFT SIDE DESIGN ###################################################################################
    # The left side involute design
    left_inv_axis = array([1, 0])
    i1 = winv(p=A, f=f, r=tube_radius)
    phi_1 = ang_p(A - t1, left_inv_axis)
    phi_2 = ang_p(tl - X, left_inv_axis)

    # defining the contour points
    # l_in = array([i1(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    l_in = parametric_points(f=i1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    # the last point of the left involute is the starting point of the left macrofocal parabola
    s3 = l_in[-1]
    #######################################

    # The left macrofocal parabola design
    left_mp_axis = X - tl
    mp1 = wmp(alpha=ang_h(left_mp_axis), f=f, r=tube_radius, p=s3)
    phi_1 = ang_p(tl - X, left_mp_axis)
    phi_2 = ang_p(X - tr, left_mp_axis)
    l_mp = array([mp1(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    ########################################
    #######################################################################################################

    #######################################################################################################
    # RIGHT SIDE DESIGN ###################################################################################
    right_inv_axis = array([1, 0])
    i2 = uinv(p=A, f=f, r=tube_radius)
    phi_1 = ang_p(A - t2, left_inv_axis)
    phi_2 = ang_p(tr - X, right_inv_axis)
    r_in = array([i2(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    s4 = r_in[-1]

    # The left macrofocal parabola design
    right_mp_axis = X - tr
    mp2 = ump(alpha=ang_h(right_mp_axis), f=f, r=tube_radius, p=s4)
    phi_1 = ang_p(tr - X, right_mp_axis)
    phi_2 = ang_p(X - tl, right_mp_axis)
    r_mp = array([mp2(x) for x in linspace(start=phi_1, stop=phi_2, num=nbr_pts)])
    #######################################################################################################

    # Check if the cpc_type aperture must be upwards or downwards and rotate the optics correspondingly.
    # It returns the tuple in the order from the left to the right.
    # Obviously the order of the left involute and macrofocal parabola are inverted.
    # This reversion aims to return a continuous contour to plot when concatenating this arrays

    if upwards:
        return l_mp[::-1] + dt, l_in[::-1] + dt, r_in + dt, r_mp + dt
    else:
        l_mp = rotate2D(points=l_mp, center=A, tau=pi) + array([0, 2 * gap])
        l_in = rotate2D(points=l_in, center=A, tau=pi) + array([0, 2 * gap])
        r_in = rotate2D(points=r_in, center=A, tau=pi) + array([0, 2 * gap])
        r_mp = rotate2D(points=r_mp, center=A, tau=pi) + array([0, 2 * gap])

        return r_mp[::-1] + dt, r_in[::-1] + dt, l_in + dt, l_mp + dt


def symmetric_cec2evacuated_tube(tube_center: array, tube_radius: float, cover_radius: float,
                                 source_distance: float, source_width: float,
                                 nbr_pts=50, upwards=True, dy=0.):

    """
    This functions presents the design of a Compound Elliptical Concentrator (CEC) optic to an evacuated tube, as
    presented by Chaves [1].

    Since the cec optic cannot touch the receiver, the gap problem is addressed by the virtual receiver solution
    proposed by Winston [2]. It considers the gap between the optic and the receiver as the outer cover radius
    (and with an additional/optional displacement in the vertical direction).

    It returns a tuple of arrays that corresponds to the elements of a Compound Parabolic Concentrator (cpc)
    to a tube.

    :param tube_center: the center of the evacuated tube, a point-array in millimeters.
    :param tube_radius: the radius of absorber tube.
    :param cover_radius: the radius of the outer cover tube.

    :param source_distance: The distance between the tube center and the emitter.
    :param source_width: The width of the emitter

    :param nbr_pts: The number of point in which the cec sections are discretized
    :param upwards: A boolean sign to indicate whether an upwards (True) or a downwards (False) optic is returned
    :param dy: The additional gap to displace the cust of the cec optic

    :return: a tuple with the arrays which define the contour of the optics, from the left to the right side.


    [1] Chaves, J., 2016. Introduction to Nonimaging Optics. CRC Press, New York, 2nd Edition.
    [2] Winston, R., 1978. Ideal flux concentrators with reflector gaps.
        Applied Optics 17, 1668–1669. https://doi.org/10.1364/AO.17.001668.
    """

    # This function first consider the center of the tube as the origin ######################
    # Then, it translates the coordinate system by the function argument 'tube_center'.
    f = array([0, 0])
    ##########################################################################################

    # The edge-points (left and right) in the Source ####
    f2 = array([-0.5 * source_width, source_distance])
    f1 = array([+0.5 * source_width, source_distance])
    #####################################################

    # The receiver edge regarding the left side of the emitter lies in the right of the receiver
    tan_pts = tgs2tube(point=f2, tube_center=f, tube_radius=tube_radius)
    t1 = tan_pts[0] if tan_pts[0][0] > tan_pts[1][0] else tan_pts[1]

    # The receiver edge regarding the right side of the emitter lies in the left of the receiver
    tan_pts = tgs2tube(point=f1, tube_center=f, tube_radius=tube_radius)
    t2 = tan_pts[0] if tan_pts[0][0] < tan_pts[1][0] else tan_pts[1]
    ####################################################################################################

    # The cusp point of the cec optic #############
    # This is the case for a symmetric design

    gap = cover_radius + dy
    A = f - array([0, gap])

    t3, t4 = tgs2tube(point=A, tube_center=f, tube_radius=tube_radius)
    if t3[0] < t4[0]:
        t3, t4 = t4, t3
    ###############################################

    # The left side involute design
    left_inv_axis = array([1, 0])
    i1 = winv(p=A, f=f, r=tube_radius)
    phi_1 = ang_p(A - t4, left_inv_axis)
    phi_2 = ang_p(t2 - f1, left_inv_axis)

    l_in = parametric_points(f=i1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    # the last point of the left involute is the starting point of the left macrofocal parabola
    s3 = l_in[-1]
    #######################################

    # macrofocal ellipse parametric equation
    me1 = wme(f=f, g=f1, r=tube_radius, p=s3)
    # ellipse axis goes from the macrofocus to the point focus
    left_me_axis = f1 - f
    phi_1 = ang_p(t2 - f1, left_me_axis)
    phi_2 = ang_p(f2 - t1, left_me_axis)

    l_me = parametric_points(f=me1, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    ###############################################################################

    # Right involute
    right_inv_axis = array([1, 0])
    i2 = uinv(p=A, f=f, r=tube_radius)

    phi_1 = ang_p(A - t3, right_inv_axis)
    phi_2 = ang_p(t1 - f2, right_inv_axis)
    r_in = parametric_points(f=i2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)
    s4 = r_in[-1]

    # macrofocal ellipse parametric equation
    me2 = ume(f=f, g=f2, r=tube_radius, p=s4)
    # ellipse axis goes from the macrofocus to the point focus
    right_me_axis = f2 - f
    # opening angles defined by the edge-rays
    phi_1 = ang_p(t1 - f2, right_me_axis)
    phi_2 = ang_p(f1 - t2, right_me_axis)
    # contour surface points
    r_me = parametric_points(f=me2, phi_1=phi_1, phi_2=phi_2, nbr_pts=nbr_pts)

    # Check if the cec optic aperture must be upwards or downwards and rotate the optics correspondingly.
    # It returns the tuple in the order from the left to the right.
    # Obviously the order of the left involute and macrofocal parabola are inverted.
    # This reversion aims to return a continuous contour to plot when concatenating this arrays

    if upwards:
        return l_me[::-1] + tube_center, l_in[::-1] + tube_center, r_in + tube_center, r_me + tube_center
    else:
        l_me = rotate2D(points=l_me, center=A, tau=pi) + array([0, 2 * gap])
        l_in = rotate2D(points=l_in, center=A, tau=pi) + array([0, 2 * gap])
        r_in = rotate2D(points=r_in, center=A, tau=pi) + array([0, 2 * gap])
        r_me = rotate2D(points=r_me, center=A, tau=pi) + array([0, 2 * gap])

        return r_me[::-1] + tube_center, r_in[::-1] + tube_center, l_in + tube_center, l_me + tube_center


def real_cpc_tube_data(theta_a: float, tube_radius: float, outer_glass_radius: float):

    theta_a_rad = deg2rad(theta_a)

    beta = arccos(tube_radius / outer_glass_radius)
    s = outer_glass_radius * sin(beta.rad)

    a = 2 * (pi * tube_radius - beta * tube_radius + s) / sin(theta_a_rad)
    h = a / (2 * tan(theta_a_rad)) + tube_radius / sin(theta_a_rad)

    return a, h


def ideal_cpc_tube_data(theta_a: float, tube_radius: float):

    theta_a_rad = deg2rad(theta_a)

    a = 2 * pi * tube_radius / sin(theta_a_rad)
    h = a / (2 * tan(theta_a_rad) + tube_radius / sin(theta_a_rad))

    return a, h


def oommen_cpc(tube_radius: float,
               gap_radius: float,
               theta_a: float,
               theta_max: float = None,
               number_pts=150) -> array:

    # ToDo: include a detailed explanation regarding argument 'theta_max' in the documentation.

    """
    This function implements procedure proposed by Oomen and Jayaraman [1] to design a compound parabolic concentrator
    (CPC) to an absorber tube encapsulated by a cover -- as also detailed by Abbas et al. [2].

    The gap problem [3] is here solved by the virtual receiver design solution, as proposed by Winston [4]. Oommen and
    Jayaraman [1] denominate it as the extended receiver design.

    Lastly, the argument 'theta_max' is included to (...). See Refs. [5,6] for application examples.

    This function returns an array of [x,y] points that define the cpc optic contour.

    :param tube_radius: the radius of the absorber tube, in millimeters.
    :param gap_radius: the radius of the gap between optic and receiver, in millimeters.
    :param theta_a: the half-acceptance angle of the cpc optic, in degrees.
    :param theta_max: the maximum angle to develop the cpc optic, in degrees.
    :param number_pts: number of contour points to return.

    :return: the contour points of the cpc optic.

    [1] Oommen, R., Jayaraman, S., 2001. https://doi.org/10.1016/S0196-8904(00)00113-8.
    [2] Abbas et al., 2018. https://doi.org/10.1016/j.apenergy.2018.09.224.
    [3] Rabl, A., 1985. Active Solar Collectors and Their Applications. Oxford University Press, New York.
    [4] Winston, R., 1978. Ideal flux concentrators with reflector gaps. https://doi.org/10.1364/AO.17.001668.
    [5] Cheng et al., 2018. A novel optical optimization model for linear Fresnel reflector concentrators.
        https://doi.org/10.1016/j.renene.2018.06.019
    [6] Men et al., 2021. Study on the annual optical comprehensive performance of linear Fresnel reflector
        concentrators with an effective multi-objective optimization model.
        https://doi.org/10.1016/j.solener.2021.07.051
    """

    # storing input data in auxiliary variables ###
    r1 = tube_radius
    r2 = gap_radius
    theta_a_rad = deg2rad(theta_a)
    ###############################################

    # defining auxiliary variables #######################
    r_ratio = r2/r1
    beta = (r_ratio**2 - 1)**0.5 - arccos(1/r_ratio)
    ######################################################

    # range of the parameter used for the parametric equation #######################
    # beginning of the first segment of the optic
    theta_0 = arccos(r1/r2)
    # end of the first segment / start of the second
    theta_1 = theta_a_rad + (pi/2)
    # end of the second segment
    theta_2 = (3*pi/2) - theta_a_rad if theta_max is None else theta_max * (pi/180.)
    theta_range = linspace(start=theta_0, stop=theta_2, num=number_pts)
    #################################################################################

    # Calculating the contour points of the cpc optic ##############################################
    curve_pts = zeros(shape=(theta_range.shape[0], 2))
    for i, theta in enumerate(theta_range):

        if theta_0 <= abs(theta) <= theta_1:
            rho = r1 * (theta + beta)
        elif theta_1 < abs(theta) <= theta_2:
            num = theta + theta_a_rad + 0.5*pi + 2*beta - cos(theta - theta_a_rad)
            den = 1 + sin(theta - theta_a_rad)
            rho = r1 * num / den
        else:
            raise ValueError('Values out of range!')

        curve_pts[i] = r1 * sin(theta) - rho * cos(theta), -r1 * cos(theta) - rho * sin(theta)
    ################################################################################################

    return curve_pts
