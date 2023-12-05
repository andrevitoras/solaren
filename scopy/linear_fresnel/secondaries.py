"""
Created by:
andrevitoras@gmail.com / avas@uevora.pt 2023-05-11 11:28

This module aims to hold function and classes related to the design of secondary optics for linear Fresnel collectors.
The respective sources which present the designs are cited in the functions.
"""
from copy import deepcopy

from matplotlib import pyplot as plt
from numpy import array, arctan, sin, cos, pi, arcsin, linspace, arccos, tan
from portion import closed, Interval, empty
from scipy.optimize import fsolve

from niopy.geometric_transforms import dst, nrm, R, tgs2tube, ang_pn, V, isl, ang_h
from niopy.plane_curves import par, hyp

########################################################################################################################
# Auxiliary functions ###### ###########################################################################################


def primary_field_edge_points(primaries: array):
    f1 = primaries[0][-1] if primaries[0][-1][0] > primaries[0][-1][-1] else primaries[0][0]  # right side edge
    f2 = primaries[-1][-1] if primaries[-1][-1][0] < primaries[-1][0][0] else primaries[-1][0]  # left side edge

    return f1, f2


def primary_field_edge_mirrors_centers(primaries: array):
    n_pts = len(primaries[0])
    k = int((n_pts - 1) // 2)

    f1 = primaries[0][k] if primaries[0][k][0] > primaries[-1][k][0] else primaries[-1][k]
    f2 = primaries[-1][k] if primaries[-1][k][0] < primaries[0][k][0] else primaries[0][k]

    return f1, f2


def reduce_interval(a: Interval, b: Interval) -> Interval:
    """
    :param a: continuous interval [a1, a2]: a2 > a1 -- portion.closed(a1, a2)
    :param b: continuous interval [b1, b2]: b2 > b1 -- portion.closed(b1, b2)
    :return: This functions returns the difference between 'a' and its intersection with 'b',
    and checks for an empty interval
    """
    c = a - (a & b)
    if c == empty():
        c = closed(0, 0)

    return c


########################################################################################################################
########################################################################################################################


########################################################################################################################
# Zhu's adaptative secondary optic design ##############################################################################
"""
The following functions comprises implementations of Zhu's adaptative method to design an LFC secondary optic.
In this sense, it contains a main function and other auxiliary functions.

[1] Zhu G. New adaptive method to optimize the secondary reflector of linear Fresnel collectors.
    Solar Energy 2017;144:117–26. https://doi.org/10.1016/j.solener.2017.01.005.
"""


def principal_incidence(point: array, primaries: array,
                        tube_center: array, tube_radius: float,
                        central_incidence=False):
    """
    This function calculates the so-called principal direction.
    It is a key parameter to define the secondary optic contour.

    :param point: the current point in the secondary surface contour, a [x, y] point-array.
    :param primaries: the primary field an array of point-arrays (heliostats).
    :param tube_center: the center point of the absorber tube, a [x, y] point-array.
    :param tube_radius: the radius of the absorber tube
    :param central_incidence: a boolean sign to calculate by central incidence (True) or intensity analysis (False).

    :return: a vector-array that represents the principal incidence direction
    """

    # A unit vector which defines the x-axis.
    Ix = array([1, 0])

    ####################################################################################################################
    # The angular view in which the point sees the receiver ############################################################

    # limiting vector -- from the point to the absorber surface
    tg1, tg2 = tgs2tube(point=point, tube_center=tube_center, tube_radius=tube_radius)
    vll, vlr = tg1 - point, tg2 - point

    # The angles in which the limiting vectors regarding the x-axis.
    # A counterclockwise direction means a positive angle. A clockwise one means a negative values.
    # With this definition, angles are within -pi to +pi
    theta_1, theta_2 = ang_pn(v=vll, u=Ix), ang_pn(v=vlr, u=Ix)

    # The angular interval defined by the above two angles.
    # It is the finite interval from the range between -pi to +pi in which the point in the secondary's surface
    # sees the tubular absorber
    blockage_interval = closed(min(theta_1, theta_2), max(theta_1, theta_2))

    ####################################################################################################################

    ####################################################################################################################
    # The angular view in which the point sees the primary field #######################################################

    # edges points of the primary field
    # the first one is the right edge, the other one is the left edge.
    # f1, f2 = primary_field_edge_points(primaries)
    f1, f2 = primary_field_edge_mirrors_centers(primaries)

    # incidence vectors to the edge. They define the range of incidences in the point on the secondary surface
    v1 = nrm(f1 - point)
    v2 = nrm(f2 - point)

    # Calculating the angular view in which the point sees the primary field
    theta_3, theta_4 = ang_pn(v=v1, u=Ix), ang_pn(v=v2, u=Ix)
    incidence_interval = closed(min(theta_3, theta_4), max(theta_3, theta_4))
    ####################################################################################################################

    ####################################################################################################################
    # Accounting for the absorber blockage #############################################################################
    net_interval = reduce_interval(a=incidence_interval, b=blockage_interval)

    # The angular interval defined by the receiver acceptance is the blockage interval.
    # If it lies within the range of the angular interval in which the point sees the primary field, the 'net_interval'
    # is divided in two intervals.

    # Calculating the principal incidence ##############################################################################
    # If the 'net_interval' has only one interval, only it must be used to compute the principal incidence direction
    if len(net_interval) == 1:
        incidence_interval = closed(net_interval.lower, net_interval.upper)

    # If two intervals are produced by the blockage calculations, then it should be considered only one.
    # The problem is: which one?
    elif len(net_interval) == 2:
        # One proposition ########################################################################
        # A point in the left side of the secondary's surface will just considers the incidence range from the left.
        # This is suggested by Zhu in Fig. 2 and Fig. 3 of the paper.
        if point[0] < tube_center[0]:
            incidence_interval = closed(net_interval[0].lower, net_interval[0].upper)
        else:
            incidence_interval = closed(net_interval[-1].lower, net_interval[-1].upper)
        ##########################################################################################

        # Other proposition ######################################################################
        # It does not mater which side the point is. It only accounts for the farthest interval from the point.
        # That is, the one which has the highest deviation from the position -pi/2
        # if net_interval[0].lower + pi/2 > net_interval[1].upper + pi/2:
        #     incidence_interval = closed(net_interval[0].lower, net_interval[0].upper)
        # else:
        #     incidence_interval = closed(net_interval[1].lower, net_interval[1].upper)
        # Results of this proposition does not seem right, nonetheless.
        ###########################################################################################
    # If more than two intervals are generated, it raises an error.
    else:
        print(point)
        raise ValueError('More than two intervals were produced by the receiver blockage calculations.')
    ####################################################################################################################

    ####################################################################################################################
    # Determining the principal incidence ##############################################################################
    # It calculates the angular direction from the x-axis and then returns the vector
    # This vector points from the point in the secondary surface to the correspondent point in the primary field
    # which define the principal incidence

    # If the central incidence approach is considered. It is the most straightforward one since the intensity analysis
    # is neglected. Thus, the principal incidence is the bisector vector.
    if central_incidence:
        angular_direction = 0.5 * (incidence_interval.lower + incidence_interval.upper)
    # If an intensity analysis is considered. Then, the acceptance of the point is considered.
    else:
        point_acceptance = 2 * arcsin(tube_radius / dst(p=point, q=tube_center))
        # If the point is on the left side of the secondary's surface, the most distant edge is the left one.
        if point[0] < tube_center[0]:
            angular_direction = incidence_interval.upper - 0.5 * point_acceptance
        else:
            angular_direction = incidence_interval.lower + 0.5 * point_acceptance

        # if abs(incidence_interval.lower + pi/2) > abs(incidence_interval.upper + pi/2):
        #     most_distant_direction = incidence_interval.lower
        #     a = most_distant_direction + 0.5 * point_acceptance
        # else:
        #     most_distant_direction = incidence_interval.upper
        #     a = most_distant_direction - 0.5 * point_acceptance
    ####################################################################################################################
    ####################################################################################################################

    return V(angular_direction)


def adaptative_advance(staring_point: array, primaries: array,
                       tube_center: array, tube_radius: float,
                       ds: float, central_incidence: bool) -> array:
    """
    This function implement the adaptative advance as defined by Zhu [1].

    It considers a starting point, calculates the principal direction, and then advance with the adaptative approach
    to determine the other points that define the secondary optic contour.

    :param staring_point: the starting point of the secondary optic, a [x, y] point-array.
    :param primaries:
    :param tube_center: the center point of the absorber tube, a [x, y] point-array.
    :param tube_radius: the radius of the absorber tube
    :param ds: the length of the adaptative step, in millimeters
    :param central_incidence: a boolean sign to calculate by central incidence (True) or intensity analysis (False).
    :return:
    """

    aim = array([tube_center[0], tube_center[-1]])
    p = deepcopy(staring_point)

    # A list to hold the calculated points for the secondary optic surface
    secondary_pts = [p]
    ####################################################################################################################

    ####################################################################################################################
    # The adaptative calculations for the secondary optic surface points ###############################################
    while secondary_pts[-1][0] < aim[0]:
        # Principal direction calculation
        # That are two options: a simple approach with the central incidence, and a more complicated one.
        principal_direction = principal_incidence(primaries=primaries, tube_center=tube_center,
                                                  tube_radius=tube_radius, point=p, central_incidence=central_incidence)
        principal_direction = nrm(principal_direction)
        target_direction = nrm(aim - p)

        # Normal and tangent direction at the point
        # It considers the incident ray as the principal incidence, and the reflected one as the target direction
        normal_vector = principal_direction + target_direction
        # In this case, both vector are from the point in the secondary's surface to the points in the primary field,
        # and aim at the absorber, respectively.
        # Therefore, by such an equation the normal vector at the point also points out of it, i.e, towards the field.

        # The tangent vector at the surface point.
        tangent_vector = R(0.5 * pi).dot(normal_vector)

        # Calculating the new point in the secondary surface and append it to the list of points.

        # # Zhu's proposition for a better surface approximation ############################################
        # p2 = p + ds * nrm(tangent_v)
        # pd2 = principal_incidence(primaries=primaries, tube_center=tube_center,
        #                           tube_radius=tube_radius, point=p, central_incidence=central_incidence)
        # pd2 = nrm(pd2)
        # target_direction2 = nrm(aim - p2)
        # normal2 = pd2 + target_direction2
        #
        # theta = ang(normal_v, normal2).rad
        # tangent_v = R(0.5 * pi + 0.5*theta).dot(normal_v)
        #####################################################################################################

        # The next point in the secondary's surface
        p = p + ds * nrm(tangent_vector)

        secondary_pts.append(p)

    ####################################################################################################################

    return array(secondary_pts)


def zhu_adaptative_secondary(primaries: array, centers: array, widths: array,
                             tube_center: array, tube_radius: float,
                             delta_h: float, source_rms_width: float, ds=0.1,
                             flat_mirrors=False, central_incidence=False):
    """
    This function is based on the Zhu's [1] adaptative method to design a linear Fresnel secondary optic for a tubular
    absorber.
    """

    aim = array([tube_center[0], tube_center[-1]])

    ####################################################################################################################
    # The starting point calculations ##################################################################################

    # Secondary aperture ########################################
    # the angular aperture which comprises 95% of total power of the effective source beam.
    beta = 4 * source_rms_width  # the 95% criterion

    # Data of the farthest mirror
    cn = centers[0] if centers[0][0] > centers[-1][0] else centers[-1]  # center
    ln = dst(cn, aim)  # distance between the center and the aim point
    wn = widths[0] if centers[0][0] > centers[-1][0] else widths[-1]  # width
    tau_n = arctan(cn[0] / aim[-1])  # tracking angle for a normal incidence

    # Secondary optic aperture. It considers flat and bent mirrors (parabolical or cylindrical)
    a = ln * sin(beta) + wn * cos(tau_n) if flat_mirrors else ln * sin(beta)
    ###############################################################

    # The starting point ###########################################################
    # Zhu's criterion for the starting point. It is one of the edges of the aperture.
    # Here, it is considered the edged on the left side of the absorber tube
    p = aim - array([a / 2, delta_h])
    # The starting point ###########################################################

    secondary_points = adaptative_advance(staring_point=p, primaries=primaries,
                                          tube_center=tube_center, tube_radius=tube_radius,
                                          ds=ds, central_incidence=central_incidence)

    return secondary_points


# def my_adaptative_secondary(primaries: array, centers: array, widths: array,
#                             tube_center: array, tube_radius: float,
#                             start_point: array, ds=0.1,
#                             flat_mirrors=False, central_incidence=False):
#     pass

########################################################################################################################
########################################################################################################################

def parabolic_wings(field_width: float, tube_center: array, tube_radius: float, gap_size: float, num_points=120):
    """
    This function implements the Parabolic Wings Concentrator (PWC) design proposed by Grena and Tarquini [1] as a
    secondary optic for linear Fresnel collectors.
    It returns an array of [x,y] points that represents one symmetrical segment of the PWC optic.

    :param field_width: the total width of the primary field, in millimeters.
    :param tube_center: the center point of the absorber tube, in millimeters.
    :param tube_radius: the radius of the absorber tube, in millimeters.
    :param gap_size: the size of the gap between absorber and the PWC optic.
    :param num_points: the number of point to discretize the contour.

    :return: returns an array with the contour points of the optic.

    [1] Grena, R., Tarquini, P., 2011. https://doi.org/10.1016/j.energy.2010.12.003.
    """

    # Defining auxiliary variables #################################
    # height of the absorber tube
    h = tube_center[-1]

    # half-width of the primary field
    w = field_width / 2

    # angular size of the field regarding the absorber center
    theta_max = arctan(w/h)

    # parameter 'i' in Fig. 2 of Ref.[1]
    par_rotating_angle = theta_max / 2
    ################################################################

    # Calculating the parametric equation of the tilted parabola #####################################
    # point in the vertical axis that the tilted parabola passes through
    # it is like a cusp of the optic
    y_intercept_point = array([0, tube_radius + gap_size])

    # parametric equation of the tilted parabola
    par_f = par(alpha=1.5 * pi + par_rotating_angle,
                f=array([0, 0]),
                p=y_intercept_point)

    # angular range of the parameter
    # angles are measured from the parabola optical axis (from the vertex to the foci)
    phi_1 = pi - par_rotating_angle
    phi_2 = pi/2
    ##################################################################################################

    # tilted parabola points defined within the parameter range
    secondary_points = array([par_f(x) for x in linspace(start=phi_1, stop=phi_2, num=num_points)])

    return secondary_points + tube_center

########################################################################################################################
########################################################################################################################

########################################################################################################################
# CPC secondary for LFCs ###############################################################################################


def virtual_receiver_perimeter(tube_radius: float, cover_outer_radius: float):
    r = abs(tube_radius)
    rg = abs(cover_outer_radius)

    beta = arccos(r / rg)

    return 2 * r * (pi - beta + tan(beta))


def edges2tube(f1: array, f2: array, tube_center: array, tube_radius):
    tg1, tg2 = tgs2tube(point=f1, tube_center=tube_center, tube_radius=tube_radius)
    t1 = tg1 if tg1[0] < tube_center[0] else tg2

    tg3, tg4 = tgs2tube(point=f2, tube_center=tube_center, tube_radius=tube_radius)
    t2 = tg3 if tg3[0] > tube_center[0] else tg4

    return t1, t2


def hotel_strings(f1, f2, s1, s2):
    a = dst(f1, s1) - dst(f1, s2)
    b = dst(f2, s1) - dst(f2, s2)

    return abs(a) + abs(b)


def aperture_from_flow_line(f1, f2, e1, e2, flow_line, phi):
    p = flow_line(phi)
    s1 = isl(p=f1, v=p - f1, q=f2, u=e2)
    s2 = isl(p=f2, v=p - f2, q=f1, u=e1)

    return s1, s2


def cpc_aperture(f1: array, f2: array, tube_center: array, tube_radius):
    t1, t2 = edges2tube(f1=f1, f2=f2, tube_center=tube_center, tube_radius=tube_radius)
    e1, e2 = t1 - f1, t2 - f2

    A = isl(p=f1, v=e1, q=f2, u=e2)

    if dst(f1, tube_center) > dst(f2, tube_center):
        flow_line = hyp(f=f1, g=f2, p=A)
        phi_0 = ang_h(A - f1) - pi
    else:
        flow_line = hyp(f=f2, g=f1, p=A)
        phi_0 = ang_h(A - f2)

    def delta_etendue(phi):

        a1, a2 = aperture_from_flow_line(f1=f1, f2=f2, e1=e1, e2=e2, flow_line=flow_line, phi=phi)
        u = hotel_strings(f1=f1, f2=f2, s1=a1, s2=a2)

        return u - 4 * pi * tube_radius

    phi_a = fsolve(delta_etendue, x0=phi_0)[0] - pi
    s1, s2 = aperture_from_flow_line(f1=f1, f2=f2, e1=e1, e2=e2, flow_line=flow_line, phi=phi_a)

    return s1, s2

########################################################################################################################
########################################################################################################################

#
# t_radius = 35
# g_radius = 62.5
# t_center = array([0, 8000])
#
# tube_points = t_radius * array([[sin(t), cos(t)] for t in linspace(0, 2*pi, 50)]) + t_center
# cover_points = g_radius * array([[sin(t), cos(t)] for t in linspace(0, 2*pi, 50)]) + t_center
# in_cover_points = (g_radius - 3) * array([[sin(t), cos(t)] for t in linspace(0, 2*pi, 50)]) + t_center
#
#
# pwc_optic = parabolic_wings(field_width=16560,
#                             tube_center=t_center,
#                             tube_radius=t_radius,
#                             gap_size=g_radius - t_radius)
#
# fig = plt.figure(dpi=300)
# plt.plot(*tube_points.T, color='black')
# plt.plot(*cover_points.T, color='red', ls='dashed')
# plt.plot(*in_cover_points.T, color='red', ls='dashed')
#
#
# plt.plot(*pwc_optic.T, color='blue', label="Parabolic Wings Concentrator")
#
# plt.xlim(t_center[0] - 120, t_center[0] + 120)
# plt.ylim(t_center[1] - 120, t_center[1] + 120)
#
# plt.legend()
# plt.show()
