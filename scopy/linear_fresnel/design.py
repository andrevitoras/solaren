from copy import deepcopy

from numpy import array, linspace, sqrt, zeros, tan, cos, sin, pi, deg2rad, cross, power, absolute
from scipy.optimize import fsolve

from niopy.geometric_transforms import ang, dst, V


########################################################################################################################
# LFR design functions #################################################################################################

def design_cylindrical_heliostat(center: array, width: float, radius: float, nbr_pts=51):
    """
    This function returns the surface points of a cylindrical shape heliostat.
    This set of points define the heliostat contour.

    Units should be in millimeters to be coherent with the classes, methods, and functions presented in this module.

    :param center: the mirror center point
    :param width: the mirror width
    :param radius: the mirror curvature radius
    :param nbr_pts: the number of points in which the surface is discretized

    :return: An array of points that define the contour of the cylindrical surface.
    """

    # Auxiliary variables ##############################################################################################

    # ensure that the center point is a [x, z] array point.
    hc = array([center[0], center[-1]])

    # ensure that positive values of width and radius are used
    w = abs(width)
    r = abs(radius)

    # ensure an odd number of point to represent the heliostat surface
    # in this way center of the mirror is also an element in the array of points which describes
    # the heliostat surface.
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    ####################################################################################################################

    ####################################################################################################################
    # Calculations #####################################################################################################

    # The array of x-values which the heliostat ranges.
    x_range = linspace(start=-0.5 * w, stop=+0.5 * w, num=n_pts)

    # the function which analytically describes the cylindrical surface which comprises the heliostat
    def y(x): return -sqrt(r ** 2 - x ** 2) + abs(radius)

    # the computation of the points which discretize the heliostat surface
    hel_pts = array([[x, y(x)] for x in x_range]) + hc
    ####################################################################################################################

    return hel_pts


def design_flat_heliostat(center: array, width: float, nbr_pts: int):
    """
    This function returns the surface points of a flat shape heliostat.
    This set of points define the heliostat contour.

    Units should be in millimeters to be coherent with the classes, methods, and functions presented in this module.

    :param center: heliostat center point
    :param width: heliostat width
    :param nbr_pts: number of point to parametrize.

    :return: This function returns a list of points from the function of the heliostat.
    """

    # Auxiliary variables ##############################################
    # ensure that the center point is a [x, z] array point.
    hc = array([center[0], center[-1]])

    # ensure that positive values of width and radius are used
    w = abs(width)

    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    ###################################################################

    hel_pts = zeros(shape=(n_pts, 2))
    hel_pts[:, 0] = linspace(start=-0.5 * w, stop=0.5 * w, num=n_pts) + hc[0]

    return hel_pts


def uniform_centers(mirror_width: float, nbr_mirrors: int,
                    total_width: float = None, center_distance: float = None) -> array:
    """
    This functions calculates the center points of a uniform primary field.


    :param mirror_width: the width of the mirrors in the primary field.
    :param nbr_mirrors: number of heliostats.

    :param total_width: the total width of the primary field. The distance between the outer edges of the edge mirrors.
    :param center_distance: the distance between two consecutive center points.

    :return: This function returns a list with all mirrors center point in the x-y plane
    in the form of [xc, 0]. It considers a uniform shift between primaries.
    """

    # Old implementation ###############################################################################################
    # Changed in 09-Mar-2023
    # centers = zeros((number_mirrors, 2))
    # centers[:, 0] = linspace(start=0.5 * (total_width - mirror_width), stop=-0.5 * (total_width - mirror_width),
    #                          num=number_mirrors)
    #
    # return centers
    ####################################################################################################################

    # New implementation ###############################################################################################
    # The following routines are used to calculate the center points of the primary mirrors ########################
    # Logically, it considers a uniform distribution of mirrors along the primary field.
    # It also considers that the center points of the mirrors lie in the same line, i.e., the x-axis

    centers = zeros((nbr_mirrors, 2))
    # If the total_width of the primary field is given and not the distance between two consecutive centers
    if total_width is not None and center_distance is None:
        w = abs(total_width)
    # If the distance between the centers is given and not the total primary field width
    elif center_distance is not None and total_width is None:
        s = abs(center_distance)
        w = s * (nbr_mirrors - 1) + mirror_width
    # If both of them are given
    elif center_distance is not None and total_width is not None:
        if center_distance == (total_width - mirror_width) / (nbr_mirrors - 1):
            w = abs(total_width)
        else:
            raise ValueError('Function argument error: values do not make sense')
    # If none of the parameters is given
    else:
        raise ValueError('Function argument error: '
                         'Please add a "total_width" or "center_distance" values in millimeters')

    centers[:, 0] = linspace(start=0.5 * (w - mirror_width), stop=-0.5 * (w - mirror_width), num=nbr_mirrors)

    return centers
    ####################################################################################################################


def gap_angle_centers(total_width: float, mirror_width: float, aim: array, theta: float, d_min=0):
    """
    This function returns the center points of the primary field considering the gap angle criterion [1].

    :param total_width: the total width of the primary field, in mm.
    :param mirror_width: the width of the heliostats, in mm.
    :param aim: the aim point at the receiver.
    :param theta: the gap angle, in rad.
    :param d_min: the minimum distance between two neighbor heliostats, in mm.

    :return: It returns an array with the center points of the whole primary field.


    [1] Santos A V., Canavarro D, Collares-Pereira M.
    Renewable Energy 2021;163:1397–407. https://doi.org/10.1016/j.renene.2020.09.017.
    """

    ###############################
    # auxiliary variables
    sm = array([aim[0], aim[-1]])
    h = sm[1]
    ###############################

    #############################################################
    # Creating a list to append the center of the mirrors.
    # The fist mirror does not follow the gap-angle criterion.
    centers = [array([(total_width - mirror_width) / 2, 0])]
    #############################################################

    #################################################################################################################
    # Calculating the center points of the other mirrors (one side of the primary field).
    # The while condition refers to the fact that center of the last mirror must not be on the other side of the
    # symmetry axis, i.e., xc=0.
    while centers[-1][0] - (1.5 * mirror_width - d_min) > 0:

        x1 = centers[-1]
        beta_1 = ang(sm - x1, array([0, 1])) / 2.0

        beta_2 = fsolve(lambda x: tan(2 * beta_1) - (mirror_width / (2 * h)) * (cos(beta_1) +
                                                                                sin(beta_1) * tan(
                    2 * beta_1 + theta) + cos(x) + sin(x) * tan(2 * beta_1 + theta)) - tan(2 * x), array([beta_1]))[0]

        x2 = array([round(h * tan(2 * beta_2)), 0])

        if dst(x2, x1) - mirror_width > d_min:
            centers.append(x2)
        else:
            centers.append(x1 - array([mirror_width + d_min, 0]))
    ################################################################################################################

    ###############################################################################
    # Checking if it is possible to include a mirror right at the symmetry axis.
    if dst(centers[-1], array([0, 0])) > d_min + mirror_width:
        centers.append(array([0, 0]))
    ###############################################################################

    #########################################################################
    # calculating the other side of the primary field
    other_centers = deepcopy(centers)
    other_centers.reverse()
    b = []

    for i in range(len(other_centers)):
        if other_centers[i][0] > 0:
            b.append(array([- other_centers[i][0], 0]))
    #######################################################################

    return array(centers + b)


def rabl_curvature(center: array, aim: array, theta_d: float = 0.0) -> float:
    """
    A function to calculate the ideal curvature radius of a cylindrical heliostat as defined by Rabl [1, p.179]. A more
    detailed explanation can be found in the work done by Santos et al. [2].

    :param center: heliostat's center point.
    :param aim: aim point at the receiver.
    :param theta_d: design position, a transversal incidence angle (in degrees).

    :return: This function returns the ideal cylindrical curvature.

    [1] Rabl A. Active Solar Collectors and Their Applications. New York: Oxford University Press, 1985.
    [2] Santos et al., 2023. https://doi.org/10.1016/j.renene.2023.119380.

    It is important to highlight that calculations are for a ZX plane, where transversal incidence angles are positive
    on the right side of the z-axis direction (a positive rotation about the y-axis).
    See the comments for the module 'scopy.sunlight'.

    """

    # Angle from the horizontal which defines the direction of the incoming sunlight at the transversal plane.
    # a positive value angle lie is the first quadrant
    alpha = 0.5 * pi - deg2rad(theta_d)
    vi = V(alpha)

    # forcing the center and aim as 2D array points: [x, y]
    hc = array([center[0], center[-1]])
    f = array([aim[0], aim[-1]])

    # Check if the direction of the incoming sunlight is aligned with the mirror focusing vector since
    # the function 'ang(u, v)' used here sometimes calculates a wrong value when u || v.
    # Then, calculate the curvature absorber_radius.
    if cross(f - hc, vi).round(5) == 0:
        r = 2 * dst(hc, f)
    else:
        mi = 0.5 * ang(f - hc, vi)
        r = 2. * dst(hc, f) / cos(mi)

    return r


def boito_curvature(center: array, aim: array, lat: float) -> float:
    """
    Equation proposed by Boito and Grena (2017) for the optimum curvature absorber_radius of an LFR cylindrical primary.
    For a further understanding, one must read:
    Boito, P., Grena, R., 2017. https://doi.org/10.1016/j.solener.2017.07.079.

    :param center: heliostat's center point
    :param aim: aim point at the receiver
    :param lat: local latitude, in radians
    :return: the cylindrical curvature absorber_radius of an LFR primary mirror
    """

    hc = array([center[0], center[-1]])
    sm = array([aim[0], aim[-1]])

    a = 1.0628 + 0.0467 * power(lat, 2)
    b = 0.7448 + 0.1394 * power(lat, 2)

    v = sm - hc
    x, h = absolute(v)

    r = 2 * h * (a + b * power(x / h, 1.6))

    return r


def primaries_curvature_radius(centers: array, rec_aim: array, curvature_design='zenithal'):

    sm = array([rec_aim[0], rec_aim[-1]])

    radii = zeros(shape=centers.shape[0])

    for i, center in enumerate(centers):
        hc = array([center[0], center[-1]])

        if isinstance(curvature_design, (float, int)):
            r = rabl_curvature(center=hc,
                               aim=rec_aim,
                               theta_d=curvature_design)
        elif curvature_design == 'zenithal':
            r = rabl_curvature(center=hc,
                               aim=rec_aim,
                               theta_d=0.)
        elif curvature_design == 'SR':
            r = 2 * dst(hc, sm)
        elif curvature_design == 'flat':
            r = 0
        else:
            raise ValueError("Please, verify documentation for possible inputs of the 'curvature design' argument!")

        radii[i] = r

    return radii

########################################################################################################################
########################################################################################################################


# def non_shading_shift():
#     pass
#
#
# def nixon_primary_field(w: float, x1: float, ):
#     pass
