from pathlib import Path

from numpy import array, deg2rad, arccos, pi, sin, zeros, cos, linspace
from soltracepy import Element, OpticalSurface, Sun, Optics, Stage, Geometry, Trace, ElementStats, soltrace_script, \
    run_soltrace, read_element_stats
from soltracepy.auxiliary import reflective_surface, absorber_tube_surface, cover_surfaces, create_tube, \
    create_primary_mirror


########################################################################################################################
# functions for this program ###########################################################################################


def mid_point(p: array, q: array) -> array:
    """
    This function (...).

    :param p: a point in space
    :param q: a point in space

    :return: the mid-point between p and q
    """

    return (p + q) * 0.5


def R(alpha: float, v: array = None) -> array:
    """
    This function (...).
    R(alpha) is a rotation matrix of an angle alpha. R(alpha,v)
    is a rotation matrix of an angle alpha around axis v.
    Such rotations are pivoted from ecs_origin [0,0] or [0,0,0].

    :param alpha:
    :param v:

    :return:
    """

    if v is None:
        rm = array(
            [
                [cos(alpha), -sin(alpha)],
                [sin(alpha), cos(alpha)],
            ]
        )
    else:
        if v.shape[0] != 3:
            raise Exception(f'Wrong dimension of v. Found dimension {v.shape[0]} where should be 3.')
        vn = nrm(v)
        rm = array(
            [
                [
                    cos(alpha) + vn[0] ** 2 * (1 - cos(alpha)),
                    vn[0] * vn[1] * (1 - cos(alpha)) - vn[2] * sin(alpha),
                    vn[0] * vn[2] * (1 - cos(alpha)) + vn[1] * sin(alpha),
                ],
                [
                    vn[1] * vn[0] * (1 - cos(alpha)) + vn[2] * sin(alpha),
                    cos(alpha) + vn[1] ** 2 * (1 - cos(alpha)),
                    vn[1] * vn[2] * (1 - cos(alpha)) - vn[0] * sin(alpha),
                ],
                [
                    vn[2] * vn[0] * (1 - cos(alpha)) - vn[1] * sin(alpha),
                    vn[2] * vn[1] * (1 - cos(alpha)) + vn[0] * sin(alpha),
                    cos(alpha) + vn[2] ** 2 * (1 - cos(alpha)),
                ],
            ]
        )
    return rm


def rotate2D(points: array, center: array, tau: float):

    rm = R(alpha=tau)
    translated_pts = points - center
    rotated_pts = rm.dot(translated_pts.T).T + center

    return rotated_pts


def Sy(v: array) -> array:
    """
    This function (...)

    :param v: a 2D (x-y) point, or vector.

    :return: returns the symmetrical to v relative to the y-axis
    """

    m = array([[-1, 0], [0, 1]])

    return m.dot(v)


def oommen_cpc(tube_center: array,
               tube_radius: float,
               outer_glass_radius: float,
               theta_a: float,
               number_pts=150,
               upwards=True) -> array:
    """
    This function implements procedure proposed by Oomen and Jayaraman [1] to design a compound parabolic concentrator
    (CPC) to an absorber tube encapsulated by a cover -- as also detailed by Abbas et al. [2].

    The gap problem [3] is here solved by the virtual receiver design solution, as proposed by Winston [4]. Oommen and
    Jayaraman [1] denominate it as the extended receiver design.

    This function returns an array of [x,y] points that define the cpc optic contour.

    :param tube_center: the center point of the absorber tube, in millimeters
    :param tube_radius: the radius of the absorber tube, in millimeters.
    :param outer_glass_radius: the radius of the cover tube, in millimeters.
    :param theta_a: the half-acceptance angle of the cpc optic, in degrees.
    :param number_pts: number of contour points to return.
    :param upwards:

    :return: the contour points of the cpc optic.

    [1] Oommen, R., Jayaraman, S., 2001. https://doi.org/10.1016/S0196-8904(00)00113-8.
    [2] Abbas et al., 2018. https://doi.org/10.1016/j.apenergy.2018.09.224.
    [3] Rabl, A., 1985. Active Solar Collectors and Their Applications. Oxford University Press, New York.
    [4] Winston, R., 1978. Ideal flux concentrators with reflector gaps. https://doi.org/10.1364/AO.17.001668.
    """

    # storing input data in auxiliary variables ###
    r1 = tube_radius
    r2 = outer_glass_radius
    theta_a_rad = deg2rad(theta_a)
    ###############################################

    # defining auxiliary variables #######################
    r_ratio = r2 / r1
    beta = (r_ratio**2 - 1)**0.5 - arccos(1/r_ratio)
    ######################################################

    # range of the parameter used for the parametric equation #######################
    # beginning of the first segment of the optic
    theta_0 = arccos(r1/r2)
    # end of the first segment / start of the second
    theta_1 = theta_a_rad + 0.5*pi
    # end of the second segment
    theta_2 = 1.5*pi - theta_a_rad
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

    if upwards:
        right_side = curve_pts
        left_side = array([Sy(x) for x in curve_pts])
    else:
        gap = outer_glass_radius - tube_radius
        cusp = array([0, -gap])

        rotated_pts = rotate2D(points=curve_pts, center=cusp, tau=pi) + array([0, 2 * gap])
        left_side = rotated_pts
        right_side = array([Sy(x) for x in rotated_pts])

    full_curve_pts = array(left_side[::-1].tolist() + right_side.tolist())

    return full_curve_pts + tube_center


def mag(v: array) -> float:
    """
    This function returns the magnitude (length) of vector 'v'.

    :param v: a vector-array

    :return: A float.
    """
    return v.dot(v) ** 0.5


def nrm(v: array) -> array:
    """
    This function returns the unit vector which points to the same direction as vector 'v'.

    :param v: A vector-array

    :return: A vector-array
    """
    return v / mag(v)


def dst(p: array, q: array) -> float:
    """
    This functions returns the Euclidian distance between point-arrays 'p' and 'q'.

    :param p: A point-array
    :param q: A point-array

    :return: A float.
    """

    return mag(p - q)


def primary_ecs_aim(center: array,
                    rec_aim: array,
                    sun_dir: array) -> array:
    """
    This method refers to a SolTrace implementation of the PrimaryMirror object as an SolTrace Element.
    It calculates the aim point of the Element Coordinate System (ECS) as a [x, 0, z] point-array. Since mirrors
    rotate as the sun moves, this aim point changes with the sun direction vector.

    A SolTrace Element has an Element Coordinate System (ECS) attached to it. The z-axis direction of the ECS
    is defined by the vector from the ECS origin (i.e., the heliostat center point) and the ECS aim-point, so that:
    z = ecs_aim_pt - ecs_origin. Of course, these points and vectors are defined in the Stage Coordinate System.
    See SolTrace documentation for more details.

    :param center:
    :param rec_aim: The aim point at the receiver used in the tracking procedure of the heliostat.
    :param sun_dir: A 3D vector which represents the Sun vector.

    :return: A [x, 0, z] point-array.
    """

    # Ensure the receiver aim point as a [x, 0, z] point-array
    sm = array([rec_aim[0], 0, rec_aim[-1]])
    hc = array([center[0], 0, center[-1]])
    ############################################################

    # Calculating the focusing vector
    vf = nrm(sm - hc)
    ###############################################################################

    # Computing the projection of the sun vector in the zx-plane
    # Only this projection is used in the tracking calculations.
    st = nrm(array([sun_dir[0], 0, sun_dir[-1]]))
    ################################################################

    # The normal vector at mirror center point
    n = nrm(vf + st)
    #####################################################

    # The Element Coordinate System (ECS) aim-point
    ecs_aim_pt = hc + 2 * n
    ###########################################################

    return ecs_aim_pt


def create_plane_curve(curve_pts: array,
                       length: float, optic: OpticalSurface,
                       name: str) -> list:

    elements_list = []

    for i in range(len(curve_pts) - 1):
        # Selecting the two consecutive points in the PlaneCurve #######
        # that will define the flat segment
        pt_a = curve_pts[i]
        pt_b = curve_pts[i + 1]
        ################################################################

        # Calculates the ecs_origin and the aim-point of the flat element #####
        origin = mid_point(p=pt_a, q=pt_b)
        aim_pt = origin + R(-pi / 2).dot(pt_b - pt_a)
        ###################################################################

        # Converts to meters and to a [x, 0, z] arrays #########
        origin = array([origin[0], 0, origin[-1]])
        aim_pt = array([aim_pt[0], 0, aim_pt[-1]])
        ########################################################

        # Calculates the width of the flat Element ####
        width = dst(p=pt_a, q=pt_b)
        ###############################################

        # Defining the aperture of the Element ########
        aperture = list([0] * 9)
        aperture[0:3] = 'r', width, length
        ###############################################

        # Defining the surface of the Element #########
        surface = list([0] * 9)
        surface[0] = 'f'
        ###############################################

        # Appending the current Element to the elements list ##################
        elements_list += [Element(name=f"{name}_{i + 1}",
                                  ecs_origin=origin, ecs_aim_pt=aim_pt, z_rot=0,
                                  aperture=aperture, surface=surface,
                                  optic=optic,
                                  reflect=True, enable=True)]
        #######################################################################

    return elements_list


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Concentrator geometric data  #########################################################################################

# units in meters
# main data
receiver_height = 7.200
total_width = 16.560
nbr_mirrors = 16
mirror_width = 0.750

concentrator_length = 60.000

absorber_tube_center = array([0, receiver_height])
absorber_tube_radius = 35.e-3
outer_cover_radius = 62.5e-3
inner_cover_radius = outer_cover_radius - 3.


# CPC secondary optic contour
cpc_points = oommen_cpc(tube_center=absorber_tube_center,
                        tube_radius=absorber_tube_radius,
                        outer_glass_radius=outer_cover_radius,
                        upwards=False,
                        theta_a=45,
                        number_pts=150)

# the aim-point at the receiver for the tracking
# is the CPC aperture mid-point
receiver_aim = mid_point(p=cpc_points[0], q=cpc_points[-1])


# calculating the center points of the primary mirrors
xPos = linspace(start=(total_width - mirror_width)/2,
                stop=-(total_width - mirror_width)/2,
                num=nbr_mirrors)
centers = array([[x, 0] for x in xPos])

# curvature radius of the primaries by a specific reference design
radii = [2*dst(p=hc, q=receiver_aim) for hc in centers]

########################################################################################################################
########################################################################################################################

########################################################################################################################
# SolTrace objects and definitions #####################################################################################

# The OPTICS ##################################################################
primaries_reflectivity = reflective_surface(name='mirrors_refl',
                                            rho=0.93,
                                            slope_error=2,
                                            spec_error=3)

secondary_reflectivity = reflective_surface(name='secondary_refl',
                                            rho=0.88,
                                            slope_error=2,
                                            spec_error=3)

tube_absorptivity = absorber_tube_surface(name='tube_abs',
                                          alpha=0.96)

outer_cover_trans, inner_cover_trans = cover_surfaces(name='glass_cover',
                                                      tau=0.96,
                                                      refractive_index=1.55)
optical_properties = [primaries_reflectivity,
                      secondary_reflectivity,
                      tube_absorptivity,
                      outer_cover_trans,
                      inner_cover_trans]

# the Optics box object
optics = Optics(properties=optical_properties)

#######################################################################################

# The SUN #############################################
sun = Sun(sun_dir=array([0, 0, 1]),
          profile='g',
          size=2.8)
###############################################################

##############################################################################
# The GEOMETRY ###############################################################

# primary mirrors Element Coordinate System aim-point
primaries_aim = [primary_ecs_aim(center=hc,
                                 rec_aim=receiver_aim,
                                 sun_dir=sun.vector)

                 for hc in centers]

# primary mirrors as SolTrace Elements
elements = [create_primary_mirror(width=mirror_width, center=hc, radius=rr,
                                  length=concentrator_length,
                                  aim=a_pt,
                                  optic=primaries_reflectivity,
                                  name=f'heliostat_{i + 1}')

            for i, (hc, rr, a_pt) in enumerate(zip(centers, radii, primaries_aim))]

# secondary optic as SolTrace Elements
elements += create_plane_curve(curve_pts=cpc_points,
                               length=concentrator_length,
                               optic=secondary_reflectivity,
                               name='secondary_optic')

# Outer glass cover as an Element
elements += create_tube(center=absorber_tube_center,
                        radius=outer_cover_radius,
                        name='outer_glass_cover',
                        optic=outer_cover_trans,
                        length=concentrator_length,
                        reflect=False)

# Inner glass cover as an Element
elements += create_tube(center=absorber_tube_center,
                        radius=inner_cover_radius,
                        name='inner_glass_cover',
                        optic=inner_cover_trans,
                        length=concentrator_length,
                        reflect=False)

# Absorber tube as an Element
elements += create_tube(center=absorber_tube_center,
                        radius=absorber_tube_radius,
                        name='absorber_tube',
                        optic=tube_absorptivity,
                        length=concentrator_length, reflect=True)

# creating a Stage to insert all elements defined before
stage = Stage(elements=elements,
              name='linear_fresnel')

# the Geometry box
geometry = Geometry(stages=[stage])
##############################################################################

# The INTERSECTIONS #############################################################

# the ElementStats object to export intersection data of the absorber tube
absorber_stats = ElementStats(stats_name='absorber_flux',
                              stage_index=0,
                              element_index=len(elements) - 1,
                              x_bins=50, y_bins=15, final_rays=True)

#################################################################################

# The TRACE #############################################################

trace = Trace(rays=1.e5, cpus=8,
              sunshape=True, optical_errors=True,
              point_focus=False, simulate=True)
#########################################################################

# The SCRIPT ###############################################################

# creating the script file
script_file = soltrace_script(file_path=Path.cwd(),
                              file_name='optic',
                              sun=sun,
                              optics=optics,
                              geometry=geometry,
                              stats=absorber_stats,
                              trace=trace, version=2012)

# running the script
run_soltrace(lk_file_full_path=script_file)
#
# reading stats file
absorber_stats = read_element_stats(absorber_stats.file_full_path)
