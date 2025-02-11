"""
It is important to highlight that SolTrace consider dimensions in meters.
"""

from soltracepy import *

########################################################################################################################
# Optical properties functions #########################################################################################


def reflective_surface(name: str, rho: float, slope_error: float, spec_error: float):

    front = OpticInterface(name=name, reflectivity=round(rho, 4), transmissivity=0,
                           slp_error=round(slope_error, 4), spec_error=round(spec_error, 4), front=True)
    back = OpticInterface(name=name, reflectivity=0, transmissivity=0,
                          slp_error=round(slope_error, 4), spec_error=round(spec_error, 4), front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def secondary_surface(name: str, rho: float, slope_error: float, spec_error: float):

    front = OpticInterface(name=name, reflectivity=round(rho, 4), transmissivity=0,
                           slp_error=round(slope_error, 4), spec_error=round(spec_error, 4), front=True)
    back = OpticInterface(name=name, reflectivity=rho, transmissivity=0,
                          slp_error=round(slope_error, 4), spec_error=round(spec_error, 4), front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def flat_absorber_surface(name: str, alpha: float):
    front = OpticInterface(name=name, reflectivity=1 - alpha, transmissivity=0, front=True)
    back = OpticInterface(name=name, reflectivity=1, transmissivity=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def absorber_tube_surface(name: str, alpha: float):
    front = OpticInterface(name=name, reflectivity=round(1 - alpha, 4), transmissivity=0, front=True)
    back = OpticInterface(name=name, reflectivity=round(1 - alpha, 4), transmissivity=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def transmissive_surface(name: str, tau: float, nf: float, nb: float, slope_error=0.0, specular_error=0.0):

    front = OpticInterface(name=name, reflectivity=0, transmissivity=tau, real_refractive_index=nf, front=True,
                           spec_error=slope_error, slp_error=specular_error)
    back = OpticInterface(name=name, reflectivity=0, transmissivity=tau, real_refractive_index=nb, front=False,
                          spec_error=slope_error, slp_error=specular_error)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def cover_surfaces(tau: float, name='cover', refractive_index=1.52, slope_error=0.0, specular_error=0.0):

    tau_s = round(tau**0.5, 5)

    out_surf = transmissive_surface(name=f'{name}_outer_cover',
                                    tau=tau_s, nf=refractive_index, nb=1.0,
                                    specular_error=specular_error, slope_error=slope_error)

    inn_surf = transmissive_surface(name=f'{name}_inner_cover', tau=tau_s, nf=1.0, nb=refractive_index)

    # out_surf = transmissive_surface(name=f'{name}_outer_cover', tau=tau_s, nf=1.0, nb=refractive_index)
    # inn_surf = transmissive_surface(name=f'{name}_inner_cover', tau=tau_s, nf=refractive_index, nb=1.0)

    return out_surf, inn_surf

#######################################################################################################################
#######################################################################################################################

########################################################################################################################
# Geometry functions ###################################################################################################


def create_tube(center: array,
                radius: float,
                length: float,
                name: str,
                optic: OpticalSurface,
                reflect=True) -> list:
    """
    This function defines a cylinder over the Y-axis as a SolTrace Element.

    :param center: the center point of the tube.
    :param radius: the radius of the tube.
    :param length: the length of the tube (in the Y-axis).
    :param name:
    :param optic:
    :param reflect:
    :return:
    """

    # Setting the ecs_origin and aim-pt of the Element Coordinate System ########################
    # Origin of the ECS
    # For a tube element, the ecs_origin of the ECS is at the bottom of the tube (see SolTrace illustration)
    ecs_origin = array([center[0], 0, center[-1]]) - array([0, 0, radius])

    # Aim-pt of the ECS
    ecs_aim_pt = ecs_origin + array([0, 0, 2*radius])
    ############################################################################################

    # Setting the aperture and surface #########################################################
    # Aperture
    aperture = list([0] * 9)
    aperture[0], aperture[3] = 'l', length
    #####################################
    # Surface
    surface = list([0] * 9)
    surface[0], surface[1] = 't', 1 / radius
    #####################################
    ############################################################################################

    elem = Element(name=name, ecs_origin=ecs_origin, ecs_aim_pt=ecs_aim_pt, z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=reflect)

    return [elem]


def flat_element(name: str, ecs_origin: array, ecs_aim_pt: array, width: array, length: float,
                 optic: OpticalSurface, reflect=True, enable=True):

    aperture = list([0] * 9)
    aperture[0:3] = 'r', width, length

    surface = list([0] * 9)
    surface[0] = 'f'

    elem = Element(name=name, ecs_origin=ecs_origin, ecs_aim_pt=ecs_aim_pt, z_rot=0,
                   optic=optic, aperture=aperture, surface=surface,
                   reflect=reflect, enable=enable)

    return elem


def create_primary_mirror(width: float,
                          center: array,
                          radius: float,
                          aim: array,
                          length: float,
                          optic: OpticalSurface,
                          name: str) -> Element:
    """
    This function defines a linear Fresnel primary mirror as a Soltrace Element object (see soltracepy.Element).
    It considers flat and cylindrical shape mirrors.

    :param radius: the curvature radius of the mirror
    :param center: the center point of the mirror, an [x,0,z] array.
    :param width: the width of the mirror
    :param aim: The aiming point at the receiver used as reference for the tracking, a point-array.
    :param length: The length of the heliostat.
    :param optic: The optical property to associate, a SolTrace OpticalSurface (soltracepy.OpticalSurface).
    :param name: A variable_name to be inserted in the 'comment' argument of the Element.

    :return: It returns an Element object.
    """

    # The ecs_origin and ecs_aim_pt of the Element Coordinate System #####################
    # Remember that SolTrace uses the units in meters
    ecs_origin = array([center[0], 0, center[-1]])
    ecs_aim = array([aim[0], 0, aim[-1]])
    # A vector from the ecs_origin to the ecs_aim_pt defines the z-axis of the ECS
    ######################################################################################

    # Creating auxiliary variables to implement #############################
    L = length
    aperture = list([0] * 9)
    surface = list([0] * 9)
    #########################################################################

    # New implementation -- from 08-Mar-2023 ############################################
    if radius == 0:
        # Defining the aperture of the mirror #########
        aperture[0:3] = 'r', width, L
        ###########################################################

        # Defining the surface type ########
        surface[0] = 'f'
        ####################################
    else:
        # Current implementation (as of 12-May-23) ##################################################################
        # The old implementation was many times slower in the SolTrace 2012 version (and I do not know why).
        # This implementation approximates the cylindrical surface to the central section of a symmetrical parabola
        # in which the relation between parabola's focal length and circle curvature radius is R = 2f.
        # This approximation would add negligible errors -- see Santos et al. [1]

        # Defining the aperture
        aperture[0:3] = 'r', width, L

        # Defining the surface
        rc = radius
        # The 'c' factor is the parabola's gradient, as defined in SolTrace.
        c = 1 / rc
        surface[0:2] = 'p', c
        ##############################################################################################################

        # Old implementation ############################################################################
        # It considers a spherical surface, that when used with an aperture
        # defined by a 'Single Axis Curvature Section' emulates a circular surface
        # (see SolTrace 2012.7.9 Help files).

        # Defining the aperture of the cylindrical mirror #########
        # x1 = -0.5 * hel.width / 1000
        # x2 = 0.5 * hel.width / 1000
        # aperture[0:4] = 'l', x1, x2, L

        # Defining the surface type ############################################
        # Curvature absorber_radius and converts it to meters.
        # rc = hel.radius / 1000
        # c = 1 / rc
        # surface[0:2] = 's', c
        ##############################################################################################################

    elem = Element(name=name, ecs_origin=ecs_origin, ecs_aim_pt=ecs_aim, z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=True)

    return elem


def create_parabolic_trough(width: float,
                            vertex: array,
                            focal_length: float,
                            length: float,
                            optic: OpticalSurface,
                            name: str,
                            optical_axis_direction=array([0, 0, 1])):

    # The ecs_origin and ecs_aim_pt of the Element Coordinate System #####################
    # Remember that SolTrace uses the units in meters
    ecs_origin = array([vertex[0], 0, vertex[-1]])
    ecs_aim = ecs_origin + focal_length * optical_axis_direction
    # A vector from the ecs_origin to the ecs_aim_pt defines the z-axis of the ECS
    ######################################################################################

    aperture = list([0] * 9)
    surface = list([0] * 9)

    aperture[0:3] = 'r', width, length

    # Defining the surface

    # The 'c' factor is the parabola's gradient, as defined in SolTrace.
    c = 1 / (2 * focal_length)
    surface[0:2] = 'p', c

    elem = Element(name=name, ecs_origin=ecs_origin, ecs_aim_pt=ecs_aim, z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=True)

    return elem

########################################################################################################################
########################################################################################################################


def create_csi_file(curve_pts: array, knots_derivatives: array, file_path: Path, file_name: str) -> Path:
    """
    This function creates the file used by SolTrace to construct an element by a rotationally symmetric cubic spline.
    The arguments of this function must be in meters, as SolTraces work with such unit.

    :param curve_pts: The curve points as [x, y] point-arrays.
    :param knots_derivatives: The first derivatives at the first and last points of the curve.
    :param file_path: The path in which to create the spline file.
    :param file_name: The variable_name of the spline file.

    :return: The full path of the spline file.
    """

    # Curve points and knots derivatives ######
    x_values, y_values = curve_pts.T
    df_1, df_n = knots_derivatives
    ###########################################

    # Creating the rotationally symmetric cubic spline file path #####
    # A 'csi' extension file for SolTrace to correctly read it
    full_file_path = Path(file_path, f"{file_name}.csi")
    ##################################################################

    # Creating file and writing the data into it ################################################################
    file = open(full_file_path, 'w')
    # The first line must contain the number of points which defines the surface
    file.write(f"{x_values.shape[0]}\n")
    for x, y in zip(x_values, y_values):
        # write in the file the point coordinates values in meters
        file.write(f"{x}\t{y}\n")

    # the last line should contain the first derivatives at both edge knots.
    file.write(f"{df_1}\t{df_n}")  # writes the first derivatives at both edges knots
    file.close()  # closes the file
    ############################################################################################################

    return full_file_path
