from numpy import array, sign, cross, absolute, pi, arange, zeros
from scipy.interpolate import interp1d, LinearNDInterpolator, InterpolatedUnivariateSpline

from niopy.geometric_transforms import ang, R, mid_point
from scopy.linear_fresnel.optical_method import intercept_factor, acceptance_analysis
from scopy.sunlight import SiteData, sun2lin


########################################################################################################################
# Auxiliary functions ##################################################################################################

def transform_vector(v: array):
    """
    This function transforms a vector-array (or a point-array) of the kind [x, y] into a [x, 0, z], and vice-versa.

    :param v: a point-array or a vector-array.

    :return: a point-array or a vector-array.
    """

    if v.shape[0] == 3 and v[1] == 0:
        return array([v[0], v[2]])
    elif v.shape[0] == 2:
        return array([v[0], 0, v[1]])
    else:
        raise Exception(f'The input must be an array of the kind [x, 0, z] or [x, y].')


def transform_heliostat(hel: array):
    """
    This function transforms an array of vector-arrays (or a point-arrays) of the kind [x, y] into a [x, 0, z],
    and vice-versa.

    :param hel: an array of point-arrays (or vector-arrays).

    :return: an array of point-arrays (or vector-arrays).
    """

    n = len(hel)
    if hel.shape[-1] == 3:
        vectors = zeros(shape=(n, 2))
    elif hel.shape[-1] == 2:
        vectors = zeros(shape=(n, 3))
    else:
        raise Exception(f'The input must be arrays of the kind [x, 0, z] or [x, y].')

    vectors[:] = [transform_vector(v) for v in hel]
    return vectors


def transform_field(heliostats: array):
    return array([transform_heliostat(hel) for hel in heliostats])


def rotate_points(points: array, center: array, tau: float, axis: array):
    translated_pts = points - center
    rm = R(alpha=tau, v=axis)

    rotated_pts = rm.dot(translated_pts.T).T + center
    return rotated_pts


def rotate_vectors(vectors: array, tau: float, axis: array):
    rm = R(alpha=tau, v=axis)
    rotated_vec = rm.dot(vectors.T).T

    return rotated_vec


def angular_position(center: array, rec_aim: array):
    """
    This function calculates the angular position of a primary mirror of an LFC concentrator, defined by the
    'center' point, regarding the 'aim' point at the receiver.
    It considers a [x,0,z] point and vector-arrays. The LFC transversal plane is the ZX plane.

    :param center: a point-array, in millimeters.
    :param rec_aim: a point-array, in millimeters.

    :return: an angle, in radians
    """

    Iz = array([0, 0, 1])

    sm = array([rec_aim[0], 0, rec_aim[-1]])
    hc = array([center[0], 0, center[-1]])

    aim_vector = sm - hc
    lamb = sign(cross(aim_vector, Iz)[1]) * ang(aim_vector, Iz)

    return lamb


def tracking_angle(center: array, rec_aim: array,
                   theta_t: float,
                   degrees=True):
    """
    :param center: the primary mirror center point.
    :param rec_aim: the aim-point at the receiver.

    :param theta_t: the transversal incidence angle.
    :param degrees: a boolean sign to indicate if the transversal angle is in degrees (True) or radians (False).

    :return: It returns the tracking angle of the primary mirror, in radians
    """

    # checking if the transversal incidence angle argument, i.e., 'theta_t' is in radians or degrees
    theta_t_rad = theta_t * pi / 180. if degrees else theta_t

    # the angular position of the primary mirror
    lamb = angular_position(center=center, rec_aim=rec_aim)

    # the tracking angle
    # this equation is based on the angular definitions presented in the documentation of this module.
    tau = 0.5 * (theta_t_rad - lamb)

    return tau

########################################################################################################################
########################################################################################################################

########################################################################################################################
# Optical analysis functions ###########################################################################################


def factorized_intercept_factor(field: array, normals: array,
                                centers: array, widths: array,
                                s1: array, s2: array, rec_aim: array, length: float,
                                cum_eff: array = None, end_losses=False, symmetric=False,
                                dt=5.):

    angle_step = abs(dt)

    tt0 = 0. if symmetric else -90.
    transversal_angles = arange(start=tt0, stop=90. + angle_step, step=angle_step)
    longitudinal_angles = arange(start=0., stop=90. + angle_step, step=angle_step)

    transversal_values = [intercept_factor(theta_t=theta, theta_l=0.,
                                           field=field, normals=normals,
                                           centers=centers, widths=widths,
                                           s1=s1, s2=s2, aim=rec_aim,
                                           length=length,
                                           cum_eff=cum_eff, end_losses=end_losses)

                          for theta in transversal_angles]

    longitudinal_values = [intercept_factor(theta_t=0., theta_l=theta,
                                            field=field, normals=normals,
                                            centers=centers, widths=widths,
                                            s1=s1, s2=s2, aim=rec_aim,
                                            length=length,
                                            cum_eff=cum_eff, end_losses=end_losses)

                           for theta in longitudinal_angles]

    transversal_data = zeros(shape=(transversal_angles.shape[0], 2))
    transversal_data.T[:] = transversal_angles, transversal_values

    longitudinal_data = zeros(shape=(longitudinal_angles.shape[0], 2))
    longitudinal_data.T[:] = longitudinal_angles, longitudinal_values

    return transversal_data, longitudinal_data


def biaxial_intercept_factor(field: array, normals: array,
                             centers: array, widths: array,
                             s1: array, s2: array,
                             length: float, rec_aim: array = None,
                             cum_eff: array = None, end_losses=False, symmetric=False,
                             dt=5.):

    aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

    step = abs(dt)
    tt0 = 0. if symmetric else -90.

    angles_list = array([[x, y] for x in arange(tt0, 90. + step, step) for y in arange(0., 90. + step, step)])

    gamma_values = [intercept_factor(theta_t=theta[0], theta_l=theta[1],
                                     field=field, normals=normals,
                                     centers=centers, widhts=widths,
                                     s1=s1, s2=s2, aim_pt=aim_pt,
                                     length=length,
                                     cum_eff=cum_eff, end_losses=end_losses)

                    for theta in angles_list]

    biaxial_data = zeros(shape=(angles_list.shape[0], 2))
    biaxial_data.T[:] = angles_list, gamma_values

    return biaxial_data


def acceptance_function(theta_t: float,
                        field: array, normals: array,
                        centers: array, widths: array,
                        s1: array, s2: array, rec_aim: array = None,
                        cum_eff: array = None, lv=0.6, dt=0.1) -> array:

    aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

    acceptance_data = acceptance_analysis(theta_t=theta_t,
                                          field=field, normals=normals,
                                          centers=centers, widths=widths,
                                          s1=s1, s2=s2, rec_aim=aim_pt,
                                          cum_eff=cum_eff, lvalue=lv, dt=dt)

    return acceptance_data


def acceptance_angle(acceptance_data: array, ref_value=0.9):

    acc_function = InterpolatedUnivariateSpline(acceptance_data.T[0], acceptance_data.T[1] - ref_value, k=3)
    roots = acc_function.roots()

    theta_a = 0.5 * abs(roots[0] - roots[1])

    return theta_a


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Performance calculations #############################################################################################

def annual_eta(transversal_data: array, longitudinal_data: array, site: SiteData, NS=True):
    """
    A function to compute the annual optical efficiency of a linear concentrator.

    :param transversal_data: Transversal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param longitudinal_data: Longitudinal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param site: a SiteData object with the TMY data of the location.
    :param NS: a sign to inform whether a NS (North-South) or EW (East-West) mounting is to consider.

    :return: It returns the annual optical efficiency (a value between 0 and 1)
    --------------------------------------------------------------------------------------------------------------------

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of South
    are negative and West of South are positive [3]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.

    [1] IEC (International Electrotechnical Commission). Solar thermal electric plants
    - Part 5-2: Systems and components - General requirements and test methods for large-size linear Fresnel collectors.
    Solar thermal electric plants, 2021.
    [2] Hertel JD, Martinez-Moll V, Pujol-Nadal R. Estimation of the influence of different incidence angle modifier
    models on the bi-axial factorization approach. Energy Conversion and Management 2015;106:249–59.
    https://doi.org/10.1016/j.enconman.2015.08.082.
    [3] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.
    """

    ####################################################################################################################
    # Creating optical efficiency (intercept factor) functions     #####################################################

    # Creating both transversal and longitudinal optical efficiencies functions for the calculations.
    # Ref. [1] suggest that a linear interpolation should be considered.
    gamma_t = interp1d(x=transversal_data.T[0], y=transversal_data.T[1], kind='linear')
    gamma_l = interp1d(x=longitudinal_data.T[0], y=longitudinal_data.T[1], kind='linear')
    # Taking the value of optical efficiency at normal incidence.
    gamma_0 = gamma_t(0)
    ####################################################################################################################

    # Check is the factorized data is of a symmetric linear Fresnel
    symmetric_lfr = True if transversal_data.T[0].min() == 0. else False

    ####################################################################################################################
    # Importing sun position and irradiation data from external files ##################################################
    tmy_data = site.tmy_data
    df = tmy_data[tmy_data['dni'] > 0]

    zenith = df['sun zenith'].values
    azimuth = df['sun azimuth'].values
    dni = df['dni'].values

    ####################################################################################################################
    # Calculating the linear incidence angles ##########################################################################

    # Here, it considers transversal and solar longitudinal incidence angles, as defined in Refs. [1,2].
    theta_t, _, theta_i = sun2lin(zenith=zenith, azimuth=azimuth, degrees=True, NS=NS, solar_longitudinal=True)
    ####################################################################################################################

    ####################################################################################################################
    # Energetic computations ###########################################################################################
    # It does not matter if transversal incidence angle is positive or negative if the concentrator is symmetric.
    # Nevertheless, the sign of the longitudinal angle does not matter at all.
    # Since vector operations were used, it only has few lines of code.
    if symmetric_lfr:
        energy_sum = (gamma_t(absolute(theta_t)) * gamma_l(absolute(theta_i)) / gamma_0).dot(dni)
    else:
        energy_sum = (gamma_t(theta_t) * gamma_l(absolute(theta_i)) / gamma_0).dot(dni)
    ####################################################################################################################

    # energetic sum is converted to annual optical efficiency by the division for the annual sum of DNI.
    # the annual sum of dni is the available energy to be collected.
    return energy_sum / dni.sum()


def biaxial_annual_eta(biaxial_data: array, site: SiteData, NS=True):

    x, y, z = biaxial_data.T
    gamma = LinearNDInterpolator(list(zip(x, y)), z)

    symmetric_lfr = True if biaxial_data.T[0].shape[0] == biaxial_data.T[1].shape[0] else False

    tmy_data = site.tmy_data
    df = tmy_data[tmy_data['dni'] > 0]

    zenith = df['sun zenith'].values
    azimuth = df['sun azimuth'].values
    dni = df['dni'].values

    theta_t, theta_l = sun2lin(zenith=zenith, azimuth=azimuth, degrees=True, NS=NS, solar_longitudinal=False)

    if symmetric_lfr:
        energy_sum = gamma(absolute(theta_t), absolute(theta_l)).dot(dni)
    else:
        energy_sum = gamma(theta_t, absolute(theta_l)).dot(dni)

    return energy_sum / dni.sum()


########################################################################################################################
########################################################################################################################
