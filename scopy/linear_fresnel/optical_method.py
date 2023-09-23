# -*- coding: utf-8 -*-
"""
Created on Apr-8, 2023, 15:47:43
New version: Jul-21, 2023, 17:40:00
@author: André Santos (andrevitoras@gmail.com / avas@uevora.pt)
"""

from typing import Tuple
from numba import njit, float64, int64, bool_
from numpy import cos, ascontiguousarray, arccos, array, sin, pi, identity, arctan, tan, cross, sign, round, zeros, \
    interp, sqrt

"""
This module holds the functions to computes the optic-geometric performance of linear Fresnel concentrators.

It is based on the method proposed by Santos et al. [1]. It considers a 3D model of the Linear Fresnel Collector (LFC),
where the ZX plane is the transversal plane of the concentrator, and the y-axis is the longitudinal direction.

Here point and vectors are represented as [x, y, z] arrays, i.e., arrays whose shape is (3,). Furthermore, due to the
linear symmetry of the concentrators, primary mirrors and receivers are then defined in the ZX plane.
Thus, point-arrays which represents the surface of the mirrors are [x, 0, z]-like, as also other points and
vector-arrays which defines the concentrator geometry.

In this sense, a primary mirror -- a heliostat -- is defined by an array of [x, 0, z] point-arrays. In this sense:
heliostat = array([[x1, 0, z1], ..., [xn, 0, zn]]).

Then, a primary field is defined as an array of heliostats. In this sense:
primary_field = array(hel_1, ...., hel_n]).

Therefore, point and vector-arrays are represented by the datatype float64[:], which represents a 1D array. Heliostats
are represented by the datatype float64[:, :], which represents a 2D arrays - one for the set of points, and other for
the points themselves. Finally, a primary field is represented by the datatype float64[:, :, :], as expected.

OBS: This module only imports from external packages such as numpy, numba, scipy... It does not import anything from
our internal modules. In this sense, it is self-defined module.

------------
References:

[1] Santos, A. V., Canavarro, D., Horta, P., Collares-Pereira, M., 2021.
    An analytical method for the optical analysis of Linear Fresnel Reflectors with a flat receiver.
    Solar Energy 227, 203–216. https://doi.org/10.1016/j.solener.2021.08.085.
[2] Chaves, J., 2016. Introduction to Nonimaging Optics. CRC Press, New York, 2nd Edition.

"""


@njit
def mag(v: float64) -> float64:
    """
    This function calculates the magnitude of a vector

    :param v: A vector-array
    :return: The module of the vector
    """

    vv = ascontiguousarray(v)

    return vv.dot(vv) ** 0.5


@njit
def nrm(v: float64[:]) -> float64[:]:
    """
    This functions returns the norm of a vector 'v', i.e., the unit vector which represents the direction given by 'v'.

    :param v: A vector-array
    :return: The norm of 'v'
    """

    return v / mag(v)


@njit
def ang(v: float64[:], u: float64[:]) -> float64:
    """
    This function returns the dihedral angle between two vectors.
    Thus, the returned angle range from 0 to pi.

    :param v: a vector-array.
    :param u: a vector-array.

    :return: an angle, in radians.
    """

    vv = ascontiguousarray(v)
    uu = ascontiguousarray(u)

    arg = round(vv.dot(uu) / (mag(vv) * mag(uu)), 12)

    return arccos(arg)


@njit
def islp(p: float64[:], v: float64[:], q: float64[:], n: float64[:]) -> float64[:]:
    """
    This function calculates (and returns) the interception point between a straight line and a plane.
    The straight line is defined by a point 'p' and a direction given by vector 'v'.
    The plane is defined by point 'q' and a normal vector 'n'.

    :param p: The point which defines the straight line, a [x, y, z] point.
    :param v: The vector which defines the straight line, a [x, y, z] vector.
    :param q: The point which defines the plane, a [x, y, z] point.
    :param n: The normal vector which defines the plane, a [x, y, z] vector.

    :return: It returns the intersection point, a [x, y, z] point.
    """

    vv = nrm(v)
    nn = nrm(n)

    return p + vv * (p - q).dot(nn) / vv.dot(nn)


@njit
def R(alpha: float64, v: float64[:]) -> float64[:, :]:
    """
    This function calculates (and returns) a 3D rotation matrix, considering a rotation of angle 'alpha' around the
    direction given by 'v', pivoted from the origin [0,0,0].

    :param alpha: The angle of rotation, in radians.
    :param v: The direction of the rotation, a [x, y, z] vector.

    :return: A 3x3 rotation matrix.
    """

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


@njit
def rotate_points(points: float64[:, :], center: float64[:], tau: float64, axis: float64[:]) -> float64[:, :]:
    translated_pts = points - center
    rm = R(alpha=tau, v=axis)

    rotated_pts = rm.dot(translated_pts.T).T + center
    return rotated_pts


@njit
def rotate_vectors(vectors: float64[:, :], tau: float64, axis: float64[:]) -> float64[:, :]:
    rm = R(alpha=tau, v=axis)
    rotated_vec = rm.dot(vectors.T).T

    return rotated_vec


@njit
def sun_direction(theta_t: float64, theta_l: float64) -> float64[:]:
    """
    This function is a copy of sunlight.sun_direction. Further documentation can be found there.
    Modifications were only made to improve performance and avoid warnings with Numba '@njit' decorator.

    """
    theta_t_rad = theta_t * pi / 180.
    theta_l_rad = theta_l * pi / 180.

    Ix, Iy, Iz = identity(3)

    r1 = R(theta_l_rad, Ix)
    projected_angle = arctan(tan(theta_t_rad) * cos(theta_l_rad))
    r2 = R(projected_angle, Iy)

    iz = ascontiguousarray(Iz)
    rr1 = ascontiguousarray(r1)
    rr2 = ascontiguousarray(r2)

    return rr1.dot(rr2.dot(iz))


@njit
def angular_position(center: float64[:], aim: float64[:]) -> float64:
    """
    This function calculates the angular position of a primary mirror of an LFC concentrator, defined by the
    'center' point, regarding the 'aim' point at the receiver. It considers a [x, 0, z] point and vector.
    The mathematical model of this definition is presented in Eq.(8) of Ref.[1].

    In this sense, a mirror whose center lies in the left-side of the x-axis has a negative angular position, and a
    mirror whose center lies in the right-side of the x-axis has a positive angular position.

    :param center: a point, in millimeters.
    :param aim: a point, in millimeters.

    :return: an angle, in radians
    """

    Iz = array([0, 0, 1.])

    aim_vector = aim - center
    lamb = sign(cross(aim_vector, Iz)[1]) * ang(aim_vector, Iz)

    return lamb


@njit
def rotated_field_data(theta_t_rad: float64,
                       field: float64[:, :, :],
                       normals: float64[:, :, :],
                       centers: float64[:, :],
                       sm: float64[:]) -> Tuple[float64[:], float64[:, :, :], float64[:, :, :]]:
    """
    This function calculates the tracking angles and the correspondent contour points and normal vectors to the mirrors
    surface.

    :param field: the array of [x, 0, z] points that represents the field in a horizontal position.
    :param normals: the array of [x, 0, z] vectors that represents the normals to the mirrors in a horizontal position.
    :param centers: the array of points which represents the center points of the primary mirrors.
    :param sm: the aim-point at the receiver.
    :param theta_t_rad: the transversal incidence angle, in radians.

    :return: a tuple of three arrays: (tau, rot_field, rot_normals).
    """

    Iy = array([0., 1., 0.])
    lamb = zeros(len(centers))
    tau = zeros(len(centers))
    rotated_field = zeros(shape=field.shape)
    rotated_normals = zeros(shape=field.shape)

    for i, hc in enumerate(centers):
        lamb[i] = angular_position(center=hc, aim=sm)
        tau[i] = 0.5 * (theta_t_rad - lamb[i])

        rotated_field[i] = rotate_points(points=field[i], center=hc, tau=tau[i], axis=Iy)
        rotated_normals[i] = rotate_vectors(vectors=normals[i], tau=tau[i], axis=Iy)

    return tau, rotated_field, rotated_normals


@njit
def is_central_heliostat(center: float64[:], rec_aim: float64[:]) -> bool_:
    """
    This function calculates whether the 'center' is of a central heliostats, i.e., the one located right below the
    aim-point at the receiver.

    :param center: the center point of the heliostat
    :param rec_aim: the aim-point at the receiver

    :return: A boolean sign to indicate whether this is a central heliostat or not
    """

    vf = rec_aim - center
    is_central = round(vf[0], 4) == 0.
    return is_central


@njit
def reft(i: float64[:], n: float64[:]) -> float64[:]:
    """
    This function calculates the direction of a reflected ray as [x, y, z] vector-array.
    It considers that the incident ray point towards the Sun (i.e., up), and the the reflected ray will also point up.

    :param i: the incident ray direction, a [x, y, z] vector.
    :param n: the normal to the surface direction, a [x, y, z] vector.

    :return: A [x, y, z] vector.
    """

    ii = ascontiguousarray(i)
    nn = ascontiguousarray(n)
    return 2 * ii.dot(nn) * nn - ii


@njit
def define_neighbors(theta_t_rad: float64, i: int64, centers: float64[:, :],
                     rotated_field: float64[:, :, :]) -> Tuple[float64[:], float64[:]]:
    """
    This functions defines the neighbors of a particular mirror in the primary field, and then calculates edge-points
    of the neighbors that relate to shading and blocking losses.
    The mirror being analyzed is defined by the index 'i' in the field.

    :param theta_t_rad: the transversal incidence angle, in radians.
    :param i: the index of the heliostat being considered.
    :param centers: an array with the [x, 0, z] point of the center of the heliostats.

    :param rotated_field: an array of heliostats. Heliostats are defined as arrays of [x, 0, z] point.

    :return: A tuple of two [x, y, z] point-arrays: (blocking_edge, shading_edge)

    """

    # number of heliostats in the primary field
    n_hel = len(centers)

    # Select mirror's neighbor to account for blocking ###########
    if centers[i][0] > 0:
        neighbor_b = rotated_field[i + 1]
    else:
        neighbor_b = rotated_field[i - 1]

    # Select the edge point on blocking neighbor
    # It chooses the highest edge-point
    if neighbor_b[0][2] > neighbor_b[-1][2]:
        edge_pt_b = neighbor_b[0]
    else:
        edge_pt_b = neighbor_b[-1]
    ##############################################################

    # Select mirror's neighbor to account for shading #################################################

    # For a normal incidence, the neighbor mirror causing blocking and shading are the same.
    # First and last heliostats are never shaded for theta_t > 0 and theta_t < 0, respectively.
    # This computation (zero shading losses) is accounted in a different function.
    # But, here, it is needed to select one mirror for the algorithm does not get an out of index error
    if theta_t_rad == 0 or (i == 0 and theta_t_rad > 0) or (i == n_hel - 1 and theta_t_rad < 0):
        neighbor_s = neighbor_b
    # For a non-normal incidence, the neighbor mirror causing shading depends on the incidence
    # Remember that centers[0] has the most positive x-axis value.
    else:
        neighbor_s = rotated_field[i - int(1 * sign(theta_t_rad))]

    # Select the edge point on shading neighbor
    # It chooses the highest edge-point
    if neighbor_s[0][2] > neighbor_s[-1][2]:
        edge_pt_s = neighbor_s[0]
    else:
        edge_pt_s = neighbor_s[-1]
    ###################################################################################################

    return edge_pt_b, edge_pt_s


@njit
def define_neighbor_vectors(p: float64[:],
                            edge_pt_b: float64[:], edge_pt_s: float64[:],
                            ni: float64[:], nr: float64[:]) -> Tuple[float64[:], float64[:]]:
    """
    This function calculates the neighbor vectors. These vectors are used to calculate the shading and blocking losses
    in a point of a primary mirrors due to its neighbors.

    :param p: A point-array in the surface of a primary mirror.
    :param edge_pt_b: The edge of the neighbor which cause blocking losses.
    :param edge_pt_s: The edge of the neighbor which cause shading losses.
    :param ni: The normal vector which defines the incidence plane.
    :param nr: The normal vector which defines the reflection plane.

    :return: A tuple of two [x, y, z] vector-arrays: (shading, blocking)
    """

    # Calculating the neighbor vector for blocking analysis
    ymb = - (edge_pt_b - p).dot(nr) / nr[1]
    vmb = array([edge_pt_b[0], ymb, edge_pt_b[2]]) - p

    # Calculation of the neighbor vector for shading analysis
    yms = - (edge_pt_s - p).dot(ni) / ni[1]
    vms = array([edge_pt_s[0], yms, edge_pt_s[2]]) - p

    return vms, vmb


@njit
def flat_receiver_limiting_vectors(p: float64[:],
                                   sal: float64[:], sar: float64[:],
                                   nr: float64[:]) -> Tuple[float64[:], float64[:]]:
    """
    This function calculates (and returns) the limiting vectors of a point 'p' in which a particular mirrors sees the
    flat receiver. The limiting vector are defined from the point the primaries surface to the edges of the receiver.

    :param p: A [x, 0, z] point-array in the surface of a primary mirror.
    :param sal: The left edge point of the receiver.
    :param sar: The right edge point of the receiver.
    :param nr: The normal vector which defines the reflection plane.

    :return: A tuple of two [x, y, z] vector-arrays.
    """

    yll = - (sal - p).dot(nr) / nr[1]
    ylr = - (sar - p).dot(nr) / nr[1]

    vll = array([sal[0], yll, sal[2]]) - p
    vlr = array([sar[0], ylr, sar[2]]) - p

    return vll, vlr


@njit
def flat_receiver_shading_vectors(p: float64[:], ni: float64[:],
                                  sal: float64[:], sar: float64[:]) -> Tuple[float64[:], float64[:]]:
    """
    This function calculates the receiver shading vectors. These are the vectors which are used to calculate the
    losses due to receiver shading on a [x, 0, z] point-array of a primary mirror.

    :param p: A [x, 0, z] point-array in the surface of a primary mirror.
    :param ni: The normal vector which defines the incidence plane.
    :param sal: left-edge point of the receiver.
    :param sar: right-edge point of the receiver.

    :return: A tuple of two [x, y, z] vector-arrays: (left, right)
    """

    # vectors from the point 'p' in the mirror surface to the (left and right) edges of the flat receiver
    # This calculation only considers a transversal plane analysis
    vll_t, vlr_t = sal - p, sar - p

    # The longitudinal components of the above vectors due to the longitudinal incident of the sunlight
    yl = - vll_t.dot(ni) / ni[1]
    yr = - vlr_t.dot(ni) / ni[1]

    # The actual receiver shading limiting vectors
    vl_rs = array([vll_t[0], yl, vll_t[2]])
    vr_rs = array([vlr_t[0], yr, vlr_t[2]])

    return vl_rs, vr_rs


@njit
def receiver_shading_analysis(p: float64[:],
                              ni: float64[:], vi: float64[:],
                              sal: float64[:], sar: float64[:],
                              length: float64) -> float64:
    """
    This function calculates the fraction of the length (in the longitudinal direction) of a point 'p' in the surface of
    a primary mirrors that is not shaded by the receiver due to the longitudinal component of the incident sunlight.

    The non-shaded length, denominated here by ns_len_rec, is a number between 0 and 1.

    :param p: A [x, 0, z] point-array in the surface of a primary mirror.
    :param ni: The normal vector which defines the incidence plane.
    :param vi: The vector which represents the main direction of the incident sunlight.
    :param sal: The left-edge of the flat receiver.
    :param sar: The right-edge of the flat receiver.
    :param length: The length of the concentrator.

    :return: The fraction of the length that is not shaded by the receiver, a value between 0 and 1
    """

    # receiver shading vectors
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)

    # average width of the longitudinal component of the receiver shading vectors
    ym = abs(vl_rs[1] + vr_rs[1]) / 2.

    # calculates the non-shaded length of the primary mirrors
    if ni.dot(cross(vl_rs, vi)) >= 0 and ni.dot(cross(vr_rs, vi)) <= 0 and ym < length:
        ns_len_rec = ym / length
    else:
        ns_len_rec = 1.

    return ns_len_rec


@njit
def neighbor_shading_analysis(p: float64[:], theta_t_rad: float64,
                              ni: float64[:], vi: float64[:],
                              vms: float64[:], length: float64) -> float64:
    """
    This function calculates the fraction of the length (in the longitudinal direction) of a point 'p' in the surface of
    a primary mirrors that is not shaded by its neighbor due to the longitudinal component of the incident sunlight.

    The non-shaded length, denominated here by ns_sha_len, is a number between 0 and 1.

    :param p: a [x, 0, z] point in the surface of a primary mirror.
    :param theta_t_rad: the transversal incidence angle, in radians.
    :param ni: the normal vector which defines the incidence plane.
    :param vi: the vector which represents the direction of the incident sunlight.
    :param vms: the neighbor shading vector.
    :param length: the length of the concentrator.

    :return: The fraction of the length that is not shaded by the neighbor in the interval [0, 1]
    """

    if theta_t_rad != 0.:
        shaded = 1 if sign(theta_t_rad) * ni.dot(cross(vms, vi)) >= 0 else 0
    else:
        shaded = 1 if sign(p[0]) * ni.dot(cross(vms, vi)) <= 0 else 0

    if shaded == 1 and abs(vms[1]) < length:
        n_sha_len = abs(vms[1]) / length  # vms points as vi, so its 'y' component is negative.
    else:
        n_sha_len = 1

    return n_sha_len


@njit
def blocking_analysis(p: float64[:],
                      vn: float64[:], nr: float64[:],
                      vmb: float64[:]) -> float64:
    """
    This function calculates if the point 'p' in the surface of a primary mirrors is being blocked by its neighbor.

    :param p: a [x, 0, z] point in the surface of a primary mirror.
    :param vn: the direction of the reflected sunlight.
    :param nr: the normal vector to the reflection plane.
    :param vmb: the neighbor-blocking vector.

    :return: It returns 0 if the point is not blocked, and 1 if it is.
    """

    if sign(p[0]) * nr.dot(cross(vn, vmb)) >= 0:
        blocked = 0
    else:
        blocked = 1

    return blocked


@njit
def focusing_analysis(vn: float64[:], nr: float64[:],
                      vll: float64[:], vlr: float64[:]) -> float64:
    """
    This function verify is the reflected ray hits the receiver.
    This is done by checking if the direction of the reflected sunlight lies between the receiver limiting vectors.

    :param vn: the direction vector of the reflected sunlight.
    :param nr: the normal vector to the reflection plane.
    :param vll: the left limiting vector
    :param vlr: the right limiting vector

    :return: It returns 1 if the reflected ray lies within the limiting vectors, and 0 if not.
    """

    if nr.dot(cross(vn, vlr)) <= 0 and nr.dot(cross(vn, vll)) >= 0:
        focused = 1
    else:
        focused = 0

    return focused


@njit
def collimated_end_losses(p: float64[:], vn: float64[:],
                          sar: float64[:], sal: float64[:], sm: float64[:],
                          length: float64) -> float64:
    # The contour of the heliostats are defined by [x, 0, z] point-arrays.
    # Then, the interception between the reflected ray at 'p' and a plane defined by the receiver aperture has a
    # y-component that measures the longitudinal displacement (from 'p') of where the reflected ray would
    # hit the receiver. In this sense, this displacement is the lost length of point 'p' surface.
    rec_ap_normal = R(alpha=pi / 2, v=array([0, 1., 0])).dot(nrm(sar - sal))
    lost_length = abs(islp(p, vn, sm, rec_ap_normal)[1]) / length
    lost_length = 1. if lost_length > 1. else lost_length

    return lost_length


@njit
def local_collimated_analysis(p: float64[:], i: int64, n: int64, centers: float64[:, :],
                              theta_t_rad: float64, theta_l_rad: float64,
                              vi: float64[:], ni: float64[:],
                              vn: float64[:], nr: float64[:],
                              vms: float64[:], vmb: float64[:],
                              sal: float64[:], sar: float64[:], sm: float64[:],
                              length: float64,
                              end_losses: bool_) -> Tuple[float64, float64, float64, float64, float64]:
    """
    This function calculates the geometric performance factors of a point 'p' in the surface of a primary mirror.
    The performance metrics are the intercept factor and optical losses such as shading, blocking, defocusing, and
    end-losses.

    This function, as it name indicates, considers a collimated sunlight model.

    :param p: A [x, 0, z] point-array in the surface of a primary mirror.
    :param i: The index which represents the heliostat in the primary field.
    :param n: The number of mirrors in the primary field.
    :param centers: An array with the center points of all heliostats. Each center is defined by an
    [x, 0, z] point array

    :param theta_t_rad: The transversal incidence angle, in radians.
    :param theta_l_rad: The longitudinal incidence angle, in radians

    :param vi: The vector which represents the main direction of the incident sunlight.
    :param ni: The normal vector which defines the incidence plane.
    :param vn: The direction of the reflected sunlight.
    :param nr: The normal vector to the reflection plane.

    :param vms: The neighbor shading vector.
    :param vmb: The neighbor blocking vector.

    :param sal: The left edge point of the receiver.
    :param sar: The right edge point of the receiver.
    :param sm: The aim-point at the receiver.

    :param length: The length of the concentrator.
    :param end_losses: A boolean sign to compute (True) or not (False) end-losses.

    :return: It returns the performance factor
    """

    # Calculating the shading effects #######################################################################
    # It can occur shading from the neighbor or shading from the receiver
    # Due to the longitudinal component of the incident sunlight,
    # a fraction of the length of the small segment defined by point 'p' could be not shaded
    # even when the shading analysis indicates for shading.
    # In this sense, a non-shaded length (ns_len) of the small segment can exist.

    # Calculate the non-shaded length due to receiver and neighbor
    # These values are represented as fractions of the length of the concentrator,
    # i.e., values within [0,1].
    ns_len_receiver = receiver_shading_analysis(p=p,
                                                ni=ni, vi=vi,
                                                sal=sal, sar=sar,
                                                length=length)

    ns_len_nei = neighbor_shading_analysis(p=p, theta_t_rad=theta_t_rad,
                                           ni=ni, vi=vi,
                                           vms=vms,
                                           length=length)

    # The central heliostat is never shaded by its neighbors at normal incidence
    if theta_t_rad == 0 and is_central_heliostat(center=centers[i], rec_aim=sm):
        ns_len_nei = 1

    # Especial conditions for the neighbor shading #######################################
    # for theta_t > 0, the first heliostat (i == 0) is never shaded by its neighbor
    if theta_t_rad > 0. and i == 0:
        ns_len = 1.
    # for theta_t < 0, the last heliostat (i == n - 1) is never shaded
    elif theta_t_rad < 0. and i == n - 1:
        ns_len = 1.
    else:
        ns_len = min(ns_len_receiver, ns_len_nei)

    # The central mirror is never shaded by its neighbors at normal incidence.
    # the central mirror is always below the aim-point at the receiver in the x-axis
    # (angular position = 0)
    # elif theta_t_rad == 0 and round((sm - centers[i])[0], 4) == 0.:
    # elif theta_t_rad == 0. and is_central_heliostat(center=centers[i], rec_aim=sm):
    #     ns_len_nei = 1
    ########################################################################################

    #########################################################################################################

    # End-losses calculations #################################################################################
    # End-losses are calculated as the fraction of the concentrator length (elo_len) that is lost because
    # it would require a longer receiver (in the longitudinal direction) to intercept the reflected sunlight
    # End-losses only occur if the longitudinal incidence angle is greater than zero.
    if theta_l_rad == 0. or not end_losses:
        elo_len = 0.
    else:
        elo_len = collimated_end_losses(p=p, vn=vn,
                                        sar=sar, sal=sal, sm=sm,
                                        length=length)
    ##########################################################################################################

    # Performance factors ########################################################################################

    # If all length of the segment is being shaded (ns_len == 0) by the neighbor or receiver, then no energy could be
    # collected (intercept by the receiver) nor other losses could exist.
    # Therefore, shading and the non-shaded length of the segment are related by:
    sha = 1. - ns_len

    # Then, blocking losses (blo) can only occur in the non-shaded length of the segment
    blo = ns_len * blocking_analysis(p=p, vn=vn, nr=nr, vmb=vmb)

    # Defocusing losses (de) can only occur if the ray is not blocked
    vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
    de = ns_len * (1. - focusing_analysis(vn=vn, nr=nr, vll=vll, vlr=vlr)) if blo == 0 else 0

    # End-losses can only occur if the ray would hit the receiver and only for the fraction that is not-shaded
    if de == 0.:
        elo = elo_len - sha if elo_len > sha else 0.
    else:
        elo = 0.

    # Intercept energy by the receiver.
    if elo < ns_len:
        pt_if = ns_len - elo
    else:
        pt_if = 0.
        elo = ns_len

    return pt_if, sha, blo, de, elo


@njit
def angular_interval(lv: float64, uv: float64) -> float64[:]:
    """
    This function defined an angular interval in which the effective source should be integrated.

    :param lv: interval lower value
    :param uv: interval upper value

    :return: An angular interval defined as [lv, uv] array.
    """

    return array([lv, uv])


@njit
def empty_interval() -> float64[:]:
    return angular_interval(0., 0.)


@njit
def is_empty(a: float64[:]) -> bool_:
    return True if a[0] == a[1] else False


@njit
def interval_union(a: float64[:], b: float64[:]) -> float64[:]:
    return angular_interval(lv=min(a[0], b[0]), uv=max(a[1], b[1]))


@njit
def interval_intersection(a: float64[:], b: float64[:]) -> float64[:]:
    if a[1] < b[0] or a[0] > b[1]:
        return empty_interval()
    else:
        return angular_interval(lv=max(a[0], b[0]), uv=min(a[1], b[1]))


@njit
def invert_interval(a: float64[:]) -> Tuple[float64[:], float64[:]]:
    return angular_interval(lv=-pi, uv=a[0]), angular_interval(lv=a[1], uv=pi)


@njit
def reduce_interval(a: float64[:], b: float64[:]) -> float64[:]:
    # Calculates the intersection between intervals 'a' and 'b'
    c = interval_intersection(a, b)
    # If the intersection is empty, the output is the interval 'a': a - 0 = a.
    if is_empty(c):
        out_put = a
    # If an intersection exist between the two intervals
    else:
        # Calculates the inverse of the intersection
        d = invert_interval(c)

        if is_empty(interval_intersection(a, d[0])):
            out_put = interval_intersection(a, d[1])
        elif is_empty(interval_intersection(a, d[1])):
            out_put = interval_intersection(a, d[0])
        else:
            out_put = interval_union(interval_intersection(a, d[0]), interval_intersection(a, d[1]))
    #########################################################################################################

    return out_put


@njit
def interval2flux(interval: float64[:], cum_eff: float64[:, :]) -> float64:
    a, b = interval

    f_a, f_b = interp([abs(a), abs(b)], xp=cum_eff.T[0], fp=cum_eff.T[1])

    flux_fraction = abs(f_b - sign(b) * sign(a) * f_a)

    return flux_fraction


@njit
def ang_pnd(u: float64[:], v: float64[:], n: float64[:]) -> float64:
    value = n.dot(cross(u, v))

    return sign(value) * ang(v=v, u=u)


@njit
def inc_beam_analysis(p: float64[:], theta_t: float64,
                      vi: float64[:], ni: float64[:],
                      sal: float64[:], sar: float64[:],
                      vms: float64[:]) -> Tuple[float64[:], float64[:]]:
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)

    signal = 1. if p[0] < 0. else -1.

    theta1 = ang_pnd(u=vi, v=vr_rs, n=ni)
    theta2 = ang_pnd(u=vi, v=vl_rs, n=ni)
    theta3 = ang_pnd(u=vi, v=vms, n=ni)

    if theta_t == 0.:
        r_shading = angular_interval(min(theta1, theta2), max(theta1, theta2))
        n_shading = angular_interval(min(pi * signal, theta3), max(pi * signal, theta3))
    else:
        r_shading = angular_interval(min(theta1, theta2), max(theta1, theta2))
        n_shading = angular_interval(min(sign(theta_t) * pi, theta3), max(sign(theta_t) * pi, theta3))

    return r_shading, n_shading


@njit
def ref_beam_analysis(p: float64[:],
                      vn: float64[:], nr: float64[:],
                      vmb: float64[:],
                      vll: float64[:], vlr: float64[:]) -> Tuple[float64[:], float64[:]]:
    signal = 1. if p[0] > 0. else -1.

    theta1 = ang_pnd(u=vn, v=vlr, n=nr)
    theta2 = ang_pnd(u=vn, v=vll, n=nr)
    theta3 = ang_pnd(u=vn, v=vmb, n=nr)

    blocking = angular_interval(min(pi * signal, theta3), max(pi * signal, theta3))
    intercepted = angular_interval(min(theta1, theta2), max(theta1, theta2))

    return blocking, intercepted


@njit
def one_segment_flux_analysis(r_shading: float64[:],
                              n_shading: float64[:],
                              blocking: float64[:],
                              intercepted: float64[:]) -> Tuple[float64[:], float64[:], float64[:]]:
    n_sha = reduce_interval(a=n_shading, b=r_shading)
    # update neighbor blocking interval
    n_blo = reduce_interval(a=blocking, b=r_shading)
    n_blo = reduce_interval(a=n_blo, b=n_sha)
    # update intercept interval
    i1 = reduce_interval(a=intercepted, b=r_shading)
    i1 = reduce_interval(a=i1, b=n_sha)
    i1 = reduce_interval(a=i1, b=n_blo)

    return n_sha, n_blo, i1


@njit
def flux_analysis_lost_length(theta_l: float64, pt_if: float64,
                              vll: float64[:], vlr: float64[:],
                              length: float64) -> float64:
    if theta_l == 0. or pt_if == 0.:
        lost_length = 0.
    else:
        y_end = max(vll[1], vlr[1])
        if y_end >= length:
            lost_length = 1.
        else:
            lost_length = y_end / length

    return lost_length


@njit
def simple_flux_analysis(theta_l_rad: float64, length: float64,
                         vll: float64, vlr: float64[:],
                         r_shading: float64[:], n_shading: float64[:],
                         blocking: float64[:], intercepted: float64[:],
                         cum_eff: float64[:, :],
                         end_losses: bool_) -> Tuple[float64, float64, float64, float64, float64]:
    n_sha_interval, n_blo_interval, intercept_interval = one_segment_flux_analysis(r_shading=r_shading,
                                                                                   n_shading=n_shading,
                                                                                   blocking=blocking,
                                                                                   intercepted=intercepted)

    sha = interval2flux(interval=r_shading, cum_eff=cum_eff) + interval2flux(interval=n_sha_interval, cum_eff=cum_eff)
    blo = interval2flux(interval=n_blo_interval, cum_eff=cum_eff)
    pt_if = interval2flux(interval=intercept_interval, cum_eff=cum_eff)

    # Computing end-losses
    # Calculating the segment length that is not being used due to the finite length of the receiver.
    if end_losses:
        lost_length = flux_analysis_lost_length(theta_l=theta_l_rad, pt_if=pt_if,
                                                vll=vll, vlr=vlr, length=length)
    else:
        lost_length = 0.

    # Computing end-losses.
    elo = pt_if * lost_length
    # updating the intercepted fraction of the beam
    pt_if = pt_if - elo

    # Computing the defocusing losses,
    # which includes finite acceptance losses (sun shape and optical errors) [1].
    de = 1 - (pt_if + sha + blo + elo)

    return pt_if, sha, blo, de, elo


@njit
def three_segments_flux_analysis(r_shading: float64[:],
                                 n_shading: float64[:],
                                 blocking: float64[:],
                                 intercepted: float64[:]) -> \
        Tuple[float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]]:
    # First length section -- shading by the neighbor and the receiver
    n_sha1 = reduce_interval(a=n_shading, b=r_shading)
    # update neighbor blocking interval
    n_blo1 = reduce_interval(a=blocking, b=r_shading)
    n_blo1 = reduce_interval(a=n_blo1, b=n_sha1)
    # update intercept interval
    i1 = reduce_interval(a=intercepted, b=r_shading)
    i1 = reduce_interval(a=i1, b=n_sha1)
    i1 = reduce_interval(a=i1, b=n_blo1)
    # Second length section -- shaded only by the neighbor
    # update neighbor blocking interval
    n_blo2 = reduce_interval(a=blocking, b=n_shading)
    # update intercept interval
    i2 = reduce_interval(a=intercepted, b=n_shading)
    i2 = reduce_interval(a=i2, b=n_blo2)
    # calculate the intercepted flux for the second length section
    # Third section of segment -- not shaded length
    # update intercept interval
    i3 = reduce_interval(a=intercepted, b=blocking)
    # calculate the intercepted flux for the third length section

    return n_sha1, n_blo1, n_blo2, i1, i2, i3


@njit
def segments_end_losses(lost_length: float64,
                        l1: float64, l2: float64,
                        Int: Tuple[float64, float64, float64],
                        Blo: Tuple[float64, float64, float64]) -> Tuple[float64, float64]:
    I1, I2, I3 = Int
    B1, B2, B3 = Blo

    if lost_length >= l2 + l1:
        elo = l2 * I2 + l1 * I1 + (lost_length - l1 - l2) * I3
        elo_b = l2 * B2 + l1 * B1 + (lost_length - l1 - l2) * B3
    elif lost_length >= l1:
        elo = l1 * I1 + (lost_length - l1) * I2
        elo_b = l1 * B1 + (lost_length - l1) * B2
    else:
        elo = lost_length * I1
        elo_b = lost_length * B1

    return elo, elo_b


@njit
def non_shaded_lengths(p: float64[:], ni: float64[:],
                       vms: float64[:],
                       sal: float64[:], sar: float64[:],
                       length: float64) -> Tuple[float64, float64]:
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)
    ym = abs(vl_rs[1] + vr_rs[1]) / 2.

    ns_len_receiver = ym / length if ym <= length else 1.
    ns_len_nei = abs(vms[1]) / length if abs(vms[1]) <= length else 1.

    return ns_len_receiver, ns_len_nei


@njit
def segments_flux_analysis(p: float64[:], ni: float64[:],
                           theta_l_rad: float64, length: float64,
                           vll: float64, vlr: float64[:],
                           sal: float64[:], sar: float64[:],
                           vms: float64[:],
                           r_shading: float64[:], n_shading: float64[:],
                           blocking: float64[:], intercepted: float64[:],
                           cum_eff: float64[:, :], end_losses: bool_):
    ns_len_rec, ns_len_nei = non_shaded_lengths(p=p, ni=ni, vms=vms,
                                                sal=sal, sar=sar,
                                                length=length)
    # fraction of segment surface subjected to all losses
    l1 = 1.0 - ns_len_rec
    # fraction of the segment not subjected to receiver shading
    l2 = ns_len_rec - ns_len_nei
    # fraction of the segment not subjected to shading -- receiver or neighbor
    l3 = 1.0 - l1 - l2

    # Calculating the angular intervals for each one of the non-shaded and non-blocked segments.
    n_sha1, n_blo1, n_blo2, i1, i2, i3 = three_segments_flux_analysis(r_shading=r_shading,
                                                                      n_shading=n_shading,
                                                                      blocking=blocking,
                                                                      intercepted=intercepted)
    # Intercepted flux by the receiver
    I1 = interval2flux(interval=i1, cum_eff=cum_eff)
    I2 = interval2flux(interval=i2, cum_eff=cum_eff)
    I3 = interval2flux(interval=i3, cum_eff=cum_eff)

    pt_if = l1 * I1 + l2 * I2 + l3 * I3

    # Shading losses #######################################################################################
    #  for the first interval, where all losses occur -- sums receiver and neighbor shading.
    SH1 = interval2flux(interval=r_shading, cum_eff=cum_eff) + interval2flux(interval=n_sha1, cum_eff=cum_eff)
    # Shading losses for the second interval, where it does not have receiver shading.
    SH2 = interval2flux(interval=n_shading, cum_eff=cum_eff)
    # Final computation of shading losses -- the composition of angular intervals and sections lengths.
    sha = l1 * SH1 + l2 * SH2

    # Blocking losses
    B1 = interval2flux(interval=n_blo1, cum_eff=cum_eff)
    B2 = interval2flux(interval=n_blo2, cum_eff=cum_eff)
    B3 = interval2flux(interval=blocking, cum_eff=cum_eff)

    blo = l1 * B1 + l2 * B2 + l3 * B3

    # End-losses calculations.
    # Calculation of the lost length due to end-effect
    if end_losses:
        lost_length = flux_analysis_lost_length(theta_l=theta_l_rad, pt_if=pt_if,
                                                vll=vll, vlr=vlr, length=length)
        elo, elo_b = segments_end_losses(lost_length=lost_length, l1=l1, l2=l2,
                                         Int=(I1, I2, I3), Blo=(B1, B2, B3))
    else:
        lost_length, elo, elo_b = 0., 0., 0.

    # update of the point intercept factor and blocking losses -- discount of end-losses
    if elo > 0:
        pt_if = pt_if - elo if pt_if > elo else 0.  # discount the end-losses from the intercepted flux
        blo = blo - elo_b if blo > elo_b else 0.  # discount the end-losses from blocking losses
    else:
        pt_if = pt_if

    de = 1. - (pt_if + sha + blo + elo)

    return pt_if, sha, blo, de, elo


@njit
def local_flux_analysis(p: float64[:], i: int64, n: int64, centers: float64[:, :],
                        theta_t_rad: float64, theta_l_rad: float64,
                        vi: float64[:], ni: float64[:],
                        vn: float64[:], nr: float64[:],
                        vms: float64[:], vmb: float64[:],
                        sal: float64[:], sar: float64[:], sm: float64[:],
                        cum_eff: float64[:, :],
                        length: float64,
                        end_losses: bool_ = False,
                        simple_analysis: bool_ = False) -> Tuple[float64, float64, float64, float64, float64]:
    n_hel = n
    ######################################################################

    # Incident beam flux analysis ######################################################
    # It calculates the angular intervals in which receiver and neighbor shading exist
    r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t_rad,
                                             vi=vi, ni=ni,
                                             sal=sal, sar=sar,
                                             vms=vms)

    ###################################################################################

    # Calculating special cases for the neighbor shading ##############################
    # Cases where the first and last mirror will never be shaded by the neighbors
    if (theta_t_rad > 0 and i == 0) or (theta_t_rad < 0 and i == n_hel - 1):
        n_shading = empty_interval()

    # Correct account for neighbor shading of the central heliostat at normal incidence
    # if theta_t_rad == 0 and round(centers[i][0], 4) == 0.:
    if theta_t_rad == 0 and is_central_heliostat(center=centers[i], rec_aim=sm):
        n_shading = empty_interval()
    ###################################################################################

    # Reflected beam flux analysis ####################################################
    # Calculating limiting vectors for a flat receiver.
    vll, vlr = flat_receiver_limiting_vectors(p=p,
                                              sal=sal, sar=sar,
                                              nr=nr)
    # Calculating the angular intervals related with the reflected beam.
    blocking, intercepted = ref_beam_analysis(p=p,
                                              vn=vn, nr=nr,
                                              vmb=vmb,
                                              vll=vll, vlr=vlr)

    # Special case of blocking
    # The central heliostat, right below the receiver is never blocked by its neighbors
    # on the other hand, it can be shaded by the receiver and its neighbors
    # if round(centers[i][0], 4) == 0.:
    if is_central_heliostat(center=centers[i], rec_aim=sm):
        blocking = empty_interval()
    ###################################################################################

    ####################################################################################################################
    # Calculating the local intercept factor and optical losses ########################################################
    if simple_analysis:
        pt_if, sha, blo, de, elo = simple_flux_analysis(theta_l_rad=theta_l_rad, length=length,
                                                        vll=vll, vlr=vlr,
                                                        r_shading=r_shading, n_shading=n_shading,
                                                        blocking=blocking, intercepted=intercepted,
                                                        cum_eff=cum_eff, end_losses=end_losses)

    else:
        pt_if, sha, blo, de, elo = segments_flux_analysis(p=p, ni=ni,
                                                          theta_l_rad=theta_l_rad, length=length,
                                                          vll=vll, vlr=vlr,
                                                          sal=sal, sar=sar,
                                                          vms=vms,
                                                          r_shading=r_shading, n_shading=n_shading,
                                                          blocking=blocking, intercepted=intercepted,
                                                          cum_eff=cum_eff, end_losses=end_losses)
    ####################################################################################################################

    return pt_if, sha, blo, de, elo


@njit
def incidence_data(theta_t: float64, theta_l: float64) -> Tuple[float64[:], float64[:], float64, float64]:
    Ix = array([1., 0, 0])

    vi = sun_direction(theta_t, abs(theta_l))  # direction of incident sunlight
    ni = cross(vi, Ix)  # normal vector that defines the incidence plane

    # incidence angles in radians to be used
    theta_t_rad = theta_t * pi / 180.
    theta_l_rad = abs(theta_l * pi / 180.)

    return vi, ni, theta_t_rad, theta_l_rad


@njit
def flat_receiver_data(s1: float64[:], s2: float64[:], aim: float64[:]) -> Tuple[float64[:], float64[:], float64[:]]:
    """
    This function calculates left and right edges of the flat receiver, sal and sar, respectively.
    It considers the right-side edge as the one with the higher value of the x-component.

    Furthermore, it ensures the return as [x, 0, z] point-arrays.

    :param s1: One edge-point of the flat receiver.
    :param s2:  The other edge-point of the flat receiver.
    :param aim: The aim-point at the receiver.

    :return: A tuple of three arrays: (right-edge, left-edge, aim-pt)
    """

    if s2[0] > s1[0]:
        sar, sal = s2, s1
    else:
        sar, sal = s1, s2

    sar = array([sar[0], 0, sar[-1]])
    sal = array([sal[0], 0, sal[-1]])
    sm = array([aim[0], 0, aim[-1]])

    return sar, sal, sm


@njit
def cosine_effect(theta_t_rad: float64, theta_l_rad: float64, tau: float64) -> float64:
    num = cos(theta_l_rad) * cos(theta_t_rad - tau)
    den = cos(theta_t_rad) * sqrt(1 + cos(theta_l_rad) ** 2 * tan(theta_t_rad) ** 2)

    cos_effect = num / den

    return cos_effect


@njit
def analysis_over_field(theta_t_rad: float64, theta_l_rad: float64,
                        vi: float64[:], ni: float64[:],
                        rotated_field: float64[:, :, :], rotated_normals: float64[:, :, :],
                        centers: float64[:, :], tau: float64[:], widths: float64[:],
                        sal: float64[:], sar: float64[:], sm: float64[:],
                        length: float64,
                        end_losses: bool_,
                        cum_eff: float64[:],
                        simple_analysis: bool_ = False) -> float64[:]:
    # The number of heliostats in the field
    n_hel = len(centers)

    # An array to hold values of losses and intercept factor for each mirror
    gamma = zeros(n_hel)  # intercept factor

    # It runs an analysis for each mirror in heliostats field
    for i, hel in enumerate(rotated_field):

        n_pts = len(hel)
        # creates the arrays that will hold values of losses and intercept factor for each point in the mirror
        hel_if = zeros(n_pts)  # intercept factor

        # Calculates the neighbors mirrors, and the edge points for losses analysis
        edge_pt_b, edge_pt_s = define_neighbors(theta_t_rad=theta_t_rad, i=i, centers=centers,
                                                rotated_field=rotated_field)

        # It runs an analysis for each point in the mirror being analyzed
        for j, p in enumerate(hel):
            ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
            vn = reft(vi, ns)  # direction of the reflected ray on that point
            nr = reft(ni, ns)  # normal vector that defines the reflection plane

            # calculate the neighbor vectors to compute shading and blocking losses
            vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)

            if cum_eff is None:
                pt_if, sha, blo, de, elo = local_collimated_analysis(p=p, i=i, n=n_hel, centers=centers,
                                                                     theta_t_rad=theta_t_rad, theta_l_rad=theta_l_rad,
                                                                     vi=vi, ni=ni, vn=vn, nr=nr,
                                                                     vms=vms, vmb=vmb,
                                                                     sal=sal, sar=sar, sm=sm,
                                                                     length=length, end_losses=end_losses)
            else:
                pt_if, sha, blo, de, elo = local_flux_analysis(p=p, i=i, n=n_hel, centers=centers,
                                                               theta_t_rad=theta_t_rad, theta_l_rad=theta_l_rad,
                                                               vi=vi, ni=ni, vn=vn, nr=nr,
                                                               vms=vms, vmb=vmb,
                                                               sal=sal, sar=sar, sm=sm,
                                                               cum_eff=cum_eff,
                                                               length=length, end_losses=end_losses,
                                                               simple_analysis=simple_analysis)

            # Updating the array with the values for each point of the heliostat
            hel_if[j] = pt_if

        # The intercept factor of a particular heliostat is the average of the values its points.
        # Furthermore, it must account for the cosine effect in the current heliostat.
        gamma[i] = hel_if.mean() * cosine_effect(theta_t_rad=theta_t_rad, theta_l_rad=theta_l_rad, tau=tau[i])

    # The LFR intercept factor in then an average of all its primary mirrors.
    lfr_if = gamma.dot(widths) / widths.sum()  # LFR intercept factor

    return lfr_if


@njit(cache=True)
def intercept_factor(theta_t: float64, theta_l: float64,
                     field: float64[:, :, :], normals: float64[:, :, :],
                     centers: float64[:, :], widths: float64[:],
                     s1: float64[:], s2: float64[:], aim: float64[:],
                     length: float64,
                     cum_eff: float64[:] = None,
                     end_losses: bool_ = False) -> float64:
    """
    This function returns the bi-axial intercept factor of a linear fresnel concentrator with a flat receiver.
    It is based on the method published by Santos et al. [1].

    :param theta_t: Transversal incidence angle, in degrees.
    :param theta_l: Longitudinal incidence angle, in degrees.

    :param field: An array of point-arrays. Each array defined a heliostat composed of a finite number of
    [x, 0, z] point-arrays.

    :param normals: An array of vector-arrays. Each array defined the normal vectors to the heliostat
    in a vertical position composed of a number of [x, 0, z] vector-arrays.

    :param centers: An array with the center points of all heliostats. Each center is defined by an
    [x, 0, z] point array

    :param widths: An array with the widths of all heliostats.

    :param s1: One edge point of the flat receiver.
    :param s2: The other edge point of the flat receiver.
    :param aim: The aim point at the receiver which the heliostats use in the tracking procedure.

    :param cum_eff: A cumulative density function of an effective source, defined as an array of [r, f(r)] data points,
    where r is the angular displacement from the beam main direction.

    :param length: LFR length in the longitudinal direction.
    :param end_losses: A boolean sign to compute (True) or not (False) end-losses.

    :return: It returns the bi-axial intercept factor of a linear fresnel concentrator with a flat receiver.
    A value between 0 and 1
    """

    if abs(theta_t) == 90. or abs(theta_l) == 90.:
        lfr_if = 0.
    else:
        vi, ni, theta_t_rad, theta_l_rad = incidence_data(theta_t=theta_t, theta_l=theta_l)
        sar, sal, sm = flat_receiver_data(s1=s1, s2=s2, aim=aim)

        # calculating the points of each mirror after the proper rotation due to the tracking procedure
        tau, rotated_field, rotated_normals = rotated_field_data(field=field, normals=normals, centers=centers,
                                                                 theta_t_rad=theta_t_rad, sm=sm)

        lfr_if = analysis_over_field(theta_t_rad=theta_t_rad, theta_l_rad=theta_l_rad,
                                     vi=vi, ni=ni,
                                     rotated_field=rotated_field, rotated_normals=rotated_normals,
                                     centers=centers, widths=widths, tau=tau,
                                     sal=sal, sar=sar, sm=sm,
                                     cum_eff=cum_eff,
                                     length=length, end_losses=end_losses, simple_analysis=False)

    return lfr_if


@njit
def local_collimated_acceptance(p: float64[:], i: int64, n: int64, centers: float64[:, :],
                                theta_t_rad: float64,
                                vi: float64[:], ni: float64[:],
                                vn: float64[:], nr: float64[:],
                                sal: float64[:], sar: float64[:], sm: float64[:],
                                vms: float64[:], vmb: float64[:]) -> float64:
    # receiver shading calculations ############################################
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)
    if ni.dot(cross(vl_rs, vi)) >= 0 and ni.dot(cross(vr_rs, vi)) <= 0:
        rec_shading = 1
    else:
        rec_shading = 0
    #############################################################################

    # neighbor shading calculations ###############################################
    if theta_t_rad != 0:
        nei_shading = 1 if sign(theta_t_rad) * ni.dot(cross(vms, vi)) >= 0 else 0
    else:
        nei_shading = 1 if sign(p[0]) * ni.dot(cross(vms, vi)) <= 0 else 0

    if theta_t_rad > 0 and i == 0:
        nei_shading = 0
    if theta_t_rad < 0 and i == n - 1:
        nei_shading = 0
    ###############################################################################

    sha = 1 if rec_shading == 1 or nei_shading == 1 else 0

    blo = blocking_analysis(p=p, vn=vn, nr=nr, vmb=vmb) if sha == 0 else 0
    if is_central_heliostat(center=centers[i], rec_aim=sm):
        blo = 0

    vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
    de = (1 - focusing_analysis(vn=vn, nr=nr, vll=vll, vlr=vlr)) if blo == 0 else 0

    pt_if = 1 if blo == 0 and de == 0 else 0

    ############################################################################

    return pt_if


@njit
def local_flux_acceptance(p: float64[:], i: int64, n: int64,
                          theta_t: float64,
                          centers: float64[:, :],
                          sal: float64[:], sar: float64[:], sm: float64[:],
                          vi: float64[:], ni: float64[:],
                          vn: float64[:], nr: float64[:],
                          vms: float64[:], vmb: float64[:],
                          cum_eff: float64[:, :]) -> float64:
    n_hel = n

    r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t, vi=vi, ni=ni,
                                             sal=sal, sar=sar, vms=vms)

    if (theta_t > 0. and i == 0) or (theta_t < 0. and i == n_hel - 1):
        n_shading = empty_interval()

    # a forced code to correctly compute the central heliostat.
    if theta_t == 0. and is_central_heliostat(center=centers[i], rec_aim=sm):
        n_shading = empty_interval()

    # calculating the angular intervals for blocking losses and intercepted flux
    vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
    blocking, intercepted = ref_beam_analysis(p=p, vn=vn, nr=nr, vmb=vmb, vll=vll, vlr=vlr)

    # the central heliostat, right below the receiver is never blocked by its neighbors
    # on the other hand, it can be shaded by the receiver and its neighbors
    if is_central_heliostat(center=centers[i], rec_aim=sm):
        blocking = empty_interval()

    ns_interval, nb_interval, intercept_interval = one_segment_flux_analysis(r_shading=r_shading,
                                                                             n_shading=n_shading,
                                                                             blocking=blocking,
                                                                             intercepted=intercepted)
    # intercepted flux
    pt_if = interval2flux(interval=intercept_interval, cum_eff=cum_eff)

    return pt_if


@njit(cache=True)
def acceptance_computations(theta_t_rad: float64,
                            vi: float64[:], ni: float64[:],
                            rotated_field: float64[:, :, :], rotated_normals: float64[:, :, :],
                            centers: float64[:, :], widths: float64[:],
                            sal: float64[:], sar: float64[:], sm: float64[:],
                            cum_eff: float64[:] = None) -> float64:
    n_hel = len(rotated_field)
    gamma = zeros(n_hel)

    for i, hel in enumerate(rotated_field):
        # number of points in the discretized heliostat.
        n_pts = hel.shape[0]

        # creates the arrays that will hold values of losses and intercept factor for each point of the mirror
        hel_if = zeros(n_pts)  # intercept factor

        # calculate the neighbors and neighbors edge points.
        edge_pt_b, edge_pt_s, = define_neighbors(theta_t_rad=theta_t_rad, i=i, centers=centers,
                                                 rotated_field=rotated_field)

        # calculations for all points of the heliostat
        for j, p in enumerate(hel):
            # Reflection directions
            ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
            vn = reft(vi, ns)  # direction of the reflected ray on that point
            nr = reft(ni, ns)  # normal vector that defines the reflection plane

            # the neighbor vectors to compute shading and blocking
            vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)

            # efficiency calculations for collimated rays or flux analyses
            if cum_eff is None:
                pt_if = local_collimated_acceptance(p=p, i=i, n=n_hel, centers=centers,
                                                    theta_t_rad=theta_t_rad,
                                                    vi=vi, ni=ni, vn=vn, nr=nr,
                                                    sal=sal, sar=sar, sm=sm,
                                                    vms=vms, vmb=vmb)
            else:
                pt_if = local_flux_acceptance(p=p, i=i, n=n_hel, centers=centers,
                                              theta_t=theta_t_rad, sm=sm,
                                              vi=vi, ni=ni, vn=vn, nr=nr,
                                              sal=sal, sar=sar, vms=vms, vmb=vmb,
                                              cum_eff=cum_eff)

            hel_if[j] = pt_if * cos(ang(vi, ns))  # accounting for cosine losses
            # hel_if[j] = pt_if * dot(vi, ns)  # accounting for cosine losses

        gamma[i] = hel_if.mean()  # the efficiency of the heliostat is an average of all its points

    lfr_if = gamma.dot(widths) / widths.sum()

    return lfr_if


def acceptance_analysis(theta_t: float64,
                        field: float64[:, :, :], normals: float64[:, :, :],
                        centers: float64[:, :], widths: float64[:],
                        s1: float64[:], s2: float64[:], rec_aim: float64[:],
                        cum_eff: float64[:, :] = None,
                        lvalue: float64 = 0.60, dt: float64 = 0.1) -> float64[:, :]:
    """
    This function calculates the transmission-acceptance data of a linear fresnel concentrator with a flat receiver.
    It is based on the method published by Santos et al. [1].

    :param theta_t: Transversal incidence angle, in degrees.

    :param field: An array of point-arrays. Each array defined a heliostat composed of a finite number of
    [x, 0, z] point-arrays.

    :param normals: An array of vector-arrays. Each array defined the normal vectors to the heliostat
    in a vertical position composed of a number of [x, 0, z] vector-arrays.

    :param centers: An array with the center points of all heliostats. Each center is defined by an
    [x, 0, z] point array

    :param widths: An array with the widths of all heliostats.

    :param s1: One edge point of the flat receiver.
    :param s2: The other edge point of the flat receiver.
    :param rec_aim: The aim point at the receiver which the heliostats use in the tracking procedure.

    :param cum_eff: A cumulative density function of an effective source, defined as an array of [r, f(r)] data points,
    where r is the angular displacement from the beam main direction.

    :param lvalue: the lower value of normalized efficiency to run
    :param dt: the increment of the off-axis incidence, in degrees.

    :return: the acceptance data as an (#, 2) shape array
    """
    # On-axis computations #############################################################################################
    # receiver data
    sar, sal, sm = flat_receiver_data(s1=s1, s2=s2, aim=rec_aim)

    # On-axis incidence data
    vi, ni, theta_t_rad, theta_l_rad = incidence_data(theta_t=theta_t, theta_l=0.)

    # On-axis primary field data
    _, rotated_field, rotated_normals = rotated_field_data(field=field, normals=normals, centers=centers,
                                                           theta_t_rad=theta_t_rad, sm=sm)

    # On-axis flux
    on_axis_flux = acceptance_computations(theta_t_rad=theta_t_rad, vi=vi, ni=ni,
                                           rotated_field=rotated_field, rotated_normals=rotated_normals,
                                           centers=centers, widths=widths,
                                           sal=sal, sar=sar, sm=sm,
                                           cum_eff=cum_eff)
    ####################################################################################################################

    # Output variables #################################################################
    # to hold for the transmission-acceptance data, i.e., normalized efficiencies.
    norm_if = [1.0]
    # and off-axis incidence
    off_axis_angles = [0.]
    ####################################################################################

    # Acceptance calculations ##########################################################################################

    # positive off-axis incidence ########################################
    # variable to account the number of loops in the while section.
    k = 1
    while norm_if[-1] > lvalue:

        # off-axis incidence
        inc_angle = theta_t + k * dt
        vi, ni, _, _ = incidence_data(theta_t=inc_angle,
                                      theta_l=0.0)

        # flux at the receiver for the incidence
        lfr_if = acceptance_computations(theta_t_rad=theta_t_rad,
                                         vi=vi, ni=ni,
                                         rotated_field=rotated_field,
                                         rotated_normals=rotated_normals,
                                         centers=centers, widths=widths,
                                         sal=sal, sar=sar, sm=sm,
                                         cum_eff=cum_eff)

        # appending normalized flux and off-axis incidence
        norm_if.append(lfr_if / on_axis_flux)
        off_axis_angles.append(inc_angle - theta_t)

        # checking if last normalized flux is close to zero
        # if a full range acceptance is considered (i.e., lvalue=0) then the loop would never end
        # there are no negative flux
        if round(norm_if[-1], 3) == 0.:
            break

        # updating index for the next off-axis incidence
        k += 1
    ######################################################################

    # negative off-axis incidence ########################################
    # variable to account the number of loops in the while section.
    k = 1
    while norm_if[0] > lvalue:

        # off-axis incidence
        inc_angle = theta_t - k * dt
        vi, ni, _, _ = incidence_data(theta_t=inc_angle,
                                      theta_l=0.0)

        # flux at the receiver for the incidence
        lfr_if = acceptance_computations(theta_t_rad=theta_t_rad,
                                         vi=vi, ni=ni,
                                         rotated_field=rotated_field,
                                         rotated_normals=rotated_normals,
                                         centers=centers, widths=widths,
                                         sal=sal, sar=sar, sm=sm,
                                         cum_eff=cum_eff)

        # appending normalized flux and off-axis incidence
        norm_if.insert(0, lfr_if / on_axis_flux)
        off_axis_angles.insert(0, inc_angle - theta_t)

        # checking if last normalized flux is close to zero
        # if a full range acceptance is considered (i.e., lvalue=0) then the loop would never end
        # there are no negative flux
        if round(norm_if[0], 3) == 0.:
            break

        # updating index for the next off-axis incidence
        k += 1
    ######################################################################
    ####################################################################################################################

    # output array ########################################
    transmission_data = zeros(shape=(len(norm_if), 2))
    transmission_data.T[:] = off_axis_angles, norm_if
    #######################################################

    return transmission_data
