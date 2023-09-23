
from typing import Any
from numpy import arccos, array, dot, pi, sign, cos, sin, cross


def V(x: float):
    """
    This function returns a unit vector defined by angle 'x'. It considers that 'x' is the angle of the unit vector
    regarding the horizontal direction.

    :param x: An angle, in radians

    :return: A xy vector-array.
    """
    return array([cos(x), sin(x)])


def mag(v: array) -> float:
    """
    This function returns the magnitude (length) of vector 'v'.

    :param v: a vector-array

    :return: A float.
    """
    return dot(v, v) ** 0.5


# @njit('float64[:](float64[:])', cache=True)
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


def ang(v: array, u: array) -> float:
    """
    This function returns the dihedral angle between two vectors in the range from 0 to pi.

    :param v: a vector-array.
    :param u: a vector-array.

    :return: an angle, in radians.
    """
    vv = nrm(v)
    uu = nrm(u)

    u_dot_v = vv.dot(uu)

    return arccos(u_dot_v.round(12))


def ang_p(v: array, u: array, ) -> float:
    """
    This function returns the angle between vector 'v' regarding vector 'u'.
    That is, the angle that vector 'v' makes relative to vector 'u' in the positive direction.
    In this sense, it returns an angle within 0 to 2*pi.

    :param v: a vector-array in the xy plane
    :param u: a vector-array in the xy plane

    :return: An angle, in radians.
    """

    alpha = ang(v=v, u=u)

    angle = alpha if (u[0] * v[1] - u[1] * v[0] >= 0) else 2. * pi - alpha

    return angle


def ang_h(v: array) -> float:
    """
    :param v: an array vector

    :return: the angle vector v makes to the horizontal {1,0} in the positive direction and in the range from 0 to 2*pi.
    """

    return ang_p(v=v, u=array([1, 0]))


def ang_pn(v: array, u: Any = None) -> float:
    """
    This function calculates the angle that vector 'v' makes regarding vector 'u'.
    It is positive if u is clockwise from v and negative otherwise.

    If 'u' is None, it considers the angle regarding the horizontal direction.
    In this case, it is positive if 'v' points up and negative if 'v' points down.

    :param v: A vector-array in the xy plane.
    :param u: A vector-array in the xy plane.

    :return: An angle, in radians.
    """

    if u is None:
        alpha = sign(v[1]) * ang_h(v=v)
    else:
        alpha = ang(v=v, u=u) if (u[0] * v[1] - u[1] * v[0] >= 0) else -ang(v=v, u=u)

    return alpha


def ang_pnd(u: array, v: array, n: array) -> float:
    """
    This function (...)

    :param u:
    :param v:
    :param n:

    :return:
    """

    return sign(dot(n, cross(u, v))) * ang(v=v, u=u)


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


def Rot(v: array, alpha: float) -> array:
    """
    This function (...).
    rotates a vector v by an angle alpha.

    :param v:
    :param alpha:

    :return: This function returns the rotated vector v by the angle alpha
    """

    return dot(R(alpha=alpha), v)


def islp(p: array, v: array, q: array, n: array) -> array:
    """
    This function (...).

    If the geometry is three-dimensional, returns the intersection point between a straight line defined by point P
    and vector v and a plane defined by point Q and normal vector n.
    If the geometry is two-dimensional, the function returns the intersection of a straight line defined by point P
    and vector v and another straight line trough point Q with normal n.

    :param p:
    :param v:
    :param q:
    :param n:

    :return:
    """

    vn = nrm(v)
    nn = nrm(n)

    return p + vn * dot((q - p), nn) / dot(vn, nn)


def isl(p: array, v: array, q: array, u: array) -> array:
    """
    This function

    :param p:
    :param v:
    :param q:
    :param u:

    :return:
    """

    return islp(p, v, q, Rot(v=u, alpha=pi / 2))


def mid_point(p: array, q: array) -> array:
    """
    This function (...).

    :param p: a point in space
    :param q: a point in space

    :return: the mid-point between p and q
    """

    return (p + q) * 0.5


def Sy(v: array) -> array:
    """
    This function (...)

    :param v: a 2D (x-y) point, or vector.

    :return: returns the symmetrical to v relative to the y-axis
    """

    m = array([[-1, 0], [0, 1]])

    return dot(m, v)


def Sx(v: array) -> array:
    """
    :param v: a 2D (x-y) point, or vector.
    :return: returns the symmetrical to v relative to the x-axis
    """
    m = array([[1, 0], [0, - 1]])

    return dot(m, v)


def proj_vector_plane(v: array, n: array) -> array:
    """
    This function (...)

    :param v: a 3D vector
    :param n: a 3D vector that represents a normal to a plane
    :return: This function returns the projection of v onto a plane defined by its normal vector n
    """

    u = n * dot(v, n) / dot(n, n)

    return v - u


def tgs2tube(point: array, tube_center: array, tube_radius: float) -> array:
    """
    This function (...).

    :param point: A point in the xy plane
    :param tube_center: Tube center point
    :param tube_radius: Tube absorber_radius

    :return: It returns the two points in the tube surface which tangent lines passes through the outer point.
    """

    beta = arccos(tube_radius / dst(p=point, q=tube_center))

    p1 = tube_center + R(beta).dot(nrm(point - tube_center)) * tube_radius
    p2 = tube_center + R(-beta).dot(nrm(point - tube_center)) * tube_radius

    return array([p1, p2])
