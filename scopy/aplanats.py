from numpy import array, tan, power, linspace, sin, zeros, cos, arcsin, pi


def dual_aplanat_functions_for_tube(s: float, k: float):
    """
    This function returns the primary and secondary optics closed-form equations for a dual aplanat design for a linear
    concentrator, as given by the input parameters 's' and 'k'.

    It is based on the expressions are based on Gordon's studies [1, 2, 3], which consider the Abbe's unit sphere to
    give dimensionless equations. Furthermore, the aplanat focus is at the ecs_origin: [0, 0].

    [1] Solar Energy 2019;191:697–706. https://doi.org/10.1016/j.solener.2019.08.037.
    [2] Optics Express 2010;18:A41. https://doi.org/10.1364/OE.18.000A41.
    [3] Applied Optics 2009;48:4926–31. https://doi.org/10.1364/AO.48.004926.

    :param s: The (dimensionless) distance between secondary vertex and focus.
    :param k: The (dimensionless) distance between primary and secondary vertexes.
    :return: It returns the primary and secondary parametric surface equations.
    """

    # Defining the auxiliary function regarding the surface equations ####
    def g(phi): return s - (1 - s) * tan(0.5 * phi)**2
    def h(phi): return power(abs(g(phi) / s), s / (s - 1))
    ######################################################################

    # The aplanatic primary optic parametric equation ########################################
    def primary_function(phi):
        xp = sin(phi)
        yp = s - cos(0.5 * phi)**2 + (g(phi) / s) * (1 - k * h(phi)) * cos(0.5 * phi)**4
        return array([xp, yp])
    ##########################################################################################

    # The aplanatic secondary optic parametric equation ########################################
    def secondary_function(phi):
        xs = (2 * s * k * h(phi) * tan(0.5 * phi)) / (k * h(phi) * tan(0.5 * phi)**2 + g(phi))
        ys = - xs / tan(phi)
        return array([xs, ys])
    ############################################################################################

    return primary_function, secondary_function


def dual4tube(s: float, k: float, NA: float, phi_min=0.00001, upward=True, nbr_pts=151, tube_center=array([0, 0])):

    """

    This function returns the primary and secondary aplanat optics contour surface points. It is based on the aplanat
    input parameters: 's', 'k', and 'NA'.

    This function is based on Gordon's studies [1, 2, 3], which consider the Abbe's unit sphere to give dimensionless
    equations. Furthermore, the aplanat focus is at the ecs_origin: [0, 0].

    [1] Solar Energy 2019;191:697–706. https://doi.org/10.1016/j.solener.2019.08.037.
    [2] Optics Express 2010;18:A41. https://doi.org/10.1364/OE.18.000A41.
    [3] Applied Optics 2009;48:4926–31. https://doi.org/10.1364/AO.48.004926.

    :param s: The (dimensionless) distance between secondary vertex and focus.
    :param k: The (dimensionless) distance between primary and secondary vertexes.
    :param NA: The Numerical Aperture, a value lower than 1.

    :param phi_min: An angle to represent the degree of truncation of the optics, in radians.
    :param upward: A boolean sign to indicate whether an upward or downward optics is to be designed.
    :param nbr_pts: The number of surface points to be yielded for both primary and secondary optics.
    :param tube_center: A point to translate the aplanats. A new focus.

    :return: This function returns the primary and secondary aplanat optics contour surface points.

    """

    # The parametric functions for the contours of primary and secondary optics ######
    primary_function, secondary_function = dual_aplanat_functions_for_tube(s=s, k=k)
    ##################################################################################

    # defining the angular (discrete) interval for the surface parametric equation #############
    phi_max = arcsin(NA)

    if upward:
        phi_values = linspace(start=phi_min, stop=phi_max, num=nbr_pts)
    else:
        phi_values = linspace(start=pi-phi_max, stop=pi-phi_min, num=nbr_pts)
    #############################################################################################

    # Calculating the primary optic surface points #################
    primary_pts = zeros(shape=(phi_values.size, 2))
    primary_pts[:] = [primary_function(phi) for phi in phi_values]

    # primary_pts_left = array([Sy(pt) for pt in primary_pts])
    ################################################################

    # Calculating the secondary optic surface points ####################
    secondary_pts = zeros(shape=(phi_values.size, 2))
    secondary_pts[:] = [secondary_function(phi) for phi in phi_values]

    # secondary_pts_right = array([Sy(pt) for pt in secondary_pts])
    #####################################################################

    return primary_pts + tube_center, secondary_pts + tube_center
