# -*- coding: utf-8 -*-
"""
Created on Thu, Dec 3,2020 15:47:43
New version: Feb 8 2022 09:00:55
@author: André Santos (andrevitoras@gmail.com / avas@uevora.pt)
"""

import json
from copy import deepcopy
from pathlib import Path

from numpy import array, linspace, pi, zeros, cos, sin, sign, cross, power, sqrt, tan, arctan, \
    absolute, deg2rad, ones, arange, log

from scipy.interpolate import interp1d, LinearNDInterpolator, InterpolatedUnivariateSpline
from scipy.optimize import fsolve

from niopy.geometric_transforms import nrm, ang, R, dst, V, mid_point

from niopy.plane_curves import PlaneCurve
from niopy.reflection_refraction import rfx_nrm
from scopy.linear_fresnel.optical_method import intercept_factor, acceptance_analysis

from scopy.nio_concentrators import symmetric_cpc2tube, cpc_type, symmetric_cpc2evacuated_tube, \
    symmetric_cec2evacuated_tube

from scopy.sunlight import sun2lin, SiteData

from soltracepy import OpticalSurface, Element

from soltracepy.auxiliary import reflective_surface, secondary_surface, absorber_tube_surface, \
    flat_absorber_surface, cover_surfaces, flat_element

from utils import dic2json


class OpticalProperty:
    """
    This class aims to represent common optical properties in the context of the design and analysis of the
    linear Fresnel collector.

    It contains reflective, absorptive and transmissive properties.
    """

    class reflector:
        """
        This class stands for a common one-side reflective surface, as in a linear Fresnel primary mirror.
        """

        def __init__(self, name: str, rho=1.0, slope_error=0., spec_error=0.):
            """
            :param name: The variable_name of the property
            :param rho: The reflector hemispherical reflectance. It should be a value between 0 and 1.
            :param slope_error: The slope error of the reflector surface, in radians.
            :param spec_error: THe specular error of the reflector surface, in radians.
            """

            assert 0 <= abs(rho) <= 1, ValueError('Reflectance value must be between 0 and 1.')

            self.rho = abs(rho)
            self.type = 'reflect'
            self.slope_error = abs(slope_error)
            self.spec_error = abs(spec_error)
            self.name = name

        def to_soltrace(self):
            """
            This method transform the OpticalProperty in an equivalent SolTrace OpticalSurface object.
            See 'soltracepy.OpticalSurface'.

            An OpticalSurface object has two sides: front and back. In this case of a single-side reflector, the front
            is the reflective interface with the correspondent property, and the back side is a perfect absorber.
            It is important to highlight that the front interface is the positive direction given by the z-axis of the
            Element Coordinate System (ECS) of the SolTrace Element in which the property is applied to.

            :return: As equivalent SolTrace OpticalProperty object.
            """

            return reflective_surface(name=self.name, rho=self.rho,
                                      slope_error=self.slope_error * 1e3, spec_error=self.spec_error * 1e3)

    class flat_absorber:

        def __init__(self, name: str, alpha=1.):
            """
            :param alpha: The absorbance of the absorber surface. It must be a value between 0 and 1.
            """

            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')

            self.type = 'reflect'
            self.alpha = abs(alpha)
            self.name = name

        def to_soltrace(self):
            """
            :return: This method returns an equivalent SolTrace Optic object.
            """
            return flat_absorber_surface(name=self.name, alpha=self.alpha)

    class absorber_tube:

        def __init__(self, name: str, alpha=1.):
            """
            :param alpha: The absorbance of the absorber surface. It must be a value between 0 and 1.
            """

            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')

            self.type = 'reflect'
            self.alpha = abs(alpha)
            self.name = name

        def to_soltrace(self):
            """
            :return: This method returns an equivalent SolTrace Optic.
            """

            return absorber_tube_surface(name=self.name, alpha=self.alpha)

    # class transmissive_cover:
    #
    #     def __init__(self, tau: float, refractive_index=1.52, name='cover'):
    #         assert 0 <= abs(tau) <= 1, ValueError('Transmittance value must be between 0 and 1.')
    #
    #         self.name = name
    #         self.tau = abs(tau)
    #         self.ref_index = refractive_index
    #
    #     def to_soltrace(self):
    #
    #         return transmissive_surface(name=self.name, tau=self.tau, nf=1, nb=self.ref_index)

    class evacuated_tube:

        def __init__(self, alpha: float, tau: float, ref_index=1.52, name='evacuated_tube'):
            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')
            assert 0 <= abs(tau) <= 1, ValueError('Transmittance value must be between 0 and 1.')

            self.name = name

            self.alpha = abs(alpha)
            self.tau = abs(tau)
            self.n = abs(ref_index)

        def to_soltrace(self):
            """
            :return: It returns a list with equivalent SolTrace Optics, from the outer cover to the absorber.
            """

            outer_cover, inner_cover = cover_surfaces(tau=self.tau, refractive_index=self.n, name=self.name)

            absorber = OpticalProperty.absorber_tube(name=f'{self.name}_absorber', alpha=self.alpha).to_soltrace()

            return [outer_cover, inner_cover, absorber]

    class secondary:

        def __init__(self, name: str, rho=1.0, slope_error=1.e-4, spec_error=1.e-4):
            """
            :param name: The variable_name of the property.
            :param rho: The reflector hemispherical reflectance. It should be a value between 0 and 1.
            :param slope_error: The slope error of the reflector surface, in mrad.
            :param spec_error: THe specular error of the reflector surface, in mrad.
            """

            assert 0 <= abs(rho) <= 1, ValueError('Reflectance value must be between 0 and 1.')

            self.rho = abs(rho)
            self.type = 'reflect'
            self.slope_error = abs(slope_error)
            self.spec_error = abs(spec_error)
            self.name = name

        def to_soltrace(self):
            """
            :return: This method returns an equivalent SolTrace Optic.
            """

            return secondary_surface(name=self.name, rho=self.rho,
                                     slope_error=self.slope_error, spec_error=self.spec_error)


class Absorber:
    """
    This class stands to represent the most common types of absorber considered in the context of the linear Fresnel
    collector.

    """

    class flat:

        def __init__(self, width: float, center: array, axis=array([1, 0]), name='flat_absorber'):
            self.name = name

            self.width = abs(width)
            self.center = array([center[0], center[-1]])

            self.axis = array([axis[0], axis[-1]])

            self.s1 = - 0.5 * self.width * nrm(self.axis) + self.center
            self.s2 = + 0.5 * self.width * nrm(self.axis) + self.center

            self.sm = mid_point(self.s1, self.s2)

            self.left_edge = self.s1
            self.right_edge = self.s2

            self.contour = array([self.s1, self.s2])

        def as_soltrace_element(self, length: float, optic: OpticalSurface, name=None):
            elem = flat_absorber2soltrace(geometry=self, optic=optic, length=length,
                                          name=self.name if name is None else name)

            return elem

    class tube:

        def __init__(self, radius: float, center: array, name='absorber_tube', nbr_pts=121):
            self.name = name

            self.radius = abs(radius)
            self.center = array([center[0], center[-1]])

            self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

            unit_circle = array([[cos(x), sin(x)] for x in linspace(start=0, stop=2 * pi, num=self.n_pts)])
            self.contour = unit_circle * self.radius + self.center

            self.left_edge = self.center - array([self.radius, 0])
            self.right_edge = self.center + array([self.radius, 0])

        def design_cpc(self, half_acceptance: float, nbr_pts=121, upwards=True):
            l_mp, l_in, r_in, r_mp = symmetric_cpc2tube(theta_a=half_acceptance,
                                                        tube_center=self.center, tube_radius=self.radius,
                                                        degrees=True, upwards=upwards, nbr_pts=nbr_pts)

            return cpc_type(left_conic=l_mp, left_involute=l_in, right_involute=r_in, right_conic=r_mp)

        def as_soltrace_element(self, length: float, optic: OpticalSurface, name: str = None):
            return tube2soltrace(tube=self,
                                 name=self.name if name is None else name,
                                 length=length, optic=optic)

    class evacuated_tube:

        def __init__(self, center: array,
                     absorber_radius: float, inner_cover_radius: float, outer_cover_radius: float,
                     name='evacuated_tube', nbr_pts=121):

            self.name = name

            self.radius = abs(absorber_radius)
            self.center = array([center[0], center[-1]])

            self.abs_radius = abs(absorber_radius)
            self.inner_radius = min(abs(outer_cover_radius), abs(inner_cover_radius))
            self.outer_radius = max(abs(outer_cover_radius), abs(inner_cover_radius))

            self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

            self.outer_tube = Absorber.tube(center=self.center, radius=self.outer_radius,
                                            name='outer_cover', nbr_pts=nbr_pts)

            self.inner_tube = Absorber.tube(center=self.center, radius=self.inner_radius,
                                            name='inner_cover', nbr_pts=nbr_pts)

            self.absorber_tube = Absorber.tube(center=self.center, radius=self.abs_radius,
                                               name='absorber', nbr_pts=nbr_pts)

        def design_cpc(self, half_acceptance: float, nbr_pts=121, upwards=True, dy=0):
            """
            This method designs a Compound Parabolic Concentrator (CPC) for the evacuated tubular absorber.

            :param half_acceptance: The half-acceptance angle of the cpc optic, in degrees.
            :param nbr_pts: The number of points to discretize each section of the cpc optic.
            :param upwards: A boolean sign to return an upwards (True) or downwards (False) cpc optic.
            :param dy: A linear displacement of the cpc cusp to it not touch the outer cover

            :return: A cpc_type object.
            """

            l_mp, l_in, r_in, r_mp = symmetric_cpc2evacuated_tube(theta_a=half_acceptance, tube_center=self.center,
                                                                  tube_radius=self.radius,
                                                                  cover_radius=self.outer_radius,
                                                                  degrees=True, upwards=upwards, dy=dy,
                                                                  nbr_pts=nbr_pts)

            return cpc_type(left_conic=l_mp, left_involute=l_in, right_involute=r_in, right_conic=r_mp)

        def design_cec(self, source_width: float, source_distance: float, nbr_pts=50, upwards=False, dy=0):
            l_me, l_in, r_in, r_me = symmetric_cec2evacuated_tube(tube_center=self.center, tube_radius=self.radius,
                                                                  cover_radius=self.outer_radius,
                                                                  source_width=source_width,
                                                                  source_distance=source_distance,
                                                                  upwards=upwards, dy=dy, nbr_pts=nbr_pts)

            return cpc_type(left_conic=l_me, left_involute=l_in, right_involute=r_in, right_conic=r_me)

        def as_soltrace_element(self, length: float,
                                outer_cover_optic: OpticalSurface,
                                inner_cover_optic: OpticalSurface,
                                absorber_optic: OpticalSurface,
                                name: str = None) -> list:
            return evacuated_tube2soltrace(geometry=self,
                                           name=self.name if name is None else name,
                                           length=length,
                                           outer_cover_optic=outer_cover_optic,
                                           inner_cover_optic=inner_cover_optic,
                                           absorber_optic=absorber_optic)


class Secondary:
    class trapezoidal:

        def __init__(self, aperture_center: array, aperture_width: float, tilt: float, height: float,
                     name='trapezoidal_secondary'):
            self.name = name

            Ix = array([1, 0])
            self.aperture_center = array([aperture_center[0], aperture_center[-1]])
            self.aperture_width = abs(aperture_width)
            self.tilt = abs(tilt)
            self.height = abs(height)

            self.ap_left = - 0.5 * self.aperture_width * Ix + self.aperture_center
            self.ap_right = + 0.5 * self.aperture_width * Ix + self.aperture_center

            self.back_left = self.ap_left + V(self.tilt * pi / 180) * self.height
            self.back_right = self.ap_right + V(pi - self.tilt * pi / 180) * self.height

            self.left_edge = self.ap_left
            self.right_edge = self.ap_right

            self.contour = array([self.ap_left, self.back_left, self.back_right, self.ap_right])

        def as_soltrace_element(self, length: float, optic: OpticalSurface):
            return trapezoidal2soltrace(geometry=self, name=self.name, length=length, optic=optic)


class Heliostat:
    """
    The class Heliostat aims to represent a linear Fresnel primary mirror. It can be flat or cylindrical.

    Parabolic primaries were not included since they are equivalent to the cylindrical ones and not that simple
    to design. For a further understanding, one must read Refs. [1-5], particularly Ref.[6].

    [1] Abbas R., Montes MJ., Piera M., Martínez-Val JM. 2012. https://doi.org/10.1016/j.enconman.2011.10.010.
    [2] Qiu Y, He YL, Cheng ZD, Wang K. 2015. https://doi.org/10.1016/j.apenergy.2015.01.135.
    [3] Abbas R, Martínez-Val JM. 2017. https://doi.org/10.1016/j.apenergy.2016.01.065.
    [4] Boito, P., Grena, R. 2017. https://doi.org/10.1016/j.solener.2017.07.079.
    [5] Cheng ZD, Zhao XR, He YL, Qiu Y. 2018. https://doi.org/10.1016/j.renene.2018.06.019.
    [6] Santos, A.V., Canavarro, D., Horta, P., Collares-Pereira, M., 2023. https://doi.org/10.1016/j.renene.2023.119380
    """

    def __init__(self, width: float, center: array, radius: float, nbr_pts=201):

        # The basic attributes of a Heliostat object #####
        self.width = abs(width)
        self.radius = abs(radius)
        self.center = array([center[0], center[-1]])
        ##################################################

        # Defining an attribute for the shape of the heliostats ###
        # It can be flat or cylindrical
        if self.radius == 0:
            self.shape = 'flat'
        else:
            self.shape = 'cylindrical'
        ############################################################

        # Ensure an odd number of points to discretize the heliostat contour surface ####
        # In this way, the center point is part of the set of points.
        self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
        #################################################################################

        #############################################################################################################
        # Heliostat attributes that are defined as [x,y] point-arrays/vector-arrays #################################

        # The design of the point-arrays which defines the heliostat contour
        # These point-arrays consider the mirror at the horizontal position
        # The edges and the center are in this set.
        if self.shape == 'cylindrical':
            self.contour = design_cylindrical_heliostat(hc=self.center, w=self.width,
                                                        rc=self.radius, nbr_pts=self.n_pts)
        else:
            self.contour = design_flat_heliostat(hc=self.center, w=self.width,
                                                 nbr_pts=self.n_pts)

        # Attribute that holds the Heliostat object as a PlaneCurve object.
        self.curve = self.as_plane_curve()

        # The normal vectors at the heliostat surface points.
        self.normals = self.curve.normals2surface()

        #############################################################################################################
        #############################################################################################################

        #########################################################################################################
        # Heliostat attributes that are defined as [x, 0, z] point-arrays/vector-arrays #########################
        # That is, the transversal plane is the ZX-plane.
        self.zx_center = transform_vector(self.center)
        self.zx_pts = transform_heliostat(self.contour)
        self.zx_normals = transform_heliostat(self.normals)

        # This data is used only to calculate the intercept factor of the heliostat and the linear Fresnel.
        # Therefore, they are only defined in terms of the zx-plane.
        self.seg_pts, self.seg_normals = self.segments_of_equal_projected_aperture()
        #########################################################################################################
        #########################################################################################################

    def segments_of_equal_projected_aperture(self):
        """
        This method calculates [x,0,z] point and normals in the heliostat contour that are the center of segments that
        has the same projected width in the aperture of the heliostat. It ensures a uniform discretization of the mirror
        aperture and are essential to compute efficiency calculations.

        :return: A tuple of [x,0,z] point-arrays and [x,0,z] vector-arrays.
        """

        # The heliostat as a cubic spline ##############################
        hel_spline = self.curve.as_spline(centered=False, rep=False)
        ################################################################

        # The [x,y] point-arrays ####################################################################################
        # Points which define segments of the heliostat surface.
        # Their projected width in the aperture are equal.
        # 'x_values' represent the x-coordinates of central points of segments in the heliostat contour which has
        # a project width on the aperture that is uniform.
        x_coord = array([0.5 * (self.curve.x[i] + self.curve.x[i + 1])
                         for i in range(self.curve.x.shape[0] - 1)])
        # The y-coordinates of the point in the heliostat contour of these 'x_values'.
        y_coord = hel_spline(x_coord)

        seg_pts = zeros(shape=(x_coord.shape[0], 2))
        seg_pts.T[:] = x_coord, y_coord
        ##############################################################################################################

        # The [x,y] normal vectors ###############################
        normals = zeros(shape=(seg_pts.shape[0], 2))
        dy_dx = hel_spline.derivative()

        normals[:] = [nrm(V(arctan(dy_dx(p)))) for p in x_coord]
        normals = R(pi / 2).dot(normals.T).T
        ##########################################################

        # Transforming the [x,y] point and vector-arrays to [x, 0, z] #########
        # seg_points = transform_heliostat(seg_pts).round(3)
        # seg_normals = transform_heliostat(normals).round(6)

        seg_points = transform_heliostat(seg_pts)
        seg_normals = transform_heliostat(normals)
        #######################################################################

        return seg_points, seg_normals

    def as_plane_curve(self):
        """
        A method to return the Heliostat object as a PlaneCurve object defined by the contour point.

        :return: It returns the xy points of the heliostat as a PlaneCurve object.
        """
        return PlaneCurve(curve_pts=self.contour, curve_center=self.center)

    def angular_position(self, aim: array) -> float:
        """
        This method calculates the heliostat angular position regarding a tracking aim-point at the receiver.
        It assumes the ZX-plane as the transversal plane, i.e., the one which defines the linear Fresnel geometry.

        :param aim: The aim-point at the receiver, a point-array.

        :return: The heliostat angular position, in radians.
        """

        # Old version ##################################################
        # This version considers the transversal plane as the XY plane.

        # sm = array([aim[0], aim[-1]])
        # vf = sm - self.center
        # lamb = sign(self.center[0]) * ang(vf, array([0, 1])).rad
        ################################################################

        # New version in 2022-11-08 ####################################
        # This version considers the transversal plane as the ZX plane.
        # Therefore, a different equation is used.
        lamb = angular_position(center=self.center, aim=aim)
        ################################################################

        return lamb

    def tracking_angle(self, aim: array, theta_t: float) -> float:
        """
        This method calculates the heliostat tracking angle regarding an aim-point at the receiver for a particular
        value of transversal incidence angle (in degrees).

        It assumes the ZX-plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param aim: The aim-point at the receiver, a point-array.
        :param theta_t: The transversal incidence angle, in degrees.

        :return: The heliostat tracking angle, in radians.
        """

        theta_t_rad = theta_t * pi / 180.
        lamb = self.angular_position(aim=aim)

        tau = (theta_t_rad - lamb) / 2

        return tau

    def rotated_points(self, aim: array, theta_t='horizontal', return_xy=True, to_calculations=True) -> array:

        """
        This method returns the surface points of the rotated heliostat for a given transversal incidence angle
        ('theta_t'), in degrees, considering a particular tracking aim-point at the receiver ('aim')

        :param aim: The tracking aim-point at the receiver, an array.
        :param theta_t: The transversal incidence angle, in degrees.
        :param return_xy: A boolean sign to whether return [x, y] or [x, 0, z] points.

        :param to_calculations: A boolean sign to identify whether points in the surface will be used to efficiency
        calculations or not. If 'True', points represent segments of the surface that has the same projected width on
        the aperture plane of the heliostat.

        When the param 'theta_t' = 'horizontal', the heliostat is returned in its horizontal position.

        :return: An array of array-points.

        """

        # Check whether horizontal or a particular incidence was inputted. ###################################
        if isinstance(theta_t, (float, int)):
            # A particular transversal incidence was selected.
            # All operations are done considering a ZX plane. Thus, mirrors rotated around the y-axis.
            # and the heliostats points are the kind [x, 0, z]
            Iy = array([0, 1, 0])

            # The correspondent tracking angle for the transversal incidence given by 'theta_t'.
            tau = self.tracking_angle(aim=aim, theta_t=theta_t)

            # Calculating the rotated points of the heliostat ################################################
            # Verify if the points will be used in efficiency calculations or not ############################
            if to_calculations:
                points = self.seg_pts
            else:
                points = self.zx_pts
            rotated_points = rotate_points(points=points, center=self.zx_center, tau=tau, axis=Iy)

        # Return heliostat at the horizontal position #########################################################
        else:
            rotated_points = self.zx_pts
        #######################################################################################################

        # Checking to return [x, y] or [x, 0, z] array-points #################################################
        if return_xy:
            rotated_points = transform_heliostat(rotated_points)
        #######################################################################################################

        return rotated_points

    def rotated_normals(self, aim: array, theta_t: float, return_xy=False, to_calculations=True) -> array:

        """
        This method returns the normal vectors to the contour points of a rotated heliostat for a given
        transversal incidence angle ('theta_t'), in degrees.
        It considers a particular tracking aim-point at the receiver ('aim') -- an array-point.

        :param aim: The tracking aim-point at the receiver, an array.
        :param theta_t: The transversal incidence angle, in degrees.
        :param return_xy: A boolean sign to whether return [x, y] or [x, 0, z] array-vectors.

        :param to_calculations: A boolean sign to identify whether points in the surface will be used to efficiency
        calculations or not. If 'True', points represent segments of the surface that has the same projected width on
        the aperture plane of the heliostat.

        :return: It returns an array of array-vectors.
        """

        Iy = array([0, 1, 0])
        # The correspondent tracking angle for the transversal incidence given by theta_t
        tau = self.tracking_angle(aim=aim, theta_t=theta_t)

        # Calculating the rotated normals vectors of the heliostat.
        if to_calculations:
            normals = self.seg_normals
        else:
            normals = self.zx_normals
        rotated_normals = rotate_vectors(vectors=normals, tau=tau, axis=Iy)

        if return_xy:
            rotated_normals = transform_heliostat(rotated_normals)

        return rotated_normals

    def local_slope(self, weighted=True):

        slope_f = self.curve.as_spline().derivative()
        l_slope = arctan(slope_f(self.seg_pts.T[0])) if weighted else arctan(slope_f(self.contour.T[0]))

        return l_slope

    def ecs_aim_pt(self, rec_aim: array, SunDir: array) -> array:
        """
        This method refers to a SolTrace implementation of the Heliostat object as an SolTrace Element.
        It calculates the aim point of the Element Coordinate System (ECS) as a [x, 0, z] point-array. Since mirrors
        rotate as the sun moves, this aim point changes with the sun direction vector.

        A SolTrace Element has an Element Coordinate System (ECS) attached to it. The z-axis direction of the ECS
        is defined by the vector from the ECS origin (i.e., the heliostat center point) and the ECS aim-point, so that:
        z = ecs_aim_pt - ecs_origin. Of course, these points and vectors are defined in the Stage Coordinate System.
        See SolTrace documentation for more details.

        :param rec_aim: The aim point at the receiver used in the tracking procedure of the heliostat.
        :param SunDir: A 3D vector which represents the Sun vector.

        :return: A [x, 0, z] point-array.
        """

        # Ensure the receiver aim point as a [x, 0, z] point-array
        sm = array([rec_aim[0], 0, rec_aim[-1]])
        ############################################################

        # Calculating the focusing distance of the heliostat
        # The Euclidian distance between the center point and the aim at the receiver
        f = dst(p=self.zx_center, q=sm)
        ###############################################################################

        # Computing the projection of the sun vector in the zx-plane
        # Only this projection is used in the tracking calculations.
        st = nrm(array([SunDir[0], 0, SunDir[-1]]))
        ################################################################

        # The normal vector at mirror center point
        n = nrm(rfx_nrm(i=st, r=self.zx_center - sm))
        #####################################################

        # The Element Coordinate System (ECS) aim-point
        ecs_aim_pt = self.zx_center + f * array([n[0], 0, n[-1]])
        ###########################################################

        return ecs_aim_pt

    def as_soltrace_element(self, name: str, length: float, rec_aim: array, sun_dir: array,
                            optic: OpticalSurface) -> Element:

        aim = array([rec_aim[0], 0, rec_aim[-1]])
        elem = heliostat2soltrace(hel=self, name=name, length=length, rec_aim=aim, sun_dir=sun_dir, optic=optic)

        return elem


class PrimaryField:
    """
    This class represents a primary field of a linear Fresnel collector.

    """

    def __init__(self, heliostats: list):
        """
        :param heliostats: A list of Heliostat objects.

        """

        self.heliostats = []
        for i, hel in enumerate(heliostats):
            if isinstance(hel, Heliostat):
                self.heliostats += [hel]
            else:
                raise f'A non Heliostat instance was inputted. Please, check the {i + 1}-element of the inputted list'

        self.nbr_mirrors = len(self.heliostats)
        self.radius = array([hel.radius for hel in self.heliostats])

        self.widths = zeros(self.nbr_mirrors)
        self.widths[:] = [hel.width for hel in self.heliostats]

        # XY attributes #################################################################
        # These attributes are point and vector arrays with the format [x, y]
        self.centers = zeros(shape=(self.nbr_mirrors, 2))
        self.centers[:] = [hel.center for hel in self.heliostats]

        self.primaries = array([hel.contour for hel in self.heliostats])
        #################################################################################

        # ZX attributes #################################################################
        # These attributes are point and vector arrays with the format [x, 0, z]

        self.zx_centers = zeros(shape=(self.nbr_mirrors, 3))
        self.zx_centers[:] = [hel.zx_center for hel in self.heliostats]

        self.zx_primaries = array([hel.zx_pts for hel in self.heliostats])
        self.normals = array([hel.zx_normals for hel in self.heliostats])

        self.seg_primaries = array([hel.seg_pts for hel in self.heliostats])
        self.seg_normals = array([hel.seg_normals for hel in self.heliostats])
        #################################################################################

    def rotated_mirrors(self, theta_t, aim: array, return_xy=True, to_calculations=True) -> array:

        mirrors = array([hc.rotated_points(theta_t=theta_t, aim=aim,
                                           return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.heliostats])

        return mirrors

    def rotated_normals(self, theta_t: float, aim: array, return_xy=True, to_calculations=True) -> array:

        normals = array([hc.rotated_normals(theta_t=theta_t, aim=aim,
                                            return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.heliostats])

        return normals

    def intercept_factor(self, flat_absorber: Absorber.flat,
                         theta_t: float, theta_l: float,
                         length: float = 120000.,
                         rec_aim: array = None, cum_eff=None, end_losses=False) -> float:

        s1 = flat_absorber.s1
        s2 = flat_absorber.s2

        aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

        gamma = intercept_factor(field=self.seg_primaries, normals=self.seg_normals,
                                 centers=self.zx_centers, widths=self.widths,
                                 theta_t=theta_t, theta_l=theta_l, aim_pt=aim_pt, s1=s1, s2=s2,
                                 length=length, cum_eff=cum_eff, end_losses=end_losses)

        return gamma

    def optical_analysis(self, flat_absorber: Absorber.flat, length: float = 120000.,
                         rec_aim: array = None, cum_eff=None, end_losses=False, symmetric=False, factorized=True):

        if factorized:
            output = factorized_intercept_factor(field=self, flat_absorber=flat_absorber,
                                                 rec_aim=rec_aim, length=length,
                                                 cum_eff=cum_eff, end_losses=end_losses, symmetric=symmetric)
        else:
            output = biaxial_intercept_factor(field=self, flat_absorber=flat_absorber,
                                              rec_aim=rec_aim, length=length,
                                              cum_eff=cum_eff, end_losses=end_losses, symmetric=symmetric)

        return output

    def annual_eta(self, site_data: SiteData, NS: bool,
                   flat_absorber: Absorber.flat, length: float = 120000.,
                   rec_aim: array = None, cum_eff=None, end_losses=False, symmetric=False, factorized=True):

        if factorized:

            transversal_data, longitudinal_data = self.optical_analysis(factorized=True,
                                                                        flat_absorber=flat_absorber, length=length,
                                                                        rec_aim=rec_aim, cum_eff=cum_eff,
                                                                        end_losses=end_losses, symmetric=symmetric)

            eta = annual_eta(transversal_data=transversal_data, longitudinal_data=longitudinal_data,
                             site=site_data, NS=NS)

        else:

            biaxial_data = self.optical_analysis(factorized=False,
                                                 flat_absorber=flat_absorber, length=length,
                                                 rec_aim=rec_aim, cum_eff=cum_eff,
                                                 end_losses=end_losses, symmetric=symmetric)

            eta = biaxial_annual_eta(biaxial_data=biaxial_data, site=site_data, NS=NS)

        return eta

    def acceptance_data(self, theta_t: float,
                        flat_absorber: Absorber.flat,
                        rec_aim: array = None,
                        cum_eff: array = None, lv=0.6, dt=0.1):

        acc_data = acceptance_function(theta_t=theta_t,
                                       field=self, flat_absorber=flat_absorber, rec_aim=rec_aim,
                                       cum_eff=cum_eff, lv=lv, dt=dt)

        return acc_data

    def acceptance_angle(self, theta_t: float,
                         flat_absorber: Absorber.flat,
                         rec_aim: array = None,
                         cum_eff: array = None, lv=0.6, dt=0.1, ref_value=0.9):

        acc_data = self.acceptance_data(theta_t, flat_absorber, rec_aim, cum_eff, lv, dt)

        return acceptance_angle(acceptance_data=acc_data, ref_value=ref_value)

    def to_soltrace(self, rec_aim: array, sun_dir: array, length: float, optic: OpticalSurface) -> list:

        elements = [hel.as_soltrace_element(name=f'Heliostat_{i + 1}', length=length,
                                            rec_aim=rec_aim, sun_dir=sun_dir, optic=optic)
                    for i, hel in enumerate(self.heliostats)]

        return elements


class LFR:

    def __init__(self, primary_field: PrimaryField, flat_absorber: Absorber.flat):

        # Primary field data
        self.field = primary_field  # the PrimaryField object
        self.radius = self.field.radius

        # Flat receiver data
        self.receiver = flat_absorber

    def rotated_mirrors(self, theta_t, aim: array = None, return_xy=True, to_calculations=True):

        aim_pt = mid_point(self.receiver.s1, self.receiver.s2) if aim is None else aim

        mirrors = array([hc.rotated_points(theta_t=theta_t, aim=aim_pt,
                                           return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.field.heliostats])

        return mirrors

    def rotated_normals(self, theta_t: float, aim: array = None, return_xy=True, to_calculations=True):

        aim_pt = mid_point(self.receiver.s1, self.receiver.s2) if aim is None else aim

        normals = array([hc.rotated_normals(theta_t=theta_t, aim=aim_pt,
                                            return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.field.heliostats])

        return normals

    def intercept_factor(self, theta_t: float, theta_l: float,
                         length: float = 120000.,
                         rec_aim: array = None, cum_eff=None, end_losses=False):

        return self.field.intercept_factor(flat_absorber=self.receiver,
                                           theta_t=theta_t, theta_l=theta_l, length=length,
                                           rec_aim=rec_aim, cum_eff=cum_eff, end_losses=end_losses)

    def optical_analysis(self, cum_eff=None, rec_aim: array = None, length: float = 120000.,
                         symmetric=False, end_losses=False, factorized=True):

        return self.field.optical_analysis(flat_absorber=self.receiver, rec_aim=rec_aim, symmetric=symmetric,
                                           length=length, cum_eff=cum_eff, end_losses=end_losses, factorized=factorized)

    def annual_eta(self, site_data: SiteData, NS: bool,
                   length: float = 120000.,
                   rec_aim: array = None, cum_eff=None, end_losses=False, symmetric=False, factorized=True):

        return self.field.annual_eta(flat_absorber=self.receiver, site_data=site_data, NS=NS,
                                     length=length, rec_aim=rec_aim, cum_eff=cum_eff, end_losses=end_losses,
                                     symmetric=symmetric, factorized=factorized)

    def acceptance_data(self, theta_t: float,
                        rec_aim: array = None,
                        cum_eff: array = None, lv=0.6, dt=0.1):

        return self.field.acceptance_data(theta_t=theta_t, flat_absorber=self.receiver,
                                          cum_eff=cum_eff, rec_aim=rec_aim, lv=lv, dt=dt)

    def acceptance_angle(self, theta_t: float,
                         rec_aim: array = None,
                         cum_eff: array = None, lv=0.6, dt=0.1, ref_value=0.9):

        return self.field.acceptance_angle(theta_t=theta_t, flat_absorber=self.receiver, rec_aim=rec_aim,
                                           cum_eff=cum_eff, lv=lv, dt=dt, ref_value=ref_value)

    def specific_cost(self):
        return economic_analysis(field=self.field, s1=self.receiver.s1, s2=self.receiver.s2, dH=0.5)

    def export_geometry_data(self, file_path: Path = None, file_name: str = None):

        out_dic = {"centers": self.field.centers, "widths": self.field.widths, "radius": self.radius,
                   "rec_center": self.receiver.center, "rec_width": self.receiver.width,
                   "rec_axis": self.receiver.axis}

        return dic2json(d=out_dic, file_path=file_path, file_name=file_name)

########################################################################################################################
# General auxiliary functions ##########################################################################################


def transform_vector(v: array):
    """
    This function transforms a vector-array (or a point-array) of the kind [x, y] into a [x, 0, z], and vice-versa.

    :param v: A point-array or a vector-array.

    :return: A point-array or a vector-array.
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

    :param hel: An array of point-arrays (or vector-arrays).

    :return: An array of point-arrays (or vector-arrays).
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
    assert center.shape[0] == 2 or center.shape[0] == 3, ValueError("The center point is not a [x, y] or [x, 0, z] "
                                                                    "point array.")
    assert points.shape[1] == center.shape[0], ValueError("Dimensions of 'points' and 'center' are not equal.")

    translated_pts = points - center

    if center.shape[0] == 3 and (axis == array([0, 1, 0])).all():
        rm = R(alpha=tau, v=axis)
    elif center.shape[0] == 2 and (axis == array([0, 0, 1])).all():
        rm = R(alpha=tau)
    else:
        raise ValueError('Arrays shape and rotating axis are not properly inputted')

    rotated_pts = rm.dot(translated_pts.T).T + center
    return rotated_pts


def rotate_vectors(vectors: array, tau: float, axis: array):
    rm = R(alpha=tau, v=axis)
    rotated_vec = rm.dot(vectors.T).T

    return rotated_vec


########################################################################################################################
# LFR design functions #################################################################################################

def design_cylindrical_heliostat(hc: array, w: float, rc: float, nbr_pts=51):
    """
    This function returns the surface points of a cylindrical shape heliostat.
    This set of points define the heliostat contour.

    Units should be in millimeters to be coherent with the classes, methods, and functions presented in this module.

    :param hc: Heliostat center point
    :param w: Heliostat width
    :param rc: Heliostat cylindrical curvature absorber_radius
    :param nbr_pts: Number of points in which the surface is discretized

    :return: An array of points that define the contour of the cylindrical surface.
    """

    ####################################################################################################################
    # Ensure an odd number of point to represent the heliostat surface #################################################
    # This is needed to ensure that the center of the mirror is also an element in the array of points which describes
    # the heliostat surface.
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    ####################################################################################################################

    ####################################################################################################################
    # Calculations #####################################################################################################

    # The array of x-values which the heliostat ranges.
    x_range = linspace(start=-0.5 * w, stop=+0.5 * w, num=n_pts)

    # Ensure that the center point is a XY array point.
    center = array([hc[0], hc[-1]])

    # the function which analytically describes the cylindrical surface which comprises the heliostat
    def y(x): return -sqrt(rc ** 2 - x ** 2) + rc

    # the computation of the points which discretize the heliostat surface
    hel_pts = array([[x, y(x)] for x in x_range]) + center
    ####################################################################################################################

    return hel_pts


def design_flat_heliostat(hc: array, w: float, nbr_pts: int):
    """
    This function returns the surface points of a flat shape heliostat.
    This set of points define the heliostat contour.

    Units should be in millimeters to be coherent with the classes, methods, and functions presented in this module.

    :param hc: heliostat center point
    :param w: heliostat width
    :param nbr_pts: number of point to parametrize.

    :return: This function returns a list of points from the function of the heliostat.
    """

    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    center = array([hc[0], hc[-1]])

    hel_pts = zeros(shape=(n_pts, 2))
    hel_pts[:, 0] = linspace(start=-0.5 * w, stop=0.5 * w, num=n_pts) + center[0]

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


def primary_field_design(lfr_geometry, rec_aim: array, curvature_design, nbr_pts=121) -> PrimaryField:

    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

    centers = lfr_geometry.centers
    widths = lfr_geometry.widths

    sm = array([rec_aim[0], rec_aim[-1]])
    heliostats = []

    for w, hc in zip(widths, centers):

        if isinstance(curvature_design, (float, int)):
            rr = rabl_curvature(center=hc,
                                aim=sm,
                                theta_d=curvature_design)
        elif curvature_design == 'SR':
            rr = 2 * dst(hc, sm)
        elif curvature_design == 'flat':
            rr = 0

        else:
            raise ValueError("Design position must be an Angle or 'SR'")

        hel = Heliostat(center=hc,
                        width=w,
                        radius=rr,
                        nbr_pts=n_pts)

        heliostats += [hel]

    primary_field = PrimaryField(heliostats=heliostats)

    return primary_field


class uniform_lfr_geometry:

    def __init__(self, name: str, mirror_width: float, nbr_mirrors: int,
                 total_width: float = None, center_distance: float = None):
        self.name = name

        self.mirror_width = abs(mirror_width)
        self.nbr_mirrors = abs(nbr_mirrors)
        self.widths = ones(self.nbr_mirrors) * self.mirror_width

        self.centers = uniform_centers(mirror_width=self.mirror_width, nbr_mirrors=self.nbr_mirrors,
                                       total_width=total_width, center_distance=center_distance)

        self.center_distance = dst(p=self.centers[0], q=self.centers[1])
        self.total_width = dst(p=self.centers[0], q=self.centers[-1]) + self.mirror_width

        self.filling_factor = self.widths.sum() / self.total_width

        self.left_edge = array([-0.5 * self.total_width, 0])
        self.right_edge = array([+0.5 * self.total_width, 0])

        self.primary_field = None
        self.absorber = None
        self.secondary = None

    def export_geometry(self, file_path):
        file_full_path = Path(file_path, f'{self.name}_geometry.json')

        d = {'variable_name': self.name, 'mirror_width': self.mirror_width, 'nbr_mirrors': self.nbr_mirrors,
             'total_width': self.total_width, 'center_distance': self.center_distance,
             'units': 'millimeters'}

        with open(file_full_path, 'w') as file:
            json.dump(d, file)

        return file_full_path

    def design_primary_field(self, receiver_aim: array, curvature_design, nbr_pts: int):

        return primary_field_design(lfr_geometry=self,
                                    rec_aim=receiver_aim,
                                    curvature_design=curvature_design,
                                    nbr_pts=nbr_pts)


########################################################################################################################
# LFR optical analysis functions #######################################################################################

def angular_position(center: array, aim: array):
    """
    This function calculates the angular position of a primary mirror of an LFC concentrator, defined by the
    'center' point, regarding the 'aim' point at the receiver.
    It considers a [x,0,z] point and vector-arrays. The LFC transversal plane is the ZX plane.

    :param center: a point-array, in millimeters.
    :param aim: a point-array, in millimeters.

    :return: an angle, in radians
    """

    Iz = array([0, 0, 1])

    sm = array([aim[0], 0, aim[-1]])
    hc = array([center[0], 0, center[-1]])

    aim_vector = sm - hc
    lamb = sign(cross(aim_vector, Iz)[1]) * ang(aim_vector, Iz)

    return lamb


def tracking_angle(center: array, aim: array, theta_t: float, degree=True):
    """


    :param center:
    :param aim:
    :param theta_t:
    :param degree:
    :return:
    """

    lamb = angular_position(center=center, aim=aim)
    tau = 0.5 * (deg2rad(theta_t) - lamb) if degree else 0.5 * (theta_t - lamb)

    return tau


def factorized_intercept_factor(field: PrimaryField, flat_absorber: Absorber.flat,
                                length: float, rec_aim: array = None,
                                cum_eff: array = None, end_losses=False, symmetric=False,
                                dt=5.):

    step = abs(dt)

    tt0 = 0. if symmetric else -90.
    transversal_angles = arange(start=tt0, stop=90. + step, step=step)
    longitudinal_angles = arange(start=0., stop=90. + step, step=step)

    s1, s2 = flat_absorber.s1, flat_absorber.s2
    aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

    transversal_values = [intercept_factor(theta_t=theta, theta_l=0.,
                                           field=field.seg_primaries, normals=field.seg_normals,
                                           centers=field.zx_centers, widths=field.widths,
                                           s1=s1, s2=s2, aim=aim_pt,
                                           length=length,
                                           cum_eff=cum_eff, end_losses=end_losses)

                          for theta in transversal_angles]

    longitudinal_values = [intercept_factor(theta_t=0., theta_l=theta,
                                            field=field.seg_primaries, normals=field.seg_normals,
                                            centers=field.zx_centers, widths=field.widths,
                                            s1=s1, s2=s2, aim=aim_pt,
                                            length=length,
                                            cum_eff=cum_eff, end_losses=end_losses)

                           for theta in longitudinal_angles]

    transversal_data = zeros(shape=(transversal_angles.shape[0], 2))
    transversal_data.T[:] = transversal_angles, transversal_values

    longitudinal_data = zeros(shape=(longitudinal_angles.shape[0], 2))
    longitudinal_data.T[:] = longitudinal_angles, longitudinal_values

    return transversal_data, longitudinal_data


def biaxial_intercept_factor(field: PrimaryField, flat_absorber: Absorber.flat,
                             length: float, rec_aim: array = None,
                             cum_eff: array = None, end_losses=False, symmetric=False,
                             dt=5.):

    s1, s2 = flat_absorber.s1, flat_absorber.s2
    aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

    step = abs(dt)
    tt0 = 0. if symmetric else -90.

    angles_list = array([[x, y] for x in arange(tt0, 90. + step, step) for y in arange(0., 90. + step, step)])

    gamma_values = [intercept_factor(theta_t=theta[0], theta_l=theta[1],
                                     field=field.seg_primaries, normals=field.seg_normals,
                                     centers=field.zx_centers, widhts=field.widths,
                                     s1=s1, s2=s2, aim_pt=aim_pt,
                                     length=length,
                                     cum_eff=cum_eff, end_losses=end_losses)

                    for theta in angles_list]

    biaxial_data = zeros(shape=(angles_list.shape[0], 2))
    biaxial_data.T[:] = angles_list, gamma_values

    return biaxial_data


def annual_eta(transversal_data: array, longitudinal_data: array, site: SiteData, NS=True):
    """
    A function to compute the annual optical efficiency of a linear concentrator.

    :param transversal_data: Transversal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param longitudinal_data: Longitudinal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param site: a SiteData object with the TMY data of the location.
    :param NS: A sign to inform whether a NS (North-South) or EW (East-West) mounting was considered.

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


def acceptance_function(theta_t: float,
                        field: PrimaryField, flat_absorber: Absorber.flat,
                        rec_aim: array = None,
                        cum_eff: array = None, lv=0.6, dt=0.1) -> array:

    s1, s2 = flat_absorber.s1, flat_absorber.s2

    aim_pt = mid_point(s1, s2) if rec_aim is None else rec_aim

    acceptance_data = acceptance_analysis(theta_t=theta_t,
                                          field=field.seg_primaries, normals=field.seg_normals,
                                          centers=field.zx_centers, widths=field.widths,
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
# Soltrace functions ###################################################################################################

def heliostat2soltrace(hel: Heliostat, name: str,
                       sun_dir: array, rec_aim: array, length: float,
                       optic: OpticalSurface) -> Element:
    """
    This function considers a Heliostat object and returns it as a Soltrace Element object (see soltracepy.Element).

    :param hel: Heliostat object.
    :param name: A variable_name of the element to be inserted in the 'comment' argument.
    :param sun_dir: A 3D vector-array which represents the sun direction.
    :param rec_aim: The aiming point at the receiver used as reference for the tracking, a point-array.
    :param length: The length of the heliostat.
    :param optic: The optical property to associate, a SolTrace OpticalSurface (soltracepy.OpticalSurface).

    :return: It returns an Element object.

    [1] Santos et al., 2023. https://doi.org/10.1016/j.renene.2023.119380.
    """

    # The ecs_origin and ecs_aim_pt of the Element Coordinate System #####################
    # Remember that SolTrace uses the units in meters
    # the linear_fresnel module is defined in millimeters
    origin = hel.zx_center / 1000
    aim_pt = hel.ecs_aim_pt(rec_aim=rec_aim, SunDir=sun_dir) / 1000
    # A vector from the ecs_origin to the ecs_aim_pt defines the z-axis of the ECS
    ######################################################################################

    # Creating auxiliary variables to implement #############################
    L = length / 1000  # convert to meters
    aperture = list([0] * 9)
    surface = list([0] * 9)
    #########################################################################

    # New implementation -- from 08-Mar-2023 ############################################
    if hel.shape == 'flat':
        # Defining the aperture of the cylindrical mirror #########
        aperture[0:3] = 'r', hel.width / 1000, L
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
        aperture[0:3] = 'r', hel.width / 1000, L

        # Defining the surface
        rc = hel.radius / 1000
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

    elem = Element(name=name, ecs_origin=origin, ecs_aim_pt=aim_pt, z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=True)

    return elem


def flat_absorber2soltrace(geometry: Absorber.flat, name: str, optic: OpticalSurface, length: float,
                           upwards=False) -> list:
    # Converting units to meters ####
    w = geometry.width / 1000
    L = length / 1000
    #################################

    # Setting the ecs_origin and aim-pt of the Element Coordinate System ###############################
    # Origin of the ECS
    hc = array([geometry.center[0], 0, geometry.center[-1]]) / 1000

    # Aim-pt of the ECS
    # To define the front side of the Element (positive direction of the z-axis of the ECS)
    if upwards:
        aim_pt = geometry.center + R(pi / 2).dot(geometry.s2 - geometry.s1)
    else:
        aim_pt = geometry.center + R(-pi / 2).dot(geometry.s2 - geometry.s1)

    aim_pt = array([aim_pt[0], 0, aim_pt[-1]]) / 1000
    #####################################################################################################

    # Setting the aperture and surface ####################
    # Aperture
    aperture = list([0] * 9)
    aperture[0:3] = 'r', w, L
    ############################

    # Surface
    surface = list([0] * 9)
    surface[0] = 'f'
    ############################
    #######################################################

    elem = Element(name=name, ecs_origin=hc.round(6), ecs_aim_pt=aim_pt.round(6), z_rot=0,
                   aperture=aperture, surface=surface,
                   optic=optic, reflect=True)

    return [elem]


def tube2soltrace(tube: Absorber.tube, name: str, optic: OpticalSurface, length: float, reflect=True) -> list:
    # Converting the units to meters
    r = tube.radius / 1000
    L = length / 1000
    ################################

    # Setting the ecs_origin and aim-pt of the Element Coordinate System ########################
    # Origin of the ECS
    # For a tube element, the ecs_origin of the ECS is at the bottom of the tube (see SolTrace illustration)
    hc = transform_vector(tube.center) - array([0, 0, tube.radius])
    hc = hc / 1000

    # Aim-pt of the ECS
    aim_pt = hc + array([0, 0, 2 * tube.radius]) / 1000
    ############################################################################################

    # Setting the aperture and surface #########################################################
    # Aperture
    aperture = list([0] * 9)
    aperture[0], aperture[3] = 'l', L
    #####################################
    # Surface
    surface = list([0] * 9)
    surface[0], surface[1] = 't', 1 / r
    #####################################
    ############################################################################################

    elem = Element(name=name, ecs_origin=hc.round(6), ecs_aim_pt=aim_pt.round(6), z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=reflect)

    return [elem]


def trapezoidal2soltrace(geometry: Secondary.trapezoidal, name: str, length: float, optic: OpticalSurface) -> list:
    # ToDo: Implement by using the soltracepy.flat_element function to reduce the number of code lines

    Iy = array([0, 1, 0])
    L = length / 1000

    # The width of the flat elements that define the trapeze ########
    # SolTrace works in meters units
    w_r = dst(p=geometry.back_right, q=geometry.ap_right) / 1000
    w_l = dst(p=geometry.back_left, q=geometry.ap_left) / 1000
    w_b = dst(p=geometry.back_left, q=geometry.back_right) / 1000
    #################################################################

    # New implementation ############################################################################################
    # Calculating the Element Coordinate System and their aim-points #########################
    r_hc = transform_vector(mid_point(p=geometry.back_right, q=geometry.ap_right)) / 1000
    r_ab = transform_vector(geometry.ap_right - geometry.back_right) / 1000
    r_aim = R(pi / 2, Iy).dot(r_ab) + r_hc

    l_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.ap_left)) / 1000
    l_ba = transform_vector(geometry.back_left - geometry.ap_left) / 1000
    l_aim = R(pi / 2, Iy).dot(l_ba) + l_hc

    b_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.back_right)) / 1000
    b_bb = transform_vector(geometry.back_right - geometry.back_left) / 1000
    b_aim = R(pi / 2, Iy).dot(b_bb) + b_hc
    ############################################################################################

    # SolTrace Elements ############################################################################
    right_side_element = flat_element(name=f'{name}_right', ecs_origin=r_hc, ecs_aim_pt=r_aim, width=w_r,
                                      length=L, optic=optic, reflect=True, enable=True)

    back_side_element = flat_element(name=f'{name}_back', ecs_origin=b_hc, ecs_aim_pt=b_aim, width=w_b,
                                     length=L, optic=optic, reflect=True, enable=True)

    left_side_element = flat_element(name=f'{name}_left', ecs_origin=l_hc, ecs_aim_pt=l_aim, width=w_l,
                                     length=L, optic=optic, reflect=True, enable=True)
    #################################################################################################

    ###############################################################################################################
    ###############################################################################################################

    ###############################################################################################################
    # Old implementation ##########################################################################################
    # # Setting the aperture #########
    # aperture_r, aperture_l, aperture_b = list([0] * 9), list([0] * 9), list([0] * 9)
    # aperture_r[0:3] = 'r', w_r, L
    # aperture_l[0:3] = 'r', w_l, L
    # aperture_b[0:3] = 'r', w_b, L
    # ################################
    #
    # # Setting the surface ##########
    # surface = list([0] * 9)
    # surface[0] = 'f'
    # ################################
    #
    # r_hc = transform_vector(mid_point(p=geometry.back_right, q=geometry.ap_right)) / 1000
    # r_ab = transform_vector(geometry.ap_right - geometry.back_right) / 1000
    # r_aim = R(pi/2, Iy).dot(r_ab) + r_hc
    #
    # l_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.ap_left)) / 1000
    # l_ba = transform_vector(geometry.back_left - geometry.ap_left) / 1000
    # l_aim = R(pi/2, Iy).dot(l_ba) + l_hc
    #
    # b_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.back_right)) / 1000
    # b_bb = transform_vector(geometry.back_right - geometry.back_left) / 1000
    # b_aim = R(pi/2, Iy).dot(b_bb) + b_hc
    #
    # right_side_element = Element(name=f"{name}_right", ecs_origin=r_hc, ecs_aim_pt=r_aim, z_rot=0,
    #                              aperture=aperture_r, surface=surface, optic=optic, reflect=True, enable=True)
    #
    # left_side_element = Element(name=f"{name}_left", ecs_origin=l_hc, ecs_aim_pt=l_aim, z_rot=0,
    #                             aperture=aperture_l, surface=surface, optic=optic, reflect=True, enable=True)
    #
    # back_side_element = Element(name=f"{name}_back", ecs_origin=b_hc, ecs_aim_pt=b_aim, z_rot=0,
    #                             aperture=aperture_b, surface=surface, optic=optic, reflect=True, enable=True)
    ###############################################################################################################
    ###############################################################################################################

    return [right_side_element, back_side_element, left_side_element]


def evacuated_tube2soltrace(geometry: Absorber.evacuated_tube, name: str, length: float,
                            outer_cover_optic: OpticalSurface, inner_cover_optic: OpticalSurface,
                            absorber_optic: OpticalSurface):
    # The outer cover Element
    outer_cover = tube2soltrace(tube=geometry.outer_tube, name=f'{name}_outer_cover', length=length,
                                optic=outer_cover_optic, reflect=False)[0]

    # The inner cover Element
    inner_cover = tube2soltrace(tube=geometry.inner_tube, name=f'{name}_inner_cover', length=length,
                                optic=inner_cover_optic, reflect=False)[0]

    # The absorber tube Element
    absorber_tube = tube2soltrace(tube=geometry.absorber_tube, name=f'{name}_absorber', length=length,
                                  optic=absorber_optic, reflect=True)[0]

    return [outer_cover, inner_cover, absorber_tube]


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Functions related to an economic analysis ############################################################################
# See Moghimi et al. (2017), Solar Energy, Doi: 10.1016/J.SOLENER.2017.06.001


def mirror_cost(w: float, Cmo=30.5):
    return Cmo * (w / 0.5)


def mirror_gap_cost(g: float, Cdo=11.5):
    return Cdo * (g / 0.01)


def elevation_cost(od: float, Ceo=19.8, Nt=1):
    eta_ci = array([1.4, 1, 1])
    Ci = array([14.2, 0.9, 4.6])

    num = 0
    for i in range(len(Ci)):
        num += (Ci[i] * Nt / Ceo) * (od / 0.219) ** eta_ci[i]

    eta_ce = log(num) / (log(od / 0.219))

    return Ceo * power(od / 0.219, eta_ce)


def receiver_cost(od: float, Cro=654.0, Nt=1):
    eta_ci = array([2, 0.9, 0.7, 1.4, 0.6, 0.6])
    Ci = array([116.2, 56.6, 116.4, 136.5, 26.4, 112.6])

    num = 0
    for i in range(len(Ci)):
        num += (Ci[i] * Nt / Cro) * (od / 0.219) ** eta_ci[i]

    eta_cr = log(num) / (log(od / 0.219))

    return Cro * power(od / 0.219, eta_cr)


def economic_analysis(field: PrimaryField, s1: array, s2: array, dH=1.0):
    od = dst(s1, s2) / pi / 1.0e3

    # if len(field) == 2:
    #     heliostats, _ = field
    # else:
    #     heliostats = field

    sm = mid_point(s1, s2)
    H = sm[-1] / 1.0e3

    # widths = array(heliostats_widths(field=heliostats)) / 1.0e3
    widths = field.widths
    widths_cost = array([mirror_cost(w=w) for w in widths])

    # centers = primaries_center(field=heliostats)
    centers = field.centers
    s = [dst(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
    gaps = array([s[i] - 0.5 * (widths[i] + widths[i + 1]) for i in range(len(centers) - 1)]) / 1.0e3
    gaps_cost = array([mirror_gap_cost(g) for g in gaps])

    Ce = elevation_cost(od=od)
    Cr = receiver_cost(od=od)

    specific_cost = (sum(widths_cost, 0) + Ce * (H + dH) + sum(gaps_cost, 0) + Cr) / sum(widths, 0)

    return specific_cost

########################################################################################################################
########################################################################################################################
