# -*- coding: utf-8 -*-
"""
Created on Thu, Dec 3,2020 15:47:43
New version: Feb 8 2022 09:00:55
@author: André Santos (andrevitoras@gmail.com / avas@uevora.pt)

This module holds the functions to computes the optic-geometric performance of linear Fresnel concentrators.

It considers a 3D model of the Linear Fresnel Collector (LFC), where the ZX plane is the transversal plane of the
concentrator, and the y-axis is the longitudinal direction.

(...).
"""

from pathlib import Path

from matplotlib import pyplot as plt
from numpy import arctan, log, ndarray

from niopy.geometric_transforms import nrm
from niopy.plane_curves import PlaneCurve
from niopy.reflection_refraction import rfx_nrm

from pysoltrace import PySolTrace
from pysoltrace.geometries import flat_element as py_flat_element, tubular_element, linear_fresnel_mirror

from scopy.linear_fresnel.analysis import *
from scopy.linear_fresnel.design import *

from scopy.linear_fresnel.optical_method import intercept_factor, acceptance_analysis

from scopy.nio_concentrators import symmetric_cpc2tube, cpc_type, symmetric_cpc2evacuated_tube, \
    symmetric_cec2evacuated_tube

from scopy.sunlight import SiteData
from soltracepy import OpticalSurface, Element
from soltracepy.auxiliary import flat_element

from utils import dic2json, plot_line


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

        def to_pysoltrace(self,
                          id_number: int,
                          parent_stage: PySolTrace.Stage,
                          length: float,
                          optic: PySolTrace.Optics,
                          upwards=False):

            ecs_origin = array([self.center[0], 0, self.center[-1]])

            if upwards:
                aim = self.center + R(pi/2).dot(nrm(self.s2 - self.s1)) * self.width
            else:
                aim = self.center + R(-pi/2).dot(nrm(self.s2 - self.s1)) * self.width

            ecs_aim = array([aim[0], 0, aim[-1]])
            element_object = py_flat_element(width=self.width / 1000.,
                                             length=length/1000.,
                                             ecs_origin=ecs_origin/1000.,
                                             ecs_aim=ecs_aim/1000.,
                                             parent_stage=parent_stage, id_number=id_number, optic=optic)

            return element_object

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

        def to_pysoltrace(self, length: float,
                          parent_stage: PySolTrace.Stage, id_number: int,
                          optic: PySolTrace.Optics):

            ecs_origin = array([self.center[0], 0, self.center[-1] - self.radius])
            ecs_aim = ecs_origin + array([0, 0, 2*self.radius])

            return tubular_element(tube_radius=self.radius/1000, tube_length=length/1000,
                                   ecs_origin=ecs_origin/1000, ecs_aim=ecs_aim/1000,
                                   parent_stage=parent_stage, id_number=id_number, optic=optic)

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

        def to_pysoltrace(self, length: float,
                          outer_cover_optic: PySolTrace.Optics,
                          inner_cover_optic: PySolTrace.Optics,
                          absorber_optic: PySolTrace.Optics,
                          parent_stage: PySolTrace.Stage, id_number: int) -> list:

            optics = [outer_cover_optic, inner_cover_optic, absorber_optic]

            elements = [tube.to_pysoltrace(length=length, parent_stage=parent_stage,
                                           id_number=id_number + i, optic=optics[i])
                        for i, tube in enumerate([self.outer_tube, self.inner_tube, self.absorber_tube])]

            return elements


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


class PrimaryMirror:
    """
    The class PrimaryMirror aims to represent a linear Fresnel primary mirror. It can be flat or cylindrical.

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

        # The basic attributes of a PrimaryMirror object #####
        self.width = abs(width)
        self.radius = abs(radius)
        self.center = array([center[0], center[-1]])
        ##################################################

        # Defining an attribute for the shape of the heliostats ###
        # It can be flat or cylindrical
        if self.radius <= 4000.:
            self.shape = 'flat'
        else:
            self.shape = 'cylindrical'
        ############################################################

        # Ensure an odd number of points to discretize the heliostat contour surface ####
        # In this way, the center point is part of the set of points.
        self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
        #################################################################################

        ################################################################################################################
        # PrimaryMirror attributes that are defined as [x,y] point-arrays/vector-arrays ################################

        # The design of the point-arrays which defines the heliostat contour
        # These point-arrays consider the mirror at the horizontal position
        # The edges and the center are in this set.
        if self.shape == 'cylindrical':
            self.contour = design_cylindrical_heliostat(center=self.center, width=self.width,
                                                        radius=self.radius, nbr_pts=self.n_pts)
        else:
            self.contour = design_flat_heliostat(center=self.center, width=self.width,
                                                 nbr_pts=self.n_pts)

        # Attribute that holds the PrimaryMirror object as a PlaneCurve object.
        self.curve = self.as_plane_curve()

        # The normal vectors at the heliostat surface points.
        self.normals = self.curve.normals2surface()

        #############################################################################################################
        #############################################################################################################

        #########################################################################################################
        # PrimaryMirror attributes that are defined as [x, 0, z] point-arrays/vector-arrays #########################
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

        return seg_points.round(10), seg_normals.round(10)

    def as_plane_curve(self):
        """
        A method to return the PrimaryMirror object as a PlaneCurve object defined by the contour point.

        :return: It returns the xy points of the heliostat as a PlaneCurve object.
        """
        return PlaneCurve(curve_pts=self.contour, curve_center=self.center)

    def angular_position(self, rec_aim: array) -> float:
        """
        This method calculates the heliostat angular position regarding a tracking aim-point at the receiver.
        It assumes the ZX-plane as the transversal plane, i.e., the one which defines the linear Fresnel geometry.

        :param rec_aim: The aim-point at the receiver, a point-array.

        :return: The heliostat angular position, in radians.
        """

        # Old version ##################################################
        # This version considers the transversal plane as the XY plane.

        # sm = array([aim[0], aim[-1]])
        # vf = sm - self.center
        # lamb = sign(self.center[0]) * ang(vf, array([0, 1])).rad
        ################################################################

        # New version in 2022-11-08 #########################################
        # This version considers the transversal plane as the ZX plane.
        # Therefore, a different equation is used.
        lamb = angular_position(center=self.center, rec_aim=rec_aim)
        #####################################################################

        return lamb

    def tracking_angle(self, rec_aim: array, theta_t: float) -> float:
        """
        This method calculates the heliostat tracking angle regarding an aim-point at the receiver for a particular
        value of transversal incidence angle (in degrees).

        It assumes the ZX-plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param rec_aim: The aim-point at the receiver, a point-array.
        :param theta_t: The transversal incidence angle, in degrees.

        :return: The heliostat tracking angle, in radians.
        """

        theta_t_rad = theta_t * pi / 180.
        lamb = self.angular_position(rec_aim=rec_aim)

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
            tau = self.tracking_angle(rec_aim=aim, theta_t=theta_t)

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
        tau = self.tracking_angle(rec_aim=aim, theta_t=theta_t)

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
        This method refers to a SolTrace implementation of the PrimaryMirror object as an SolTrace Element.
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

        return ecs_aim_pt.round(6)

    def as_soltrace_element(self, name: str, length: float, rec_aim: array, sun_dir: array,
                            optic: OpticalSurface) -> Element:

        aim = array([rec_aim[0], 0, rec_aim[-1]])
        elem = heliostat2soltrace(hel=self, name=name, length=length, rec_aim=aim, sun_dir=sun_dir, optic=optic)

        return elem

    def to_pysoltrace(self,
                      sun_dir: array, rec_aim: array, length: float,
                      parent_stage: PySolTrace.Stage, id_number: int,
                      optic: PySolTrace.Optics):

        ecs_aim = self.ecs_aim_pt(rec_aim=rec_aim, SunDir=sun_dir)

        element = linear_fresnel_mirror(width=self.width/1000., length=length/1000., radius=self.radius/1000.,
                                        parent_stage=parent_stage, id_number=id_number,
                                        ecs_origin=self.zx_center/1000., ecs_aim=ecs_aim / 1000.,
                                        optic=optic)
        return element


class PrimaryField:
    """
    This class represents a primary field of a linear Fresnel collector.

    """

    def __init__(self, heliostats: list):
        """
        :param heliostats: A list of PrimaryMirror objects.

        """

        # An attribute for the list of PrimaryMirror objects that comprises the field ##################################
        # if a non PrimaryMirror object is added, an error will be raised.
        self.heliostats = []
        for i, hel in enumerate(heliostats):
            if isinstance(hel, PrimaryMirror):
                self.heliostats += [hel]
            else:
                raise f'A non PrimaryMirror instance was inputted. ' \
                      f'Please, check the {i + 1}-element of the inputted list'
        ################################################################################################################

        # Auxiliary attributes ############################################
        self.nbr_mirrors = len(self.heliostats)
        self.radius = array([hel.radius for hel in self.heliostats])

        self.widths = zeros(self.nbr_mirrors)
        self.widths[:] = [hel.width for hel in self.heliostats]
        ###################################################################

        # XY attributes #################################################################
        # These attributes are point and vector arrays with the format [x, y]
        self.centers = zeros(shape=(self.nbr_mirrors, 2))
        self.centers[:] = [hel.center for hel in self.heliostats]

        # the shifts... That is, the distance between consecutive center points
        self.shifts = array([self.centers[i][0] - self.centers[i + 1][0]
                             for i in range(len(self.centers) - 1)])

        self.gaps = array([self.shifts[i] - 0.5 * (self.widths[i] + self.widths[i + 1])
                           for i in range(len(self.centers) - 1)])

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
                                 theta_t=theta_t, theta_l=theta_l, aim=aim_pt, s1=s1, s2=s2,
                                 length=length, cum_eff=cum_eff, end_losses=end_losses)

        return gamma

    def optical_analysis(self, flat_absorber: Absorber.flat, length: float = 120000.,
                         rec_aim: array = None, cum_eff=None, end_losses=False, symmetric=False, factorized=True):

        if factorized:
            output = factorized_intercept_factor(field=self.seg_primaries, normals=self.seg_normals,
                                                 centers=self.zx_centers, widths=self.widths,
                                                 s1=flat_absorber.s1, s2=flat_absorber.s2,
                                                 rec_aim=rec_aim, length=length,
                                                 cum_eff=cum_eff, end_losses=end_losses, symmetric=symmetric)
        else:
            output = biaxial_intercept_factor(field=self.seg_primaries, normals=self.seg_normals,
                                              centers=self.zx_centers, widths=self.widths,
                                              s1=flat_absorber.s1, s2=flat_absorber.s2,
                                              rec_aim=rec_aim, length=length,
                                              cum_eff=cum_eff, end_losses=end_losses, symmetric=symmetric)

        return output

    def annual_eta(self, site_data: SiteData, NS: bool,
                   flat_absorber: Absorber.flat,
                   length: float = 120000.,
                   rec_aim: array = None,
                   cum_eff=None,
                   end_losses=False,
                   symmetric=False,
                   factorized=True):

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
                                       field=self.seg_primaries, normals=self.seg_normals,
                                       centers=self.zx_centers, widths=self.widths,
                                       s1=flat_absorber.s1, s2=flat_absorber.s2,
                                       rec_aim=rec_aim, cum_eff=cum_eff,
                                       lv=lv, dt=dt)

        return acc_data

    def acceptance_angle(self, theta_t: float,
                         flat_absorber: Absorber.flat,
                         rec_aim: array = None,
                         cum_eff: array = None, lv=0.6, dt=0.1, ref_value=0.9):

        acc_data = self.acceptance_data(theta_t, flat_absorber, rec_aim, cum_eff, lv, dt)

        return acceptance_angle(acceptance_data=acc_data, ref_value=ref_value)

    def to_soltrace(self, rec_aim: array or list, sun_dir: array, length: float, optic: OpticalSurface) -> list:

        if isinstance(rec_aim, (list, ndarray)) and len(rec_aim) == self.centers.shape[0]:
            tracking_points = rec_aim
        else:
            tracking_points = [rec_aim] * self.centers.shape[0]

        elements = [hel.as_soltrace_element(name=f'Heliostat_{i + 1}', length=length,
                                            rec_aim=a, sun_dir=sun_dir, optic=optic)
                    for i, (hel, a) in enumerate(zip(self.heliostats, tracking_points))]

        return elements

    def plot_primaries(self, theta_t, rec_aim: array,
                       color='black', lw=1.5, support_size=0.0):

        mirrors_to_plot = self.rotated_mirrors(theta_t=theta_t, aim=rec_aim,
                                               return_xy=True, to_calculations=False)
        delta_h = abs(support_size)

        [plt.plot(*hel.T, color=color, lw=lw, label='Primary mirrors' if i == 0 else None)
         for i, hel in enumerate(mirrors_to_plot)]

        if delta_h > 0:
            [plt.plot(*plot_line(a=hc, b=hc + array([0, -delta_h])), color=color, lw=lw)
             for i, hc in enumerate(self.centers)]

        return None


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
# Soltrace functions ###################################################################################################


def heliostat2soltrace(hel: PrimaryMirror, name: str,
                       sun_dir: array, rec_aim: array, length: float,
                       optic: OpticalSurface) -> Element:
    """
    This function considers a PrimaryMirror object and returns it as a Soltrace Element object (see soltracepy.Element).

    :param hel: PrimaryMirror object.
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
        aperture[0:3] = 'r', round(hel.width / 1000, 4), round(L, 4)

        # Defining the surface
        rc = hel.radius / 1000
        # The 'c' factor is the parabola's gradient, as defined in SolTrace.
        c = 1 / rc
        surface[0:2] = 'p', round(c, 4)
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

    elem = Element(name=name, ecs_origin=origin.round(4), ecs_aim_pt=aim_pt.round(4), z_rot=0,
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
    aim_pt = hc + array([0, 0, tube.radius]) / 1000
    ############################################################################################

    # Setting the aperture and surface #########################################################
    # Aperture
    aperture = list([0] * 9)
    aperture[0], aperture[3] = 'l', round(L, 4)
    #####################################
    # Surface
    surface = list([0] * 9)
    surface[0], surface[1] = 't', round(1 / r, 4)
    #####################################
    ############################################################################################

    elem = Element(name=name, ecs_origin=hc.round(6), ecs_aim_pt=aim_pt.round(6), z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=reflect)

    return [elem]


def trapezoidal2soltrace(geometry: Secondary.trapezoidal, name: str, length: float, optic: OpticalSurface) -> list:

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


def economic_analysis(field: PrimaryField, s1: array, s2: array, dH=1.0):
    # ToDo: Review Mertins [1] and Moghimi et al. [2] to implement documentation and comments.
    """
    This function (...)

    :param field:
    :param s1:
    :param s2:
    :param dH:
    :return:

    References:
    [1] Mertins, M., 2009. Technische und wirtschaftliche Analyse von horizontalen Fresnel-Kollektoren.
        University of Karlsruhe, PhD Thesis.
    [2] Moghimi et al.,2017. Solar Energy 153, 655–678. https://doi.org/10.1016/J.SOLENER.2017.06.001.
    """

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

    absorber_diameter = dst(s1, s2) / pi / 1.0e3

    sm = mid_point(s1, s2)
    H = sm[-1] / 1.0e3

    # widths = array(heliostats_widths(field=heliostats)) / 1
    # 0e3
    widths = field.widths
    widths_cost = array([mirror_cost(w=w) for w in widths])

    # centers = primaries_center(field=heliostats)
    centers = field.centers
    s = [dst(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
    gaps = array([s[i] - 0.5 * (widths[i] + widths[i + 1]) for i in range(len(centers) - 1)]) / 1.0e3
    gaps_cost = array([mirror_gap_cost(g) for g in gaps])

    Ce = elevation_cost(od=absorber_diameter)
    Cr = receiver_cost(od=absorber_diameter)

    specific_cost = (sum(widths_cost, 0) + Ce * (H + dH) + sum(gaps_cost, 0) + Cr) / sum(widths, 0)

    return specific_cost

########################################################################################################################
########################################################################################################################
