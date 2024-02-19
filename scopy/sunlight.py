# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:40:11 2020 @author: André Santos (andrevitoras@gmail.com/avas@uevora.pt)

This module implement functions and classes to represent properties and perform calculations regarding the
incident sunlight in a solar concentrator. It considers a XYZ Global (inertial) Coordinate System (GCS) as follows:
    * X points to East
    * Y points North
    * Z points Zenith.

Weather Data is defined in basis of a Typical Meteorological Year (TMY), and taken from the PVGIS API by the function
'pvlib.iotools.get_pvgis_tmy()'. The sun position is defined by a pair of solar angles 'sun zenith' and 'sun azimuth',
and values are calculated with 'pvlib.solarposition.get_solarposition()' by the SPA algorithm [1]. In this case,
the sun azimuth is measured eastward from North. Thus:
    * 0 <= sun azimuth <= 2*pi

This definition of sun azimuth differs from the one presented in some text books [2-4], in which it is measured westward
from the South. Here, the definition based on the SPA algorithm is used. Thus, solar noon happens when the solar azimuth
is 180º (pi) for the Northern hemisphere and 0 for the southern hemisphere.


[1] Reda, I., Andreas, A., 2004. Solar position algorithm for solar radiation applications.
    Solar Energy 76, 577–589. https://doi.org/10.1016/j.solener.2003.12.003
[2] Rabl, A., 1985. Active Solar Collectors and Their Applications. Oxford University Press, New York.
[3] Duffie, J.A., Beckman, W.A., 2013. Solar Engineering of Thermal Processes, 4th Ed. John Wiley & Sons, New Jersey.
[4] Kalogirou, S.A., 2014. Solar Energy Engineering, 2nd Ed. Elsevier, London. https://doi.org/10.1016/C2011-0-07038-2.


# ToDo: Crete a function to calculate (and/or plot) the frequency distribution of transversal and longitudinal incidence
    angles of a particular location defined by latitude and longitude (or a particular TMY).


"""
from pathlib import Path
from typing import Tuple, Callable

from numpy import sin, cos, tan, arctan, array, dot, pi, log, exp, linspace, zeros, sqrt, identity, deg2rad, rad2deg, \
    ndarray, power, piecewise, absolute
from pandas import DataFrame

from pvlib.iotools import get_pvgis_tmy
from pvlib.location import Location
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.stats import norm, uniform

from niopy.geometric_transforms import R
from soltracepy import Sun
from utils import dic2json


########################################################################################################################
# Classes ##############################################################################################################


class SiteData:

    def __init__(self,  name: str,
                 latitude: float, longitude: float,
                 start_year: int = 2006, end_year: int = 2016):

        self.name = name
        self.lat = latitude
        self.lon = longitude

        tmy = get_tmy_data(latitude=self.lat, longitude=self.lon,
                           start_year=start_year, end_year=end_year)

        self.tmy_data = tmy[0]
        self.tmy_info = tmy[1:]

        self.dni_sum = self.tmy_data['dni'].values.sum()

    def linear_angles(self, NS=True, solar_longitudinal=False):

        return sun2lin(zenith=self.tmy_data['sun zenith'].values,
                       azimuth=self.tmy_data['sun azimuth'].values,
                       degrees=True, NS=NS, solar_longitudinal=solar_longitudinal)

    def sun_references_positions(self):

        tt, tl = self.linear_angles()

        avg_trans = tt.dot(self.tmy_data['dni'].values) / self.tmy_data['dni'].sum()
        avg_long = tl.dot(self.tmy_data['dni'].values) / self.tmy_data['dni'].sum()

        return avg_trans, avg_long

    def export(self, files_path: Path, file_name: str):
        tmy_data_file_path = Path(files_path, f"{file_name}.csv")
        self.tmy_data.to_csv(tmy_data_file_path, index=False, header=True)

        file_full_path = Path(files_path, f"{file_name}.json")

        out_dic = {'variable_name': self.name, 'latitude': self.lat, 'longitude': self.lon,
                   'dni_sum': self.dni_sum, 'tmy_data': str(tmy_data_file_path)}

        dic2json(d=out_dic, file_path=files_path, file_name=file_name)

        return file_full_path


class RadialSource:
    """
    This class represent different sources of sunlight spread: sun shape, optical errors, and effective sources [1].
    Sources are characterized by probability density functions denominated as distributions [2]. The base distribution
    is a radial one, as defined by Rabl [1]. Nevertheless, properties of the correspondent linear distribution are
    also given.

    [1] Rabl A. Active Solar Collectors and Their Applications. New York: Oxford University Press; 1985.
    [2] Wang et al. Solar Energy 2020;195:461–74. https://doi.org/10.1016/j.solener.2019.11.035.
    """

    def __init__(self, name: str = 'ES', profile: str = None, size: float = None, user_data: array = None):
        """
        :param name: source variable_name.

        :param profile: source profile.
        Examples are: pillbox ('p'), Gaussian ('g'), Buie ('b') or user-defined source 'u'.
        If 'None', a perfect source is defined, here denominated as a collimated one.

        :param size: size of the source, in radians.
        The half-width for a pillbox, the standard deviation for a Gaussian, and the circumsolar ratio for a Buie.
        For a collimated profile, None is the input.

        :param user_data: When a user-defined source is the selected profile, one needs to define the set of data points
        which characterizes the source.
        """

        self.name = name

        if size is None or profile is None:
            self.profile = None
            self.size = 0.
            self.rms_width = 0.

            self.radial_distribution = None

            self.radial_pdf = None
            self.linear_pdf = None

            self.linear_distribution = None

        elif profile == 'pillbox' or profile == 'p':
            self.profile = 'pillbox'
            self.size = abs(size)
            self.rms_width = self.size / sqrt(2)

            self.radial_distribution = pillbox_sunshape(half_width=self.size, pdf=False)

            self.radial_pdf = radial_pdf(pillbox_sunshape(half_width=self.size, pdf=True))
            self.linear_pdf = pillbox_sunshape(half_width=self.size * sqrt(3.) / 2., pdf=True)

            self.linear_cdf = cumulative_pillbox_source(half_width=self.size, linear=True)

        elif profile == 'gaussian' or profile == 'g':
            self.profile = 'gaussian'
            self.size = abs(size)
            self.rms_width = self.size * sqrt(2)

            self.radial_distribution = gaussian_sunshape(std=self.size, pdf=False)

            self.radial_pdf = radial_pdf(gaussian_sunshape(std=self.size, pdf=True))
            self.linear_pdf = gaussian_sunshape(std=self.size, pdf=True)

            self.linear_cdf = cumulative_gaussian_source(std_source=self.size, linear=True)

        elif profile == 'buie' or profile == 'b':
            self.profile = 'buie'
            self.size = abs(size)

            sunshape = BuieSunshape(csr=self.size, csr_calibration=True)

            self.rms_width = sunshape.rms_width
            self.radial_distribution = sunshape.radial_distribution

        elif profile == 'user' or profile == 'u':
            self.profile = "'u'"
            self.values = user_data

        else:
            raise ValueError("Please, input a valid profile: 'gaussian', 'pillbox', 'buie', "
                             "or all None for a collimated sunlight model")

    def linear_cum_eff(self, nbr_datapoints: int = 30000):

        x_values = linspace(0, pi, int(nbr_datapoints))
        y_values = self.linear_cdf(x_values)

        cum_eff = zeros(shape=(x_values.shape[0], 2))
        cum_eff.T[:] = x_values, y_values

        return cum_eff

    def linear_convolution(self, slope_error: float = 0., specular_error: float = 0.):
        """
        This method compute the convolution of the linear source with an overall measure of Gaussian optical errors.

        :param slope_error: standard deviation representing the surface slope error, in radians.
        :param specular_error: standard deviation representing the surface specular error, in radians.

        :return: the cumulative effective source of the convoluted source, an array
        """

        if slope_error == 0 and specular_error == 0.:
            cum_eff = self.linear_cum_eff()
        else:
            optical_errors = sqrt(4*slope_error**2 + specular_error**2)
            errors_source = RadialSource(profile='g', size=optical_errors)

            conv_pdf = pdfs_convolution(f=self.linear_pdf,
                                        g=errors_source.linear_pdf,
                                        out_callable=True)

            conv_cdf = pdf2cdf(conv_pdf)

            x_range = linspace(start=0, stop=pi, num=100000)
            cum_eff = zeros(shape=(x_range.shape[0], 2))

            cum_eff.T[:] = x_range, conv_cdf(x_range)

        return cum_eff

    def to_soltrace(self, sun_dir: array = array([0, 0, 1])):
        # OBS: SolTrace Sun box definitions considers units in milliradians (mrad).
        # return Sun(sun_dir=sun_dir, profile=self.profile, size=self.size * 1e3)
        return source2soltrace(source=self, sun_dir=sun_dir)

    def export2dic(self):

        dic = {'profile': self.profile,
               'size': self.size}

        return dic

    def export2json(self, file_path: Path):

        file_full_path = Path(file_path, f'{self.name}.json')
        dic = self.export2dic()

        dic2json(d=dic, file_path=file_path, file_name=self.name)

        return file_full_path


class BuieSunshape:
    """
    This class stands to represent the Buie's sunshape model [1,2]: a radial sunshape that accounts for intensity
    variations within the sun disk and also the circumsolar region.

    [1] Buie et al. Solar Energy 2003;74:113–22. https://doi.org/10.1016/S0038-092X(03)00125-7.
    [2] Wang et al. Solar Energy 2020;195:461–74. https://doi.org/10.1016/j.solener.2019.11.035.
    [3] Rabl A. Active Solar Collectors and Their Applications. New York: Oxford University Press; 1985.
    """

    def __init__(self, csr: float, aureole_extension: float = 43.6e-3, csr_calibration=True):

        # Angles defined by Buie et al. [1] when defining the sunshape profile.
        self.disk_radius = 4.65e-3  # extension of the solar disk, in rad.
        self.aureole_radius = abs(aureole_extension)  # extension of the solar aureole, in rad.

        # inputted circumsolar ratio.
        self.csr = abs(csr)

        #############################################################################################################
        # Calibration of the circumsolar ratio ######################################################################

        # As observed by Buie et al. [1], the inputted csr is not the same value as the CSR that can be calculated from
        # the distribution function. Remember that Buie's model was derived from a curve-fitting of data from a
        # statistical analysis of real measurements from different databases. Therefore, small errors and deviations are
        # expected.
        # However, these propositions to correct the csr were found in Wang et al. [2]:
        # https://github.com/anustg/Tracer/blob/master/tracer/sources.py.

        if csr_calibration:
            self.crs_cali = self.CSR_calibration('CA')
        else:
            self.crs_cali = self.csr
        #############################################################################################################

        #############################################################################################################
        # Calculations of the RMS width of the sun shape ############################################################

        # See Rabl's formula [3, pp. 134-136].
        # It  follows the definition of the second momentum of a probability density function.
        # Remember that Buie's model yield a normalized function phi(x), so that phi(0) = 1, and not a pdf that
        # $\int_{0}^{\pi/2} phi(x) x \dd x = 1$.

        # Radial distribution
        self.radial_distribution = buie_sunshape(csr=self.crs_cali)

        self.distribution_area = quad(lambda x: self.radial_distribution(x) * x,
                                      0, self.aureole_radius, full_output=True)[0]

        self.second_moment = quad(lambda x: self.radial_distribution(x) * x**3,
                                  0, self.aureole_radius, full_output=True)[0]

        self.rms_width = sqrt(self.second_moment / self.distribution_area)

        #############################################################################################################
        # Old calculation of the rms width ##########################################################################
        # def N(x):
        #     return self.radial_distribution(x) * (x ** 3)
        #
        # def D(x):
        #     return self.radial_distribution(x) * x
        #
        # num = quad(N, 0, self.aureole_radius)[0]
        # den = quad(D, 0, self.aureole_radius)[0]
        #
        # # It holds the RMS width of the distribution in radians.
        # self.rms_width = sqrt(num / den)
        ############################################################################################################

    def CSR_calibration(self, source):
        """
        Further process the circumsolar ratio to true value.
        Buie's model is a curve-fitting of data from a statistical analysis of real measurement databases. Thus,
        the inputted 'csr' produces a function that when integrated to calculate its circumsolar ration it yields a
        value different to the one used to construct the function.

        In this sense, some 'calibration' is need to have a Buie's function that yield a correct CSR value.
        Here, two propositions are given:
        - 'CA' from Charles Charles-Alexis Asselineau at ANU.
        - 'tonatiuh' from Manuel Blanco at CyI.
        Source code: https://github.com/anustg/Tracer/blob/master/tracer/sources.py.
        """

        csr_g = self.csr

        if source == 'CA':
            if csr_g <= 0.1:
                csr_cali = -2.245e+3 * csr_g ** 4. + 5.207e+2 * csr_g ** 3. - 3.939e+1 * csr_g ** 2. \
                           + 1.891e+0 * csr_g + 8e-3
            else:
                csr_cali = 1.973 * csr_g ** 4. - 2.481 * csr_g ** 3. + 0.607 * csr_g ** 2. + 1.151 * csr_g - 0.020
        elif source == 'tonatiuh':
            if csr_g > 0.145:
                csr_cali = -0.04419909985804843 + csr_g * (1.401323894233574 + csr_g * (
                        -0.3639746714505299 + csr_g * (-0.9579768560161194 + 1.1550475450828657 * csr_g)))
            elif 0.035 < csr_g <= 0.145:
                csr_cali = 0.022652077593662934 + csr_g * (0.5252380349996234 +
                                                           (2.5484334534423887 - 0.8763755326550412 * csr_g)) * csr_g
            else:
                csr_cali = 0.004733749294807862 + csr_g * (4.716738065192151 + csr_g * (-463.506669149804 + csr_g * (
                        24745.88727411664 + csr_g * (-606122.7511711778 + 5521693.445014727 * csr_g))))
        else:
            csr_cali = csr_g

        return csr_cali

    def csr_calculation(self) -> float:

        circumsolar_flux = quad(lambda x: self.radial_distribution(x) * x, self.disk_radius, self.aureole_radius)[0]
        total_flux = quad(lambda x: self.radial_distribution(x) * x, 0, self.aureole_radius)[0]

        return circumsolar_flux / total_flux

    def radial_pdf(self):
        def pdf(x): return self.radial_distribution(x) * x / self.distribution_area

        return pdf

    def radial_cdf(self):
        # noinspection PyTypeChecker
        cdf = pdf2cdf(pdf=self.radial_pdf(), linear=False)

        return cdf


########################################################################################################################
# FUNCTIONS ############################################################################################################


def get_tmy_data(latitude: float, longitude: float,
                 start_year: int = 2006, end_year: int = 2016) -> Tuple[DataFrame, list, dict, list]:
    """
    This function gets a Typical Meteorological Year (TMY) from the PVGIS API.

    :param latitude: local latitude, in degrees.
    :param longitude: local longitude, in degrees.
    :param start_year: starting year to construct the TMY
    :param end_year: ending year to construct the TMY

    :return: It returns a pandas DataFrame object with the TMY data.

    The returned DataFrame has the following columns: 'temp_air', 'relative_humidity', 'ghi', 'dni', 'dhi', 'IR(h)',
    'wind_speed', 'wind_direction', 'pressure', 'sun_zenith', 'sun_azimuth'.

    The 'temp_air' column has units in ºC. The columns 'ghi', 'dni', and 'dhi' have units in W/m2.
    """

    pvgis_database, months_dic, location_data, legend = get_pvgis_tmy(latitude=latitude, longitude=longitude,
                                                                      outputformat='csv',
                                                                      map_variables=True,
                                                                      startyear=start_year, endyear=end_year)

    location = Location(latitude=location_data['latitude'],
                        longitude=location_data['longitude'],
                        altitude=location_data['elevation'],
                        tz='UTC')

    solar_position = location.get_solarposition(times=pvgis_database.index)

    pvgis_database['sun zenith'] = solar_position['zenith'].values
    pvgis_database['sun azimuth'] = solar_position['azimuth'].values

    return pvgis_database, months_dic, location_data, legend


def sun_vector(zenith, azimuth, degrees=True):
    """
    This function calculates the unit vector representing the incidence sunlight direction based in the zenith-azimuth
    pair of solar angles. The incidence direction vector always points up, i.e., z > 0.

    :param zenith: sun zenith angle.
    :param azimuth: sun azimuth angle.
    :param degrees: Boolean to indicate if solar zenith and azimuth are in degrees.

    :return: This function returns the incidence direction of sunlight.

    It has a vectorized approach that returns an array of [x, y, z] arrays if the arguments zenith and azimuth are
    arrays of the same size.

    Following the description in the beginning of this module, it considers that the sun azimuth is measured eastward
    from North: 0 <= azimuth <= 2*pi.
    """

    # Ensure values in radians to perform calculations
    zen_rad = deg2rad(zenith) if degrees else zenith
    azi_rad = deg2rad(azimuth) if degrees else azimuth

    # Components of the incidence vector
    x = sin(zen_rad) * sin(azi_rad)
    y = sin(zen_rad) * cos(azi_rad)
    z = cos(zen_rad)

    vector = array([x, y, z]).T

    return vector


def sun_direction(theta_t: float, theta_l: float):
    """
    This function calculates the unit vector representing the incidence direction of the sun as an [x, y, z] array.
    It considers the transversal and longitudinal incidence angles defined by the correspondent planes of a linear
    concentrator. The incidence direction vector always points up, i.e., z > 0.

    :param theta_t: transversal incidence angle, in degrees.
    :param theta_l: longitudinal incidence angle, in degrees.

    :return: It returns [x, y, z] array-vector which represents the incidence direction of sunlight.

    It assumes a XYZ coordinate system in which the transversal plane of the linear concentrator is the ZX plane,
    and angles in such a plane are defined as positive or negative regarding a correspondent rotation of the Z-axis
    around the Y-axis. In the same sense, the YZ plane is the longitudinal plane, and angles in such a plane are defined
    as positive or negative regarding a correspondent rotation of the Z-axis around the X-axis [1].

    Thus:
     * A positive transversal incidence angle lies in the right-side of the ZX plane: x > 0 and z > 0.
     * A negative transversal incidence angle lies in the left-side of the ZX plane: x < 0 and z > 0.

     * A positive longitudinal incidence angle lies in the left-side of the YZ plane: y < 0 and z > 0.
     * A negative longitudinal incidence angle lies in the right-side of the YZ plane: y > 0 and z > 0.

    [1] Solar Energy (2021) 227, 203–216. https://doi.org/10.1016/j.solener.2021.08.085 (see Fig. 1).
    """

    theta_t_rad, theta_l_rad = deg2rad([theta_t, theta_l])

    Ix, Iy, Iz = identity(3)

    r1 = R(theta_l_rad, Ix)
    projected_angle = arctan(tan(theta_t_rad) * cos(theta_l_rad))
    r2 = R(projected_angle, Iy)

    return dot(r1, dot(r2, Iz))


def sun2lin(zenith, azimuth, degrees=True, NS=True, solar_longitudinal=False):
    """
    A vectorized function to convert sun positions given by solar zenith and azimuth to linear concentrators incidence
    angles: transversal and longitudinal incidence angles (and also the solar longitudinal angle [1, 2]).
    It returns the linear angles in degrees, as them are usually visualized.

    :param zenith: solar zenith angle.
    :param azimuth: solar azimuth angle.

    :param degrees: a boolean sign to inform whether solar angles are in degree or radians.
    :param NS: a sign to inform whether a NS (North-South) or EW (East-West) mounting for the linear concentrator.
    :param solar_longitudinal: a boolean sing to return (True) or not (False) the solar longitudinal angle.

    :return: Returns a tuple of angles, in degrees.
    If an array of solar angles is given, then a tuple of arrays is returned.

    This functions assumes the previous definitions for the sun azimuth, as established in the beginning of the module
    and also mentioned in the documentation of the 'sun_vector' function. In summary, sun azimuth is measured eastward
    from North, so that: 0 <= azimuth <= 2*pi.

    In general, two coordinate systems can be defined: an inertial one and one attached to the linear concentrator:
    * Global (Inertial) Coordinate System (GCS) follows the definition presented in the function 'sun_vector'.
    * Concentrator Coordinate System (CCS) follows the definition presented in the function 'sun_direction'.

    Therefore, by these definitions, in  the case of an NS-oriented field both GCS and CCS are fully aligned, and the
    conversion of sun angles to linear angles is quite straightforward. Nevertheless, if a displacement from the NS
    exist (e.g., a EW-orientation), the expressions changes.


    [1] IEC (International Electrotechnical Commission). Solar thermal electric plants
    - Part 5-2: Systems and components - General requirements and test methods for large-size linear Fresnel collectors.
    Solar thermal electric plants, 2021.
    [2] Hertel JD, Martinez-Moll V, Pujol-Nadal R. Estimation of the influence of different incidence angle modifier
    models on the bi-axial factorization approach. Energy Conversion and Management 2015;106:249–59.
    https://doi.org/10.1016/j.enconman.2015.08.082.
    """

    # Ensuring zenith and azimuth in radians to perform calculations ####
    zen_rad = deg2rad(zenith) if degrees else zenith
    azi_rad = deg2rad(azimuth) if degrees else azimuth
    #####################################################################

    ###########################################################################################################
    # Implementation that only accounts for NS and EW-oriented fields ##################
    # Calculating transversal and longitudinal incidence angles #########
    tt = arctan(tan(zen_rad) * sin(azi_rad))
    tl = arctan(tan(zen_rad) * cos(azi_rad))
    #####################################################################

    # Accounting for a NS or EW mounting #########
    # angles still in radians
    tt, tl = (tt, tl) if NS else (-tl, tt)

    ##############################################

    # To return or not the solar longitudinal incidence angle ######################################
    # It also returns everything in degrees, the more usual unit for the linear incidence angles.
    if not solar_longitudinal:
        angles = rad2deg(tt), rad2deg(tl)
    else:
        ti = arctan(tan(tl) * cos(tt))
        angles = rad2deg(tt), rad2deg(tl), rad2deg(ti)
    #################################################################################################
    ###########################################################################################################

    ####################################################################################################################
    # Possible new implementation ######################################################################################
    # ToDo: General implementation for a concentrator with an azimuthal displacement (phi) from the NS-orientation:
    #  Check if it works
    # # If NS, phi = 0. If EW, phi = pi/2
    # phi = 0 if NS is True else deg2rad(NS)
    #
    # x = cos(phi) * sin(zen_rad) * sin(azi_rad) - sin(phi) * sin(zen_rad) * cos(azi_rad)
    # y = sin(phi) * sin(zen_rad) * sin(azi_rad) + cos(phi) * sin(zen_rad) * cos(azi_rad)
    # z = cos(zen_rad)
    #
    # tt = arctan(x / z)
    # tl = arctan(y / z)
    #
    # if not solar_longitudinal:
    #     angles = rad2deg(tt), rad2deg(tl)
    # else:
    #     ti = arctan(tan(tl) * cos(tt))
    #     angles = rad2deg(tt), rad2deg(tl), rad2deg(ti)
    ####################################################################################################################

    return angles


def buie_sunshape(csr: float):
    """
    This function implements Buie's model [1] for the radiance distribution of the sun.
    It returns a normalized intensity (or radiance) profile as function of the angular deviation from the sun's center,
    in radians [2].

    Buie's model is a radial sun shape profile, based on the definition of the circumsolar ratio -- the function's
    only argument.

    :param csr: The circumsolar ratio, a value between 0 and 1.
    :return: Returns a radial sun shape profile function (callable).


    [1] Buie et al. Solar Energy 2003;74:113–22. https://doi.org/10.1016/S0038-092X(03)00125-7.
    [2] Wang et al. Solar Energy 2020;195:461–74. https://doi.org/10.1016/j.solener.2019.11.035.
    """

    gamma = - 0.1 + 2.2 * log(0.52 * csr) * (csr ** 0.43)
    k = 0.9 * log(13.5 * csr) * (csr ** -0.3)

    def phi(x):  # x is an angle in radians

        return piecewise(x,
                         [absolute(x) <= 4.65e-3,
                          (4.65e-3 < absolute(x)) & (absolute(x) <= 43.6e-3),
                          absolute(x) > 43.6e-3],

                         [lambda y: cos(326 * y) / cos(308 * y),
                          lambda y: exp(k) * power(absolute(y) * 1.e3, gamma),
                          lambda y: 0.])

        # return cos(326 * x) / cos(308 * x) if abs(x) <= 4.65e-3 else exp(k) * power(abs(x) * 1.e3, gamma)

    return phi


def pillbox_sunshape(half_width: float, pdf=False):

    if not pdf:
        phi0 = uniform.pdf(0, loc=-half_width, scale=2 * half_width)

        def phi(x):
            return uniform.pdf(x, loc=-half_width, scale=2 * half_width) / phi0
    else:
        def phi(x):
            return uniform.pdf(x, loc=-half_width, scale=2 * half_width)

    return phi


def gaussian_sunshape(std: float, pdf=False):

    if not pdf:
        phi0 = norm.pdf(0, loc=0, scale=std)

        def phi(x):
            return norm.pdf(x, loc=0, scale=std) / phi0
    else:
        def phi(x):
            return norm.pdf(x, loc=0, scale=std)

    return phi


def radial_pdf(profile) -> Callable:

    area = quad(lambda x: profile(x) * x, 0, 100.e-3, full_output=True)[0]

    def B(x): return profile(x) * x / area

    return B


def linear_pdf(profile) -> Callable:

    half_area = quad(lambda x: profile(x), 0, 100.e-3, full_output=True)[0]

    def E(x): return profile(x) / (2 * half_area)

    return E


def pdf2cdf(pdf: (Callable or ndarray), linear=True) -> interp1d:
    """
    This function calculates and returns the cumulative density function (cdf) of a particular probability density
    function (pdf) defined as f(x).

    :param pdf: probability density function (a callable or an (#, 2)-shape array of data points).
    :param linear: a boolean sign to indicate if is a linear (True) or a radial (False) pdf.

    :return: the cumulative density function.

    By definition the cdf is given by:
    cdf(x) = \int_{0,-inf}^{x}{f(x) \dd x}

    where f(x) is the pdf, 0 is the lower limit for the case of a radial source, and -inf is the lower limit for the
    case of a linear source. Thus:
        * for a linear source cdf(+inf) - cdf(0) = 0.5.
        * for a radial source cdf(+inf) - cdf(0) = 1.

    This function assumes that the mean value of pdf(x) is zero, and that it is also symmetric so that f(-x) = f(x).

    The argument 'pdf' can be inputted in two ways: a callable function as pdf(x) or even as a set of array
    data points [x, pdf(x)].
    """

    assert isinstance(pdf, (ndarray, Callable)), \
        'Function argument has a invalid type. ' \
        'It should be callable (e.g., function, method), as pdf(x), or an array with shape (#,2)'

    if callable(pdf):
        pdf_function = pdf
    elif pdf.shape[1] == 2:
        pdf_function = interp1d(*pdf.T, kind='cubic')
    else:
        raise ValueError('Function argument has a invalid type. '
                         'It should be callable (e.g., function, method) as pdf(x) or an array with shape (#,2)')

    x_values = linspace(start=0, stop=100.e-3, num=500)
    y_values = array([quad(lambda y: pdf_function(y), 0, x, full_output=True)[0] for x in x_values])

    if linear:
        cdf = interp1d(x_values, y_values / (2 * y_values[-1]), kind='cubic', bounds_error=False, fill_value=0.5)
    else:
        cdf = interp1d(x_values, y_values / y_values[-1], kind='cubic', bounds_error=False, fill_value=1)

    return cdf


def cumulative_pillbox_source(half_width=4.65e-3, linear=True):

    if not linear:
        raise ValueError('Not implemented yet!')
    else:
        # The half-width of linear pillbox profile (delta) relates with the half-width of a radial pillbox profile by:
        delta = half_width * (3 ** 0.5) / 2
        x_values = linspace(0, pi, 100000)

        # The cumulative function of the linear pillbox profile
        def y(x):
            return abs(x) / (2 * delta) if abs(x) <= delta else 0.5

        y_values = array([y(x) for x in x_values])

        # The interpolated cumulative function to be returned
        cumulative_function = interp1d(x_values, y_values, kind='linear')

    return cumulative_function


def cumulative_gaussian_source(std_source, linear=True):
    if not linear:
        raise ValueError('Not implemented yet!')
    else:
        # The standard deviation of a Gaussian linear source is the same of a radial Gaussian source
        std = std_source

        x = linspace(0, pi, 100000)
        cv = norm.cdf(x, scale=std) - 0.5
        cumulative_function = interp1d(x, cv, kind='cubic')

    return cumulative_function


def pdfs_convolution(f, g, out_callable=True):

    x_values = linspace(start=-pi, stop=pi, num=100000)

    f_values = f(x_values)
    g_values = g(x_values)

    fog_values = fftconvolve(f_values, g_values, mode='same')
    fog_function = interp1d(x_values, fog_values, kind='cubic')

    fog_area = quad(lambda x: fog_function(x), x_values[0], x_values[-1], full_output=True)[0]

    if out_callable:
        pdf = interp1d(x_values, fog_values / fog_area, kind='cubic')
    else:
        pdf = zeros(shape=(x_values.shape[0], 2))
        pdf.T[:] = x_values, fog_values / fog_area

    return pdf


def source2soltrace(source: RadialSource, sun_dir: array) -> Sun:

    if source.profile == 'buie' or source.profile == 'user':

        # Angular displacement in radians
        x_range = linspace(start=0, stop=45.e-3, num=50)
        sunshape_profile = zeros(shape=(x_range.shape[0], 2))

        # SolTrace reads the angular displacement in milliradians
        sunshape_profile.T[:] = x_range * 1e3, source.radial_distribution(x_range).round(4)
        sun = Sun(sun_dir=sun_dir, profile='u', user_data=sunshape_profile)
    else:
        # SolTrace reads the size of a pillbox or a Gaussian sunshape in milliradians
        sun = Sun(sun_dir=sun_dir.round(6), profile=source.profile, size=round(source.size * 1e3, 4))

    return sun
