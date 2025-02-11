
from pysoltrace.optical_properties import reflector_property, flat_absorber_property, absorber_tube_property, \
    cover_properties, double_side_reflector

from soltracepy.auxiliary import reflective_surface, secondary_surface, absorber_tube_surface, \
    flat_absorber_surface, cover_surfaces


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

        def to_pysoltrace(self, id_number: int):

            return reflector_property(name=self.name, id_number=id_number,
                                      rho=self.rho,
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

        def to_pysoltrace(self, id_number: int):
            return flat_absorber_property(name=self.name, id_number=id_number, alpha=self.alpha)

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

        def to_pysoltrace(self, id_number: int):
            return absorber_tube_property(name=self.name, id_number=id_number, alpha=self.alpha)

    class evacuated_tube:

        def __init__(self,
                     alpha: float,
                     tau: float,
                     ref_index=1.52,
                     slope_error=0.0,
                     specular_error=0.0,
                     name='evacuated_tube'):
            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')
            assert 0 <= abs(tau) <= 1, ValueError('Transmittance value must be between 0 and 1.')

            self.name = name

            self.alpha = abs(alpha)
            self.tau = abs(tau)
            self.n = abs(ref_index)

            self.slope_error = abs(slope_error)
            self.specular_error = abs(specular_error)

        def to_soltrace(self):
            """
            :return: It returns a list with equivalent SolTrace Optics, from the outer cover to the absorber.
            """

            outer_cover, inner_cover = cover_surfaces(tau=self.tau, refractive_index=self.n,
                                                      slope_error=self.slope_error*1e3,
                                                      specular_error=self.specular_error*1e3,
                                                      name=self.name)

            absorber = OpticalProperty.absorber_tube(name=f'{self.name}_absorber', alpha=self.alpha).to_soltrace()

            return [outer_cover, inner_cover, absorber]

        def to_pysoltrace(self, id_number: int):

            outer_cover_property, inner_cover_property = cover_properties(id_number=id_number,
                                                                          name=self.name, tau=self.tau,
                                                                          refractive_index=self.n,
                                                                          slope_error=self.slope_error*1e3,
                                                                          specular_error=self.specular_error*1e3)

            absorber_property = absorber_tube_property(id_number=id_number + 2,
                                                       name=f'{self.name}_absorber', alpha=self.alpha)

            return [outer_cover_property, inner_cover_property, absorber_property]

    class secondary:

        def __init__(self, name: str, rho=1.0, slope_error=0., spec_error=0.):
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
                                     slope_error=self.slope_error * 1e3, spec_error=self.spec_error*1e3)

        def to_pysoltrace(self, id_number: int):

            return double_side_reflector(id_number=id_number,
                                         name=self.name,
                                         rho=self.rho,
                                         slope_error=self.slope_error*1e3,
                                         spec_error=self.spec_error*1e3)
