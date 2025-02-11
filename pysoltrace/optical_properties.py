from numpy import ndarray
from pysoltrace import PySolTrace


def reflector_property(name: str,
                       id_number: int,
                       rho,
                       slope_error: float,
                       spec_error: float,
                       gaussian_error=True) -> PySolTrace.Optics:

    optical_property = PySolTrace.Optics(id=id_number)
    optical_property.name = name

    # The front face #############################################
    if isinstance(rho, (float, int)):
        optical_property.front.transmissivity = 0.
        optical_property.front.reflectivity = abs(rho)
    else:
        optical_property.front.userefltable = True
        optical_property.front.refltable = rho.tolist()

    optical_property.front.dist_type = 'g' if gaussian_error else 'p'
    optical_property.front.slope_error = slope_error if slope_error > 0 else 1.e-4
    optical_property.front.spec_error = spec_error if spec_error > 0 else 1.e-4
    #################################################################

    # The back face #################################################
    optical_property.back.reflectivity = 0.
    optical_property.back.transmissivity = 0
    optical_property.back.slope_error = 1.e-4
    optical_property.back.spec_error = 1.e-4
    #################################################################

    return optical_property


def double_side_reflector(name: str,
                          id_number: int,
                          rho,
                          slope_error: float,
                          spec_error: float,
                          gaussian_error=True) -> PySolTrace.Optics:

    optical_property = PySolTrace.Optics(id=id_number)
    optical_property.name = name

    if isinstance(rho, (float, int)):
        optical_property.front.transmissivity = 0.
        optical_property.front.reflectivity = abs(rho)

        optical_property.back.transmissivity = 0
        optical_property.back.reflectivity = abs(rho)

    elif isinstance(rho, ndarray):
        optical_property.front.userefltable = True
        optical_property.front.refltable = rho.tolist()

        optical_property.back.userefltable = True
        optical_property.back.refltable = rho.tolist()
    else:
        raise ValueError('Please, check rho argument type. It should be an array for angular dependent property!')

    # the front face ##################################################
    optical_property.front.dist_type = 'g' if gaussian_error else 'p'
    optical_property.front.slope_error = slope_error if slope_error > 0 else 1.e-4
    optical_property.front.spec_error = spec_error if spec_error > 0 else 1.e-4
    #################################################################

    # The back face ##################################################
    optical_property.back.dist_type = 'g' if gaussian_error else 'p'
    optical_property.back.slope_error = slope_error if slope_error > 0 else 1.e-4
    optical_property.back.spec_error = spec_error if spec_error > 0 else 1.e-4
    ##################################################################

    return optical_property


def flat_absorber_property(name: str,
                           id_number: int,
                           alpha: float) -> PySolTrace.Optics:

    optical_property = PySolTrace.Optics(id=id_number)
    optical_property.name = name

    # The front face #############################################
    optical_property.front.transmissivity = 0.
    optical_property.front.reflectivity = 1 - abs(alpha)
    optical_property.front.slope_error = 1.e-4
    optical_property.front.spec_error = 1.e-4
    #################################################################

    # The back face #################################################
    optical_property.back.reflectivity = 1.
    optical_property.back.transmissivity = 0.
    optical_property.back.slope_error = 1.e-4
    optical_property.back.spec_error = 1.e-4
    #################################################################

    return optical_property


def absorber_tube_property(name: str,
                           id_number: int,
                           alpha: float) -> PySolTrace.Optics:

    optical_property = PySolTrace.Optics(id=id_number)
    optical_property.name = name

    # The front face #############################################
    optical_property.front.transmissivity = 0.
    optical_property.front.reflectivity = 1 - abs(alpha)
    optical_property.front.slope_error = 1.e-4
    optical_property.front.spec_error = 1.e-4
    # #################################################################

    # # The back face #################################################
    optical_property.back.transmissivity = 0.
    optical_property.back.reflectivity = 1 - abs(alpha)
    optical_property.back.slope_error = 1.e-4
    optical_property.back.spec_error = 1.e-4
    # #################################################################

    return optical_property


def transmissive_property(id_number: int, name: str,
                          tau: float, nf: float, nb: float,
                          slope_error=1.e-4, spec_error=1.e-4):

    optical_property = PySolTrace.Optics(id=id_number)
    optical_property.name = name

    # The front face #############################################
    optical_property.front.transmissivity = tau
    optical_property.front.reflectivity = 0.
    optical_property.front.slope_error = slope_error if slope_error > 0 else 1.e-4
    optical_property.front.spec_error = spec_error if spec_error > 0 else 1.e-4
    optical_property.front.refraction_real = nf
    # #################################################################

    # # The back face #################################################
    optical_property.back.transmissivity = tau
    optical_property.back.reflectivity = 0.
    optical_property.back.slope_error = spec_error if slope_error > 0 else 1.e-4
    optical_property.back.spec_error = spec_error if spec_error > 0 else 1.e-4
    optical_property.back.refraction_real = nb
    # #################################################################

    return optical_property


def cover_properties(id_number: int, tau: float, name='cover',
                     refractive_index=1.52, slope_error=1.e-4, spec_error=1.e-4):

    tau_s = round(tau**0.5, 5)

    outer_cover_property = transmissive_property(id_number=id_number,  name=f'{name}_outer_cover',
                                                 tau=tau_s,
                                                 nf=refractive_index, nb=1.,
                                                 slope_error=slope_error, spec_error=spec_error)

    inner_cover_property = transmissive_property(id_number=id_number+1,  name=f'{name}_inner_cover',
                                                 tau=tau_s,
                                                 nf=1., nb=refractive_index,
                                                 slope_error=slope_error, spec_error=spec_error)

    return [outer_cover_property, inner_cover_property]
