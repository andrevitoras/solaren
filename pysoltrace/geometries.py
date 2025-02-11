from numpy import array, inf
from pysoltrace import PySolTrace


def flat_element(width: float, length: float,
                 ecs_origin: array, ecs_aim: array,
                 parent_stage: PySolTrace.Stage, id_number: int,
                 optic: PySolTrace.Optics) -> PySolTrace.Stage.Element:

    element = PySolTrace.Stage.Element(parent_stage=parent_stage, element_id=id_number)
    element.optic = optic

    element.position.x, element.position.y, element.position.z = ecs_origin
    element.aim.x, element.aim.y, element.aim.z = ecs_aim

    element.aperture_rectangle(length_x=width, length_y=length)
    element.surface_flat()

    return element


def tubular_element(tube_radius: float, tube_length: float,
                    ecs_origin: array, ecs_aim: array,
                    parent_stage: PySolTrace.Stage, id_number: int,
                    optic: PySolTrace.Optics) -> PySolTrace.Stage.Element:

    element = PySolTrace.Stage.Element(parent_stage=parent_stage, element_id=id_number)
    element.optic = optic

    element.position.x, element.position.y, element.position.z = ecs_origin
    element.aim.x, element.aim.y, element.aim.z = ecs_aim

    element.aperture_singleax_curve(x1=0, x2=0, L=tube_length)
    element.surface_cylindrical(radius=tube_radius)

    return element


def linear_fresnel_mirror(width: float, length: float, radius: float,
                          ecs_origin: array, ecs_aim: array,
                          parent_stage: PySolTrace.Stage, id_number: int,
                          optic: PySolTrace.Optics) -> PySolTrace.Stage.Element:

    element = PySolTrace.Stage.Element(parent_stage=parent_stage, element_id=id_number)
    element.optic = optic

    element.position.x, element.position.y, element.position.z = ecs_origin
    element.aim.x, element.aim.y, element.aim.z = ecs_aim

    if radius == 0.:
        element.aperture_rectangle(length_x=width, length_y=length)
        element.surface_flat()
    else:
        element.aperture_singleax_curve(-width/2, width/2, length)

        element.surface_parabolic(focal_len_x=radius/2., focal_len_y=+inf)

    return element


def spherical_mirror(width: float, length: float, radius: float,
                     ecs_origin: array, ecs_aim: array,
                     parent_stage: PySolTrace.Stage, id_number: int,
                     optic: PySolTrace.Optics) -> PySolTrace.Stage.Element:

    element = PySolTrace.Stage.Element(parent_stage=parent_stage, element_id=id_number)
    element.optic = optic

    element.position.x, element.position.y, element.position.z = ecs_origin
    element.aim.x, element.aim.y, element.aim.z = ecs_aim

    element.aperture_rectangle(length_x=width, length_y=length)

    if radius == 0.:
        element.surface_flat()
    else:
        element.surface_spherical(radius=radius)

    return element


