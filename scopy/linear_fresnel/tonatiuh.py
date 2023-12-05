import io
import subprocess
from pathlib import Path
from typing import TextIO

from numpy import array, cross, rad2deg

from niopy.geometric_transforms import nrm, ang


class Node:

    def __init__(self,
                 name: str,
                 translation=array([0, 0, 0]),
                 rotation=array([0, 0, 1]), angle=0.,
                 scale=array([1, 1, 1])):

        self.variable_name = f'{name}_node'
        self.name = name
        self.translation = translation
        self.rotation = rotation
        self.rot_angle = angle
        self.scale = scale

    def as_script(self, script, outer_node=None):

        tx, ty, tz = self.translation
        rx, ry, rz = self.rotation
        ra = self.rot_angle
        sx, sy, sz = self.scale

        script.write('// Creating node ------------------------------------------------------------------\n')
        if outer_node is None:
            script.write(f'var {self.variable_name} = new NodeObject;\n')
            script.write(f'{self.variable_name}.setName("{self.name}");\n')
        else:
            script.write(f'var {self.variable_name} = {outer_node.variable_name}.createNode("{self.name}");\n')

        script.write(f'{self.variable_name}.setParameter("translation", "{tx} {ty} {tz}");\n')
        script.write(f'{self.variable_name}.setParameter("rotation", "{rx} {ry} {rz} {ra}");\n')
        script.write(f'{self.variable_name}.setParameter("scale", "{sx} {sy} {sz}");\n')
        script.write('// ------ ------------ ------------ ------------ ------------ ------------ ------\n\n')

        return None


def create_sun(script: TextIO,
               azimuth: float, altitude: float,
               profile: str, size: float,
               irradiance=1000.):

    """
    :param script:
    :param azimuth:
    :param altitude:
    :param profile:
    :param size:
    :param irradiance:

    :return:
    """

    if profile == 'b' or profile == 'buie' or profile == 'Buie':
        shape = 'Buie'
        size_type = 'csr'
    elif profile == 'p' or profile == 'pillbox' or profile == 'Pillbox':
        shape = 'Pillbox'
        size_type = 'thetaMax'
    elif profile == 'g' or profile == 'gaussian' or profile == 'Gaussian':
        shape = 'Gaussian'
        size_type = 'sigma'
    else:
        raise ValueError("Invalid arguments")

    script.write('// Setting the Sun configurations --------------------------\n')
    script.write('var scene = tn.getScene();\n')
    script.write('var sun = scene.getPart("world.sun");\n')
    script.write('var sun_position = sun.getPart("position");\n')
    script.write(f'sun_position.setParameter("azimuth", "{azimuth}");\n')
    script.write(f'sun_position.setParameter("altitude", "{altitude}");\n')
    script.write(f'sun_position.setParameter("irradiance", "{irradiance}");\n')

    script.write(f'sun.setParameter("shape", "{shape}");\n')

    script.write(f'var sunshape = sun.getPart("shape");\n')
    script.write(f'sunshape.setParameter("{size_type}", "{size}");\n')
    script.write('// ------------------------------------------------------------\n\n')

    return None


def create_primary_mirror(script: TextIO, node: Node,
                          width: float, radius: float, length: float,
                          rec_aim: array,
                          rho: float, slope_error: float):

    assert rec_aim.shape == (3,), ValueError('Invalid argument! Check documentation.')

    # auxiliary variables for the target point of the primary reflector
    ax, ay, az = rec_aim
    r = abs(radius)

    script.write(f'// Creating a primary mirror in the node defined in the variable "{node.variable_name}" ---------\n')

    # Setting the tracking of the primary mirror ##################################################

    script.write('// creating the tracking type\n')
    script.write(f'var tck = {node.variable_name}.createTracker();\n')

    script.write('// setting the on-axis tracking parameters\n')
    script.write(f'var target = tck.getPart("target");\n')
    script.write(f'target.setParameter("aimingType", "global");\n')
    script.write(f'target.setParameter("aimingPoint", "{ax} {ay} {az}");\n')

    script.write(f'var tck_armo = tck.insertArmature("one-axis");\n')
    script.write(f'tck_armo.setParameter("primaryShift", "0 0 0");\n')
    script.write(f'tck_armo.setParameter("primaryAxis", "0 1 0");\n')
    script.write(f'tck_armo.setParameter("primaryAngles", "-90 90");\n')
    script.write(f'tck_armo.setParameter("facetShift", "0 0 0");\n')
    script.write(f'tck_armo.setParameter("facetNormal", "0 0 1");\n\n')
    ###############################################################################################

    script.write('// Creating a node for the mirror surface\n')
    mirror_node = Node(name='mirror')
    mirror_node.as_script(script=script, outer_node=node)
    script.write(f'var shape = {mirror_node.variable_name}.createShape();\n')

    if r > 0:
        script.write(f'var s_surf = shape.insertSurface("Elliptic");\n')
        script.write(f's_surf.setParameter("aX", {radius});\n')
        script.write(f's_surf.setParameter("aY", 1e+70);\n')
        script.write(f's_surf.setParameter("aZ", {radius});\n')
    else:
        script.write(f'var s_surf = shape.insertSurface("Planar");\n')

    script.write(f'var s_prof = shape.insertProfile("Box");\n')
    script.write(f's_prof.setParameter("uSize", {width});\n')
    script.write(f's_prof.setParameter("vSize", {length});\n')

    script.write(f'var s_mat = shape.insertMaterial("Specular");\n')
    script.write(f's_mat.setParameter("reflectivity", "{rho}");\n')
    script.write(f's_mat.setParameter("slope", "{slope_error}");\n')
    script.write('// --------- --------- --------- --------- --------- --------- --------- --------- ---------\n\n')

    return None


def create_primary_field(script: TextIO, node: Node,
                         centers: array, radii, widths, length: float,
                         rec_aim: array, rho: float, slope_error: float):

    primary_field_node = Node(name='primary_field')
    primary_field_node.as_script(script=script, outer_node=node)

    for i, (hc, w, r) in enumerate(zip(centers, widths, radii)):

        row_node = Node(f'row_{i + 1}', translation=hc)
        row_node.as_script(script=script, outer_node=primary_field_node)

        create_primary_mirror(script=script, node=row_node,
                              width=w, radius=r, length=length,
                              rec_aim=rec_aim, rho=rho, slope_error=slope_error)

    return None


def create_absorber_tube(script: TextIO, node: Node,
                         center: array, axis: array, radius: float, length: float,
                         alpha: float):

    a = nrm(axis)
    v = cross(array([0, 0, 1]), a)
    theta = rad2deg(ang(a, v))

    tube_node = Node(name='absorber_tube',
                     translation=center,
                     rotation=v, angle=theta,
                     scale=array([radius, radius, 1]))
    tube_node.as_script(script=script, outer_node=node)

    script.write(f'var shape = {tube_node.variable_name}.createShape();\n')
    script.write(f'var s_surf = shape.insertSurface("Cylinder");\n')
    script.write(f's_surf.setParameter("caps", "both");\n')
    script.write(f'var s_prof = shape.insertProfile("Box");\n')
    script.write(f's_prof.setParameter("uSize", "360d");\n')
    script.write(f's_prof.setParameter("vSize", {length});\n')

    script.write(f'var s_mat = shape.insertMaterial("Specular");\n')
    script.write(f's_mat.setParameter("reflectivity", "{1 - alpha}");\n')
    script.write(f's_mat.setParameter("slope", "0");\n')

    return None


def create_script(file_name: str, file_path: Path) -> TextIO:

    file_full_path = Path(file_path, f'{file_name}.tnhpps')

    script_file = open(file_full_path, 'w')

    return script_file


def run_tonatiuh(script_path: Path,
                 tonatiuh_path=Path(r"C:\Users\INIESC\AppData\Local\Tonatiuh++\bin\Tonatiuh-Application.exe")):

    cmd = f"{tonatiuh_path}" + ' -i ' + f"{script_path}"
    # cmd = f"{tonatiuh_exe_path}" + ' --help-all'
    return subprocess.run(cmd, shell=True, check=True, capture_output=True)

