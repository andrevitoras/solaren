import subprocess
from pathlib import Path
import io
from numpy import array


class TSun:

    def _init__(self, azimuth: float, altitude: float, shape: str, size: float,
                irradiance=1000.):

        """
        This class is used to hold the data used to confiture the source of rays in Tonatiuh++

        :param azimuth: sun azimuth angle, regarding the North direction (in degrees).
        :param altitude: sun altitude angle (in degrees)
        :param shape: the sunshape profile (Pillbox, Buie, or Gaussian)
        :param size: the size of the sunshape (half-width of pillbox, csr of Buie, or standard deviations of a Gaussian)
        :param irradiance: the source irradiance, in W/m2

        :return: A Tonatiuh++ sun class
        """

        self.azimuth = azimuth
        self.altitude = altitude
        self.irradiance = abs(irradiance)
        self.size = abs(size)

        if shape == 'b' or shape == 'buie' or shape == 'Buie':
            self.shape = '"Buie"'
            self.size_type = 'csr'
        elif shape == 'p' or shape == 'pillbox' or shape == 'Pillbox':
            self.shape = '"Pillbox"'
            self.size_type = 'thetaMax'
        elif shape == 'g' or shape == 'gaussian' or shape == 'Gaussian':
            self.shape = '"Gaussian"'
            self.size_type = 'sigma'
        else:
            raise ValueError("Invalid arguments")

    def as_script(self, script: io.TextIOWrapper):

        """
        A method to write the correspondent script lines to implement the correspondent Tonatiuh source of rays.

        :param script: A IO opened file.
        :return:
        """

        script.write('// Setting the Sun configurations --------------------------\n')
        script.write('var scene = tn.getScene();\n')
        script.write('var sun = scene.getPart("world.sun");\n')
        script.write('var sp = sun.getPart("position");\n')
        script.write(f'sp.setParameter("azimuth", "{self.azimuth}");\n')
        script.write(f'sp.setParameter("altitude", "{self.altitude}");\n')
        script.write(f'sp.setParameter("irradiance", "{self.irradiance}");\n')

        script.write(f'sun.setParameter("shape", "{self.shape}");\n')

        script.write(f'var sunshape = sun.getPart("shape");\n')
        script.write(f'sunshape.setParameter("{self.size_type}", "{self.size}");')
        script.write('// ------------------------------------------------------------\n\n')

        return None


class Node:

    def __init__(self,
                 name: str,
                 translation=array([0, 0, 0]),
                 rotation=array([0, 0, 1]), angle=0,
                 scale=array([1., 1., 1.])):

        self.name = name
        self.translation = translation
        self.rotation = rotation
        self.rot_angle = angle
        self.scale = scale

    def as_script(self, script, outer_node=None):
        # ToDo: include the command for translation and rotation
        if outer_node is None:
            script.write(f'var {self.name} = new ObjectNode;\n')
            script.write(f'{self.name}.setName("{self.name}");\n')
        else:
            script.write(f'var {self.name} = {outer_node.name}.createNode("{self.name}");\n')

        script.write(f'{self.name}.setParameter("translation", "{str(self.translation.tolist())[1:-1]}");\n')
        script.write(f'{self.name}.setParameter("rotation", "{str(self.scale.tolist())[1:-1]} {self.rot_angle}");\n')
        script.write(f'{self.name}.setParameter("scale", "{str(self.scale.tolist())[1:-1]}");\n')

        return None


class Shape:

    class elliptic:

        def __init__(self, ax: float, ay: float, az: float):
            self.ax = ax
            self.ay = ay
            self.az = az

        def as_script(self, node: Node, script: io.TextIOWrapper):

            script.write(f'var shape = {node.name}.createShape();\n')
            script.write(f'var stype = shape.insertSurface("Elliptic");\n')
            script.write(f'stype.setParameter("aX", {self.ax});\n')
            script.write(f'stype.setParameter("aY", {self.ay});\n')
            script.write(f'stype.setParameter("aZ", {self.az});\n')


class Profile:
    pass


