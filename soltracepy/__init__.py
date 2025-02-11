"""
Created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
The codes are related to SolTrace commands and scripts to run ray-tracing simulations.

This module is related to writing SolTrace scripts (.lk) and input files (.stinput) which configures and run the
ray-tracing simulations, as well as exporting the selected results. Therefore, it includes defining SolTrace main boxes:
(1) Sun: rays source.
(2) Optics: surfaces optical properties.
(3) Geometry: geometrical elements definitions.
(4) Trace: rays configurations.
(5) Intersections: export results.

It is important to highlight that SolTrace units are defined in meters and milliradians.
"""

import json
import subprocess
from pathlib import Path
from typing import TextIO

from numpy import array, pi

from pysoltrace import PySolTrace, Point

soltrace_paths = {'strace': Path('C:\\SolTrace\\3.1.0\\x64\\strace.exe'),
                  'soltrace2': Path('C:\\SolTrace\\2012.7.9\\SolTrace.exe'),
                  'soltrace3': Path('C:\\SolTrace\\3.1.0\\x64\\soltrace.exe')}


########################################################################################################################
# Classes  #############################################################################################################


class Sun:
    """
    This class does not consider a point source at a finite distance, neither uses Latitude,  Day, and Hour option
    Therefore, 'ptsrc'=false, and 'useldh' = false, respectively.
    """
    def __init__(self, sun_dir: array, profile: str = None, size: float = None, user_data: array = None):
        """

        :param sun_dir: an [x,y,z] array to represent the sun vector
        :param profile: the radiance profile representing sunbeam divergence ('pillbox', 'gaussian', or 'user' defined)
        :param size: the representative size of sunbeam, in milliradians. Half-width for a 'pillbox' and

        :param user_data: for a 'user' defined profile, an array of values with the normalized radiance profile,
        such as [[0, 1.], [theta_2, value_2], ..., [theta_n, value_n]], where theta is in milliradians.
        """

        if profile == 'gaussian' or profile == 'g':
            self.shape = "'g'"
            self.sigma = abs(size)
        elif profile == 'pillbox' or profile == 'p':
            self.shape = "'p'"
            self.sigma = abs(size)
        elif profile is None:
            self.shape = "'p'"
            self.sigma = 0.05
        elif profile == 'user' or profile == 'u':
            self.shape = "'u'"
            self.values = user_data
        else:
            raise ValueError("Please, input a valid profile: "
                             "'gaussian', 'pillbox', or 'user' with 'user_data' argument!")

        self.vector = sun_dir

    def as_script(self, script: TextIO):
        """
        This method writes the lines of code of an LK script to define the Sun box of SolTrace.

        :param script: a file object to append the LK lines of code.
        :return: None
        """

        script.write("// ---- Setting the Sun (source of rays) options  -----------------------\n\n")
        script.write('sunopt({\n')
        script.write(f"'useldh'=false, 'ptsrc'=false,\n")
        script.write(f"'x'={self.vector[0]}, 'y'={self.vector[1]}, 'z'={self.vector[2]},\n")
        # Set the sun radiance profile and its size (standard deviation or half-width)
        if self.shape == "'g'":
            script.write(f"'shape'={self.shape}, 'sigma'={self.sigma}" + "});\n")
        elif self.shape == "'p'":
            script.write(f"'shape'={self.shape}, 'halfwidth'={self.sigma}" + "});\n")
        else:
            # A user defined sun profile (e.g., Buie's sun shape)
            script.write(f"'shape'={self.shape},\n")
            # The value used for this sun shape. It should be as a Python list to print it correctly in the script file.
            script.write(f"'userdata'={self.values.tolist()}" + "});\n")

        script.write("// -----------------------------------------------------------------------\n\n")

        return None

    def as_stinput(self, file: TextIO):
        """
        This function writes the lines of code related to a stinput file which configures a SolTrace Sun.

        :param file: the stinput file in which the code will be written
        :return: None
        """

        if self.shape == "'g'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE g \t SIGMA {self.sigma} \t HALFWIDTH 0\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA 0\n")
        elif self.shape == "'p'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE p \t SIGMA 0 \t HALFWIDTH  {self.sigma}\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA 0\n")
        elif self.shape == "'u'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE d \t SIGMA 0 \t HALFWIDTH  0\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA {len(self.values)}\n")
            for v in self.values:
                file.write(f"{v[0]}\t{v[1]}\n")
        else:
            raise 'Error in the inputted SolTrace Sun (Source of rays) data'

        return None

    # def to_api(self):
    #
    #     # initializing the corresponding object
    #     sun = PySolTrace.Sun()
    #     # sun vector
    #     sun.position.x, sun.position.y, sun.position.z = self.vector
    #     # Sunshape and its size
    #     sun.shape = self.shape[1:-1]
    #     if sun.shape == 'u':
    #         sun.user_intensity_table = self.values.tolist()
    #     else:
    #         sun.sigma = self.sigma
    #
    #     return sun


class OpticInterface:
    """
    An OpticInterface object is the main element of an OpticalSurface object, and is the core element of the Optics box.
    It is used for the settings of front or back sides of an optical property of the Optics box.
    """

    def __init__(self,
                 name: str,
                 reflectivity: float, transmissivity: float,
                 slp_error=0., spec_error=0.,
                 front=True,
                 real_refractive_index=1.0, img_refractive_index=1.0):

        """
        :param name: the name of the optical property.
        :param reflectivity: the reflectivity value, between 0 and 1.
        :param transmissivity: the transmissivity value, between 0 and 1.

        :param slp_error: the surface slope error, in milliradians.
        :param spec_error: the surface specular error, in milliradians.

        :param front: a boolean sign to indicate for a front (True) or back (False) side.
        :param real_refractive_index:
        :param img_refractive_index:
        """

        self.name = name
        self.ref = reflectivity
        self.tran = transmissivity
        self.err_slop = slp_error
        self.err_spec = spec_error
        self.n_real = real_refractive_index
        self.n_img = img_refractive_index
        self.side = 1 if front else 2
        self.ap_stop = 3  # Value from SolTrace GUI. I do not know what this parameter means.
        self.surf_num = 1  # Value from SolTrace GUI. I do not know what this parameter means.
        self.diff_ord = 4  # Value from SolTrace GUI. I do not know what this parameter means.
        self.grt_cff = [1.1, 1.2, 1.3, 1.4]  # static value from SolTrace GUI. I do not know what this parameter means.

        self.st_parameters = [0] * 15  # useful to construct the stinput code lines

    def as_script(self, script: TextIO):
        script.write("// ---- Set Interface -------------------\n")
        # Adds a surface side (front or back) optical property with a given name
        script.write(f"opticopt('{self.name}', {self.side}, " + "{'dist'='g',\n")
        script.write(f"'refl'={self.ref}, 'trans'={self.tran}, 'errslope'={self.err_slop},'errspec'={self.err_spec},\n")
        script.write(f"'refractr'={self.n_real}, 'refracti'={self.n_img}, 'apstop'={self.ap_stop},\n")
        script.write(f"'difford'={self.diff_ord}, 'grating'={list(self.grt_cff)}" + "});\n")
        script.write("// --------------------------------------------\n\n")

        return None

    def as_stinput(self, file: TextIO):
        self.st_parameters[0:3] = self.ap_stop, self.surf_num, self.diff_ord
        self.st_parameters[3:7] = self.ref, self.tran, self.err_slop, self.err_spec
        self.st_parameters[7:9] = self.n_real, self.n_img
        self.st_parameters[9:13] = self.grt_cff

        file.write(f"OPTICAL.v2\tg")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write(f"\n")

        return None

    def to_api(self):

        # initializing the corresponding object
        face = PySolTrace.Optics.Face()

        # setting properties of the object
        face.reflectivity = self.ref
        face.transmissivity = self.tran
        face.slope_error = self.err_slop
        face.spec_error = self.err_spec

        return face


class OpticalSurface:
    """
    An OpticalSurface object is used to represent the main element of the Optics box of SolTrace, which has front and
    back interfaces.

    It is important to highlight that the front side is defined by the positive direction of the z-axis of the Element
    Coordinate System (ECS) of the Element in which the property is applied to.
    """

    def __init__(self, name: str, front_side: OpticInterface, back_side: OpticInterface):
        self.name = name
        self.front = front_side
        self.back = back_side

    def as_script(self, script: TextIO):
        """
        A method to write this object as the correspondent lines of code of an LK script.
        :param script: the script file to write the LK code
        :return None
        """

        script.write("// ---- Add Surface Optical Property -----------------------------------------------\n\n")
        script.write(f"addoptic('{self.name}');\n")
        self.front.as_script(script=script)
        self.back.as_script(script=script)

        script.write("// -------------------------------------------------------------------------\n\n")

        return None

    def as_stinput(self, file: TextIO):
        """
        A method to write this object as the correspondent lines of code of a Soltrace input file.

        :param file:
        """
        file.write(f"OPTICAL PAIR\t{self.name}\n")
        self.front.as_stinput(file=file)
        self.back.as_stinput(file=file)

        return None

    def to_api(self, index_id: int):
        optic = PySolTrace.Optics(id=index_id)
        optic.name = self.name

        optic.front = self.front.to_api()
        optic.back = self.back.to_api()

        return optic


class Optics:
    """
    The class Optics represents the "Optics" box of SolTrace. It should contain a list of OpticalProperties objects to
    be included in a LK script or in a STINPUT file.
    """

    def __init__(self, properties: list):

        assert all([isinstance(prop, OpticalSurface) for prop in properties]),\
            "A non OpticalSurface object was added in the properties argument of the Optics class!!!"

        self.properties = properties

    def as_script(self, script: TextIO):

        script.write(f"// ---- Setting the Optics box ----------------------------------------------------------- \n\n")
        script.write(f"clearoptics();\n")

        for prop in self.properties:
            prop.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

        return None

    def as_stinput(self, file: TextIO):

        file.write(f"OPTICS LIST COUNT    {len(self.properties)}\n")
        for prop in self.properties:
            prop.as_stinput(file=file)

        return None

    def to_api(self):

        api_properties = [opt_property.to_api(index_id=i) for i, opt_property in enumerate(self.properties)]

        return api_properties


class Element:
    """
    The Element class stand to hold the data for the definition of a SolTrace Element of a Stage in the Geometry box.

    The Element is defined in a Stage and has a coordinate system attached to it: the Element Coordinate System (ECS).
    Obviously, the ECS is defined regarding the Stage Coordinate System (SCS) by the direction of its z-axis and a
    rotation about this axis (z-rot). The z-axis of the ECS is defined by the vector from the ECS origin to the
    ECS aim-point, so that:
    z = 'ecs_aim_pt' - 'ecs_origin'.
    Of course, these points and vectors are defined in the SCS. Thus, the definition of the ECS z-axis, together with a
    rotation around it, the z-rot argument, completely defines the relative orientation of the ECS regarding the SCS.
    """

    def __init__(self,
                 name: str,
                 ecs_origin: array, ecs_aim_pt: array, z_rot: float,
                 aperture: list, surface: list,
                 optic: OpticalSurface,
                 reflect=True, enable=True):

        """
        :param name: the name of the Element object, to be inserted in the comment argument.

        :param ecs_origin: an [x,y,z] array representing the ECS origin.
        :param ecs_aim_pt: an [x,y,z] array representing the ECS aim-pt.
        :param z_rot: the rotation around the z-axis of the ECS, in degrees.

        :param aperture: a list with the settings of the element aperture.
        :param surface: a list with the settings of the element surface.

        :param optic: the optical property that rules the interaction of rays with the element.

        :param reflect: a boolean sign to indicate if the Element is reflective (True) or refractive (False)
        :param enable: a boolean sign to indicate if the Element is to be considered (True) or not (False) in the
        ray-tracing simulation.
        """

        self.name = name
        self.x, self.y, self.z = ecs_origin
        self.ax, self.ay, self.az = ecs_aim_pt
        self.z_rot = z_rot
        self.aperture = aperture
        self.surface = surface
        self.optic_name = optic.name
        self.interaction = "reflection" if reflect else "refraction"
        self.en = 'true' if enable else 'false'

        # auxiliary attributes to make it easy to write the code lines of an input file
        self.EN = 1 if enable else 0
        self.INT = 2 if reflect else 1
        self.st_parameters = list(ecs_origin) + list(ecs_aim_pt) + list([z_rot]) + self.aperture

    def as_script(self, script: TextIO, el_index: int):
        script.write(f"// -- Add an element to the current stage -----------------\n\n")
        script.write(f"addelement();\n")  # It appends a new element in the current stage
        script.write(
            f"elementopt({el_index}, " + "{" + f"'en'={self.en}, " + f"'x'={self.x}, 'y'={self.y}, 'z'={self.z},\n")
        script.write(f"'ax'={self.ax}, 'ay'={self.ay}, 'az'={self.az}, 'zrot'={self.z_rot},\n")
        script.write(f"'aper'={self.aperture},\n")
        script.write(f"'surf'={self.surface}, 'interact'='{self.interaction}',\n")
        script.write(f"'optic'='{self.optic_name}', 'comment'='{self.name}'" + "});\n")
        script.write(f"// --------------------------------------------------------\n\n")

    def as_stinput(self, file: TextIO):

        file.write(f"{self.EN}")
        for p in self.st_parameters:
            file.write(f"\t{p}")

        if len(self.surface) != 2:
            for p in self.surface:
                file.write(f"\t{p}")
            file.write(f"\t\t{self.optic_name}")
        else:
            s, path = self.surface
            surface_parameters = list([s]) + list([0] * 8)

            for p in surface_parameters:
                file.write(f"\t{p}")

            file.write(f"\t{path}")
            file.write(f"\t{self.optic_name}")

        file.write(f"\t{self.INT}")
        file.write(f"\t{self.name}\n")

    def to_api(self, parent_stage: PySolTrace.Stage, element_id: int, optic_prop: PySolTrace.Optics):

        api_element = PySolTrace.Stage.Element(parent_stage=parent_stage, element_id=element_id)

        api_element.enabled = self.EN
        api_element.interaction = self.INT

        api_element.position = Point(x=self.x, y=self.y, z=self.z)
        api_element.aim = Point(x=self.ax, y=self.ay, z=self.az)
        api_element.zrot = self.z_rot

        api_element.aperture = self.aperture[0]
        api_element.aperture_params = self.aperture[1:]

        api_element.surface = self.surface[0]
        if len(self.surface) == 2:
            api_element.surface_file = self.surface[1]
        else:
            api_element.surface_params = self.surface[1:]

        api_element.optic = optic_prop

        return api_element


class Stage:
    """
    The Stage class stand to hold the data for the definition of a SolTrace Stage in the Geometry box.

    A Stage is defined regarding the Global Coordinate System (GCS) and has a coordinate system attached to it:
    the Stage Coordinate System (SCS).
    The SCS is defined regarding the GCS by the direction of its z-axis and a rotation about this axis (z-rot).
    The z-axis of the SCS is defined by the vector from the SCS origin to the SCS aim-point, so that:
    z = 'scs_aim_pt' - 'scs_origin'.
    Of course, these points and vectors are defined in the GCS. Thus, the definition of the SCS z-axis, together with a
    rotation around it, the z-rot argument, completely defines the relative orientation of the SCS regarding the GCS.
    """

    def __init__(self,
                 name: str,
                 elements: list,
                 scs_origin=array([0, 0, 0]), scs_aim_pt=array([0, 0, 1]), z_rot=0,
                 active=True, virtual=False, rays_multi_hit=True, trace_through=False):
        """

        :param name: a name to represent the Stage
        :param elements: a list of Element objects that are defined in the Stage.

        :param scs_origin: an [x,y,z] array representing the SCS origin.
        :param scs_aim_pt: an [x,y,z] array representing the SCS aim-pt.
        :param z_rot: the rotation around the z-axis of the SCS, in degrees.

        :param active: a boolean sign to set as an active Stage (True) or not (False).
        :param virtual: a boolean sign to set if rays interact with elements in the Stage (True) or not (False).

        :param rays_multi_hit:
        :param trace_through:
        """

        assert all([isinstance(el, Element) for el in elements]),\
            "A non Element object was added in the elements argument of the Stage class"

        self.name = name
        self.x, self.y, self.z = scs_origin
        self.x_aim, self.y_aim, self.z_aim = scs_aim_pt
        self.z_rot = z_rot
        self.virtual = 'true' if virtual else 'false'
        self.multi_hit = 'true' if rays_multi_hit else 'false'
        self.trace_through = 'true' if trace_through else 'false'
        self.active = active
        self.elements = elements

        self.st_parameters = list(scs_origin) + ['AIM'] + list(scs_aim_pt) + ['ZROT', z_rot]
        self.VT = 1 if virtual else 0
        self.MH = 1 if rays_multi_hit else 0
        self.TT = 1 if trace_through else 0

        self.st_parameters += ['VIRTUAL', self.VT] + ['MULTIHIT', self.MH] + ['ELEMENTS', len(elements)] + ['TRACETHROUGH', self.TT]

    def as_script(self, script: TextIO):

        if len(self.elements) == 0:
            raise ValueError("The Stage has no elements to interact with the rays. "
                             "Please, add Elements to the elements argument")

        script.write("// ---- Setting a Stage for the Geometries to be added --------------\n\n")
        script.write(f"addstage('{self.name}');\n")
        script.write(f"stageopt('{self.name}', " + "{" + f"'x'={self.x}, 'y'={self.y}, 'z'={self.z},\n")
        script.write(f"'ax'={self.x_aim}, 'ay'={self.y_aim}, 'az'={self.z_aim}, 'zrot'={self.z_rot},\n")
        script.write(f"'virtual'={self.virtual}, 'multihit'={self.multi_hit}, 'tracethrough'={self.trace_through}")
        script.write("});\n")

        if self.active:
            script.write(f"activestage('{self.name}');\n")

        for i, el in enumerate(self.elements):
            el.as_script(script=script, el_index=i)

        script.write("// -----------------------------------------------------------------\n\n")

        return None

    def as_stinput(self, file: TextIO):
        file.write(f"STAGE\tXYZ")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write("\n")
        file.write(f"{self.name}\n")

        for el in self.elements:
            el.as_stinput(file=file)

        return None

    def to_api(self, stage_id: int, elements: list):

        api_stage = PySolTrace.Stage(id=stage_id)

        api_stage.name = self.name

        api_stage.position = Point(x=self.x, y=self.y, z=self.z)
        api_stage.aim = Point(x=self.x_aim, y=self.y_aim, z=self.z_aim)
        api_stage.zrot = self.z_rot

        api_stage.is_virtual = True if self.VT == 1 else False
        api_stage.is_multihit = True if self.MH == 1 else False
        api_stage.is_tracethrough = True if self.TT == 1 else False

        api_stage.elements = elements

        return api_stage


class Geometry:

    def __init__(self, stages: list):
        self.stages = stages

    def as_script(self, script):
        script.write(f"// ---- Setting the Geometry box --------------------------------------------------------- \n\n")
        script.write(f"clearstages();\n")

        for stg in self.stages:
            stg.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

        return None

    def as_stinput(self, file):
        file.write(f"STAGE LIST COUNT \t{len(self.stages)}\n")

        for stg in self.stages:
            stg.as_stinput(file=file)

        return None


class Trace:
    """
    This class stands to hold all data needed to configurate and trace a SolTrace ray-tracing simulation.

    """

    def __init__(self, rays: float, cpus: int,
                 seed=-1,
                 sunshape=True, optical_errors=True, point_focus=False,
                 simulate=True):
        """
        :param rays: the desired number of rays intersections.
        :param cpus: the number of cpus to be used in the simulation
        :param seed: the seed number of the Monte Carlo routine of random numbers

        :param sunshape: a boolean sign to consider (True) or not (False) the sunshape.
        :param optical_errors: a boolean sign to consider (True) or not (False) optical errors.
        :param point_focus: a boolean sign to consider (True) or not (False) as a point focus optic.
        :param simulate: a boolean sign to run (True) or not (False) the simulation.
        """

        self.rays = int(rays)
        self.max_rays = 150 * self.rays
        self.cpus = int(cpus)
        self.sunshape = 'true' if sunshape else 'false'
        self.errors = 'true' if optical_errors else 'false'
        self.seed = seed
        self.point_focus = 'true' if point_focus else 'false'

        self.simulate = simulate

    def as_script(self, script):
        script.write('// ---- Setting the Ray-tracing simulation -------------------------------------------------\n\n')

        script.write('traceopt({ \n')
        script.write(f"'rays'={self.rays}, 'maxrays'={self.max_rays},\n")
        script.write(f"'include_sunshape'={self.sunshape}, 'optical_errors'={self.errors},\n")
        script.write(f"'cpus'={self.cpus}, 'seed'={self.seed}, 'point_focus'={self.point_focus}" + "});\n")

        if self.simulate:
            script.write('trace();\n')

        script.write(f"//-----------------------------------------------------------------------------------------\n\n")

        return None


class ElementStats:

    def __init__(self,
                 stats_name: str,
                 stage_index: int, element_index: int,
                 dni=1000., x_bins=15, y_bins=15,
                 final_rays=True):
        """
        This class holds the data to select an Element from a Stage and to export its rays (flux) stats data after a
        simulation and then export it to a json file -- see elementstats() LK function.
        This json file is easily read as a Python dictionary.

        :param stage_index: The index number of the Stage (the first Stage has an index of 0).
        :param element_index: The index number of the element within the Stage (the first Element has an index of 0).
        :param stats_name: The variable_name of the json file to be exported with the element stats.

        :param dni: The solar direct normal irradiance, in W/m2.
        :param x_bins: The number of grid elements (>=2) in the ECS x-axis to split the element surface.
        :param y_bins: The number of grid elements (>=2) in the ECS y-axis to split the element surface.
        """

        self.name = stats_name
        self.stg_index = abs(stage_index)
        self.ele_index = abs(element_index)
        self.dni = abs(dni)

        # the bins in x and y directions of the Element Coordinate System
        # they should be >=2, otherwise, the elementstats() functions returns a <null> table.
        self.x_bins = abs(x_bins) if abs(x_bins) >= 2 else 2
        self.y_bins = abs(y_bins) if abs(y_bins) >= 2 else 2
        self.final_rays = 'true' if final_rays else 'false'

        self.file_full_path = None

    def as_script(self, script, file_path: Path, soltrace_version=2012):
        # ToDo: include the simulation time in the LK Table exported as a json file.
        """
        This method writes the script lines to collect the information of an Element in a particular Stage and then
        export this to a json file.
        Rays (flux) data of the Element are collected by the LK function elementstats().

        This method also returns the full path of the exported json file.

        This method cannot be written as commands of a 'stinput' file but just as a 'LK' script.

        :param script: A script file to append the lines of code.
        :param file_path: The path of the file to be exported with the flux results.
        :param soltrace_version:

        :return It returns the full path of the exported json file with the element stats.
        """

        element_stats_full_path = Path(file_path, f"{self.name}_stats.json")
        self.file_full_path = element_stats_full_path

        script.write('//---- Setting the results outputs ----------------------------------------\n\n')
        # Set the source irradiance to compute flux calculations. A standard value of 1000 W/m2 was chosen.
        script.write(f"absorber_data = elementstats({self.stg_index}, {self.ele_index}, ")
        script.write(f"{self.x_bins}, {self.y_bins}, {self.dni}, {self.final_rays});\n\n")
        script.write("absorber_data{'generated_rays'} = sundata(){'nrays'};\n")

        # Setting the SolTrace current working directory as the file_path.
        script.write("cwd('" + str([str(file_path)])[2:-2] + "');\n")

        # For the SolTrace 3.1.0 version #############################################################
        # SolTrace 2012 version does not have the LK function 'json_file()' ##########################
        if soltrace_version != 2012:
            script.write(f"json_file('{self.name}_stats.json', absorber_data);\n")
            script.write(f"//--------------------------------------------------------------------\n\n")

        # For the 2012 SolTrace version ##############################################################
        else:
            # The 2012.7.9 version of SolTrace does not have any functions that writes an LK-table as a json file
            # Therefore, this set of codes are needed to write this flux stats file.

            # Writing a json file to export the element stats
            script.write(f"filename = '{self.name}_stats.json';\n")
            script.write("stats_file = open(filename, 'w');\n")

            # The starting-braces to define a Python dictionary in the json file
            script.write("write_line(stats_file, '{');\n\n")

            # Old implementation #######################################################################################
            # # Selecting some keys from the LK tabel 'absorber_data' previously defined. These are float data keys.
            # # These keys will be exported to the json file.
            # script.write('float_keys = ["min_flux", "bin_size_x", "power_per_ray", "bin_size_y", "peak_flux",\n')
            # script.write('"sigma_flux", "peak_flux_uncertainty", "uniformity", "ave_flux",\n')
            # script.write('"ave_flux_uncertainty", "radius", "num_rays"];\n')
            ############################################################################################################

            # New implementation 21-Apr-2023 ###########################################################################
            # Selecting some keys from the LK tabel 'absorber_data' previously defined. These are float data keys.
            # These keys will be exported to the json file.
            script.write('float_keys = ["generated_rays", "min_flux", "bin_size_x", '
                         '"power_per_ray", "bin_size_y", "peak_flux",\n')
            script.write('"ave_flux", "sigma_flux", "radius", "num_rays"];\n')
            ############################################################################################################

            # Writing the LK code to write in the json file the keys and values of the LK-Table ########################
            # Each key is writen in a line.
            script.write("for (i=0; i<#float_keys; i++)\n")
            script.write("{\n")
            script.write("write_line(stats_file, " + "'" + '"' + "'" + ' + float_keys[i] + ' + "'" + '": ' + "'")
            script.write(" + absorber_data{float_keys[i]} + ',');\n")
            script.write("}\n\n")

            # Writing the LK code to write in the json file the vector keys and values of the LK-Table #################
            # Each vector-key is writen in a line.
            script.write("vector_keys = ['centroid', 'xvalues', 'yvalues'];\n")
            script.write("for (i=0; i<#vector_keys; i++)\n")
            script.write("{\n")
            script.write("write_line(stats_file, " + "'" + '"' + "'" + ' + vector_keys[i] + ' + "'" + '": [' + "'")
            script.write(" + absorber_data{vector_keys[i]} + '],');\n")
            script.write("}\n\n")

            # Writing the LK code to write the 'flux' key of the elementstats LK-Table.
            script.write("rays_bins = absorber_data{'flux'};\n")
            script.write("for (i = 0; i < #rays_bins; i++)")
            script.write("{\n")

            script.write("\tif (i==0)\n")
            script.write("\t{\n")
            script.write('\tstring_to_write = ' + "'" + '"flux": [[' + "' + rays_bins[i] + '],';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")

            script.write("\telseif (i > 0 && i < #rays_bins - 1)\n")
            script.write("\t{\n")
            script.write("\tstring_to_write = '[' + rays_bins[i] + '],';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")

            script.write("\telse\n")
            script.write("\t{\n")
            script.write("\tstring_to_write = '[' + rays_bins[i] + ']]\\n';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")
            script.write("}\n")

            # The end-braces to define the Python dictionary and closing the exported file from the Soltrace.
            script.write("write_line(stats_file, '}');\n")
            script.write("close(stats_file);\n")
            script.write(f"//---------------------------------------------------------------------\n\n")
            ############################################################################################################

        return element_stats_full_path


class ElementFlux:
    """
    This class stands to hold the data from an element stats file.

    The elementstats() LK function returns an LK table -- which is similar to a Python dictionary -- with the rays data
    from a selected Element within a Stage.

    See the class ElementStats() and the function read_element_stats() in this module.
    """

    def __init__(self, stats_file: Path):

        # It should be noticed that units in SolTrace are in meters (m) and Watts (W).

        # Saving input data as attributes #######################
        # An attribute to hold the file path
        self.file_path = stats_file

        # saving stats file (json) as dictionary
        self.stats = read_element_stats(self.file_path)
        #########################################################

        # Other attributes ########################################
        # number of generated rays
        self.generated_rays = self.stats['generated_rays'] if 'generated_rays' in self.stats.keys() else None

        # surface radius
        self.radius = self.stats['radius']

        # the power, in kW, carried by each ray
        self.ray_power = self.stats['power_per_ray'] / 1000.

        # simulations running time, if exist
        self.runtime = self.stats['sim_time'] if 'sim_time' in self.stats.keys() else None
        ###########################################################

        # Bin attributes ######################################################
        # the size of the element surface bin
        self.xbin_size = self.stats['bin_size_x']  # in m
        self.ybin_size = self.stats['bin_size_y']  # in m
        self.bin_area = self.xbin_size * self.ybin_size  # in m2

        # position of the bins center
        self.xbin_values = array(self.stats['xvalues'])
        self.ybin_values = array(self.stats['yvalues'])
        self.rbin_values = (180/pi) * (self.xbin_values / self.radius) if self.radius > 0 else None
        #######################################################################

        # Flux attributes #######################################################
        # total flux, in KW
        self.total_flux = self.stats['num_rays'] * self.ray_power

        # the flux, in kW, per bin
        self.flux_per_bin = array(self.stats['flux']).T * self.ray_power

        # the flux density distribution per bin, in kW/m2
        self.flux_map = self.flux_per_bin / self.bin_area

        # the average flux, in kW/m2
        self.flux_mean = self.flux_map.mean()

        # the flux standard deviation, in kW/m2
        self.flux_std = self.flux_map.std()

        # uniformity index
        self.uniformity = self.flux_std / self.flux_mean if self.flux_mean > 0. else 0.

        # flux in the x-axis of the Element Coordinate System
        self.flux_x_distribution = self.flux_map.mean(axis=0)
        self.flux_x_std = self.flux_x_distribution.std()
        self.flux_x_uniformity = self.flux_x_std / self.flux_mean if self.flux_mean > 0 else None

        # Flux in the y-axis of the Element Coordinate System
        self.flux_y_distribution = self.flux_map.mean(axis=1)
        self.flux_y_std = self.flux_y_distribution.std()
        self.flux_y_uniformity = self.flux_y_std / self.flux_mean if self.flux_mean > 0 else None
        ###########################################################################

        # SolTrace metrics for the flux distribution ###########
        # self.average_flux = self.stats['ave_flux']
        # self.sigma_flux = self.stats['sigma_flux']
        # self.uniformity = self.stats['uniformity']
        #########################################################

        ##########################################################################
        # OLD IMPLEMENTATION #####################################################

        # # The size of the bins in x and y axes ################################
        # self.x_bin = self.stats['bin_size_x']
        # self.y_bin = self.stats['bin_size_y']
        #
        # # radial size of the x-bin, in radians
        # self.radius = self.stats['radius']
        # self.r_bin = (180/pi) * (self.x_bin / self.radius) if self.radius > 0 else 0
        #
        # # The area of a bin, in m2
        # self.bin_area = self.x_bin * self.y_bin
        # ######################################################################
        #
        # # The axis coordinates of the center of the bins ########
        # self.x_values = array(self.stats['xvalues'])
        # self.y_values = array(self.stats['yvalues'])
        #
        # # Radial coordinates of the bins, in radians
        # self.r_values = (180/pi) * (self.x_values / self.radius) \
        #     if self.radius > 0 \
        #     else zeros(self.x_values.shape[0])
        # #########################################################
        #
        # # The power that each ray carry #############
        # self.ray_power = self.stats['power_per_ray']
        # #############################################
        #
        # # The total flux #################################
        # # in Watts (W)
        # self.flux = self.stats['num_rays'] * self.ray_power
        # ##################################################
        #
        # # Rays per bin #######################################
        # # the number of rays that strikes each bin
        # self.rays = array(self.stats['flux'])
        # ######################################################
        #
        # # Flux intensity per bin #######################################
        # # Values are in Watts per square meter (W/m2)
        # self.flux_map = self.rays * self.ray_power / self.bin_area
        #
        # ######################################################
        #
        # # SolTrace metrics for the flux distribution #########
        # self.average_flux = self.stats['ave_flux']
        # self.sigma_flux = self.stats['sigma_flux']
        # self.uniformity = self.stats['uniformity']
        # ######################################################
        ##########################################################################
        ##########################################################################

    def concentration_map(self, dni: float):
        """
        A method to calculate the concentration map in the Element.
        Actually, it returns the flux concentration factor per bin.

        :param dni: The direct normal irradiance, in W/m2.

        :return: An array of arrays.
        """

        return self.flux_map / self.bin_area / dni


########################################################################################################################
########################################################################################################################


########################################################################################################################
# Script functions #####################################################################################################


def rays_file(script: TextIO):
    # ToDo: Create the LK lines of code to export a rays file as a plain text file.
    #  It only has a function to export it as a binary file.

    script.write('intersections = nintersectc();\n')
    script.write('for (i = 0; i < intersections; i++) {\n')
    script.write('')
    script.write('};')


def create_script(file_path, file_name='optic') -> TextIO:
    """
    This function creates a lk script file with the inputted variable_name.
    This file will be created at the current work directory if a full path is not inputted at the file_name parameter --
     the standard directory in Python.

    :param file_path: THe Path where the script should be created.
    :param file_name: The variable_name of the lk script file to be created.

    :return: It returns an opened LK file where the lines of code will be writen.
    """

    full_file_path = Path(file_path, f"{file_name}.lk")
    script = open(full_file_path, 'w')
    script.write('// ----------------- This set of commands will configure a SolTrace LK script ----------------\n\n\n')

    return script


def soltrace_script(file_path: Path, file_name: str,
                    sun: Sun,
                    optics: Optics,
                    geometry: Geometry,
                    trace: Trace,
                    stats: ElementStats,
                    version=2012) -> Path:

    """
    This functions creates a full SolTrace LK script file.

    :param file_path: the path of the LK script file.
    :param file_name: the variable_name of the LK script file.
    :param sun: a Sun object, refers to the SolTrace Sun bax.
    :param optics: an Optics object, refers to the SolTrace Optics box
    :param geometry: a Geometry object, refers to the SolTrace Geometry box, with Stages and Elements.
    :param trace: a Trace object, refers to the SolTrace Trace box
    :param stats: an ElementStats object, refers to the rays and flux stats of an Element within a Stage.
    :param version: an argument to indicate which version (2012 or 3.0) of SolTrace will run the script.

    :return: The full path of the LK script.
    """

    # Check if the file path is an existing directory #####
    file_path.mkdir(parents=True, exist_ok=True)
    ######################################################

    # Created the script file and keep it open ###########
    # The function 'create_script' uses this code to create the file.
    script_full_path = Path(file_path, f"{file_name}.lk")

    script = create_script(file_path=file_path,
                           file_name=file_name)
    ######################################################

    # Write the SolTrace objects as the correspondent script codes ###########################################
    sun.as_script(script=script)
    optics.as_script(script=script)
    geometry.as_script(script=script)
    trace.as_script(script=script)
    stats.as_script(script=script, file_path=file_path, soltrace_version=version)

    # Close the script file in order to enable its usage by other programs.
    script.close()
    ###########################################################################################################

    return script_full_path


def run_soltrace(lk_file_full_path: Path, soltrace_path=Path(f"C:\\SolTrace\\2012.7.9\\SolTrace.exe")):
    """
    This functions runs a SolTrace LK script from the windows cmd. It does not work for the SolTrace 3.01 version,
    currently available at NREL's website. However, it works for the 2012.7.9 version.

    :param lk_file_full_path: the full path of the LK script file.
    :param soltrace_path: the full path of the SolTrace executable file.

    :return None
    """

    cmd = f"{soltrace_path}" + ' -s ' + f"{lk_file_full_path}"
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

    return None


def read_element_stats(full_file_path: Path):
    """
    This function reads the 'json' ElementStats file and returns it as a dictionary.

    :param full_file_path: The full path of the exported json file with the Element stats.

    :return: A dictionary with the element stats.
    """

    with open(full_file_path, encoding='utf-8') as data_file:
        stats = json.loads(data_file.read())

    return stats


#######################################################################################################################
########################################################################################################################


def run_api(sun: PySolTrace.Sun,
            optics: list,
            stages: list,
            trace_options: Trace):

    ok_optics = [isinstance(o, PySolTrace.Optics) for o in optics]
    ok_stages = [isinstance(s, PySolTrace.Stage) for s in stages]

    assert False in ok_optics, 'Please, check the classes in the optics list'
    assert False in ok_stages, 'Please, check the classes in the stages list'

    # creating an PySolTrace object
    pyst = PySolTrace()

    # setting configurations regarding the inclusion of sunshape and surface errors
    pyst.is_sunshape = True if trace_options.sunshape == 'true' else False
    pyst.is_surface_errors = True if trace_options.errors == 'true' else False

    # setting the sun
    pyst.sun = sun

    # including the optics and stages to the PySoltrace object
    pyst.optics = optics
    pyst.stages = stages

    # option to run as a point focus system
    as_power_tower = True if trace_options.point_focus == 'true' else False

    pyst.run(seed=trace_options.seed, as_power_tower=as_power_tower, nthread=trace_options.cpus)

    return pyst


########################################################################################################################
# Input files functions ################################################################################################


def create_stinput(file_path, file_name='optic'):
    full_file_path = Path(file_path, f"{file_name}.stinput")
    file = open(full_file_path, 'w')
    file.write("# SOLTRACE VERSION 3.1.0 INPUT FILE\n")

    return file


def soltrace_file(file_path: Path, file_name: str,
                  sun: Sun, optics: Optics, geometry: Geometry):

    file_path.mkdir(parents=True, exist_ok=True)

    st_file = create_stinput(file_path=file_path, file_name=file_name)

    sun.as_stinput(file=st_file)
    optics.as_stinput(file=st_file)
    geometry.as_stinput(file=st_file)
    st_file.close()

    full_file_path = Path(file_path, f'{file_name}.stinput')

    return full_file_path


def run_strace(file_path, trace: Trace, strace_path='C:\\SolTrace\\3.1.0\\x64\\strace.exe'):
    """
    :param file_path: Full path of a .stinput file to be run.
    :param trace: An object Trace. It contains the data to run the simulations.
    :param strace_path: Full path of the SolTrace CLI version, i.e., the strace.exe file.

    :return: This function runs the CLI version of the SolTrace and also runs the .stinput file. At the end, it creates
    a ray data file -- as the CSV file that can be saved from the SolTrace GUI version.
    """

    ss = 1 if trace.sunshape == 'true' else 0
    er = 1 if trace.errors == 'true' else 0
    pf = 1 if trace.point_focus == 'true' else 0

    if file_path.is_file():
        cmd = f'{strace_path} {str(file_path)} {trace.rays} {trace.max_rays} {trace.seed} {ss} {er} {pf}'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    else:
        print("The path passed as argument is not of a file!")

########################################################################################################################
########################################################################################################################
