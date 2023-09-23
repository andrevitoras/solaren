"""
Created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
The codes are related to SolTrace commands and scripts to run ray-tracing simulations.

They are related to writing a script file (.lk) which configures, trace rays and export the simulation results.
They include all steps needed to perform a ray trace simulation: (1) Sun configuration (rays source), (2) surfaces
optical properties, (3) geometrical elements definitions,(4) rays configurations, (5) export results.
"""

import json
import os
import subprocess
from pathlib import Path

from numpy import array, zeros, pi

soltrace_paths = {'strace': Path('C:\\SolTrace\\3.1.0\\x64\\strace.exe'),
                  'soltrace2': Path('C:\\SolTrace\\2012.7.9\\SolTrace.exe'),
                  'soltrace3': Path('C:\\SolTrace\\3.1.0\\x64\\soltrace.exe')}


# The 'soltrace3' key refers to the path of the version 3.1.0, available in the NREL website.
# This version is a GUI which# cannot run an LK script from the prompt.
# The 'soltrace2' key refers to an GUI old version (2012.7.9) of Soltrace# which can run an LK script from the prompt.
# This version was sent by Thomas Fasquelle. Finally, the 'soltrace3' is a CLI version that can only run stinput files
# from the prompt following a precise command structure.
# For more details, please see the Notion documentation.

# ToDo: Complete the class, methods, and functions documentations.

#######################################################################################################################
# Classes  ############################################################################################################


class SoltraceSun:

    def __init__(self, sun_dir: array, profile: str = None, size: float = None, user_data: array = None):

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

    def as_script(self, script):
        """
        This method writes the lines of code of an LK script to define the Sun box of SolTrace.

        :param script: A file object to append the LK lines of code.
        :return:
        """
        # By definition, this function does not consider a point source at a finite distance, neither uses Latitude,
        # Day, and Hour option -- this is why 'ptsrc'=false, and 'useldh' = false, respectively.

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

    def as_stinput(self, file):
        """
        This function writes the lines of code related to a stinput file which configures a SolTrace Sun.

        :param file: The stinput file in which the code will be written
        :return:
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


class OpticInterface:
    """
    An OpticInterface object is the main element of an Optical Surface, and is the core element of the Optics box.
    It defines the front or back side of an OpticalSurface object. Its instances are related to the input data needed.

    Its methods implement the lines of code for both a LK script or STINPUT file
    """
    # ToDo: Check how to implement an angular variable reflectivity

    def __init__(self, name: str, reflectivity: float, transmissivity: float, slp_error=0., spec_error=0.,
                 front=True, real_refractive_index=1.0, img_refractive_index=1.2):

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

    def as_script(self, script):
        script.write("// ---- Set Surface Property  -------------------\n")
        # Adds a surface optical property with a given name
        script.write(f"opticopt('{self.name}', {self.side}, " + "{'dist'='g',\n")
        script.write(f"'refl'={self.ref}, 'trans'={self.tran}, 'errslope'={self.err_slop},'errspec'={self.err_spec},\n")
        script.write(f"'refractr'={self.n_real}, 'refracti'={self.n_img}, 'apstop'={self.ap_stop},\n")
        script.write(f"'difford'={self.diff_ord}, 'grating'={list(self.grt_cff)}" + "});\n")
        script.write("// --------------------------------------------\n\n")

    def as_stinput(self, file):
        self.st_parameters[0:3] = self.ap_stop, self.surf_num, self.diff_ord
        self.st_parameters[3:7] = self.ref, self.tran, self.err_slop, self.err_spec
        self.st_parameters[7:9] = self.n_real, self.n_img
        self.st_parameters[9:13] = self.grt_cff

        file.write(f"OPTICAL.v2\tg")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write(f"\n")


class OpticalSurface:
    """
    An OpticalSurface object is the main element of the Optics box of SolTrace.

    An OpticalSurface has two sides: front and back. The front side is defined by the positive direction of the z-axis
    of the Element Coordinate System (ECS) of the SolTrace Element in which the property is applied to.

    """

    def __init__(self, name: str, front_side: OpticInterface, back_side: OpticInterface):
        self.name = name
        self.front = front_side
        self.back = back_side

    def as_script(self, script):
        """
        A method to write this object as the correspondent lines of code of an LK script.

        :param script: The script file to write the LK code
        """

        script.write("// ---- Add Surface Property -----------------------------------------------\n\n")
        script.write(f"addoptic('{self.name}');\n")
        self.front.as_script(script=script)
        self.back.as_script(script=script)

        script.write("// -------------------------------------------------------------------------\n\n")

    def as_stinput(self, file):
        """
        A method to write this object as the correspondent lines of code of a Soltrace input file.

        :param file:
        """
        file.write(f"OPTICAL PAIR\t{self.name}\n")
        self.front.as_stinput(file=file)
        self.back.as_stinput(file=file)


class Optics:
    """
    The class Optics represents the "Optics" box of SolTrace. It should contain a list of OpticalProperties objects to
    be included in a LK script or in a STINPUT file.
    """

    def __init__(self, properties: list):
        self.properties = properties

    def as_script(self, script):

        script.write(f"// ---- Setting the Optics box ----------------------------------------------------------- \n\n")
        script.write(f"clearoptics();\n")

        for prop in self.properties:
            prop.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

    def as_stinput(self, file):

        file.write(f"OPTICS LIST COUNT    {len(self.properties)}\n")
        for prop in self.properties:
            prop.as_stinput(file=file)


class Element:
    """
    The Element class stand to hold the data for the definition of a SolTrace Element in the Geometry box.

    The Element is defined in a Stage and has a coordinate system attached to it: the Element Coordinate System (ECS).
    Obviously, the ECS is defined regarding the Stage Coordinate System (SCS) by the direction of its z-axis and a
    rotation about this axis (z-rot).
    The z-axis of the ECS is defined by the vector from the ECS ecs_origin to the ECS aim-point, so that:
    z = 'ecs_aim_pt' - 'ecs_origin'. Of course, these points and vectors are defined in the SCS.
    """

    def __init__(self, name: str, ecs_origin: array, ecs_aim_pt: array, z_rot: float,
                 aperture: list, surface: list, optic: OpticalSurface, reflect=True, enable=True):

        """

        :param name:
        :param ecs_origin: The Element Coordinate System (ECS) ecs_origin, a point-array.
        :param ecs_aim_pt: The Element Coordinate System (ECS) aim-point, a point-array.
        :param z_rot:
        :param aperture:
        :param surface:
        :param optic:
        :param reflect: A boolean sign to indicate if the Element is reflective (True) or refractive (False)
        :param enable: A boolean sign to indicate if the Element is to be considered (True) or not (False) in the
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

        self.EN = 1 if enable else 0
        self.INT = 2 if reflect else 1

        self.st_parameters = list(ecs_origin) + list(ecs_aim_pt) + list([z_rot]) + self.aperture

    def as_script(self, script, el_index: int):
        script.write(f"// -- Add an element to the current stage -----------------\n\n")
        script.write(f"addelement();\n")  # It appends a new element in the current stage
        script.write(
            f"elementopt({el_index}, " + "{" + f"'en'={self.en}, " + f"'x'={self.x}, 'y'={self.y}, 'z'={self.z},\n")
        script.write(f"'ax'={self.ax}, 'ay'={self.ay}, 'az'={self.az}, 'zrot'={self.z_rot},\n")
        script.write(f"'aper'={self.aperture},\n")
        script.write(f"'surf'={self.surface}, 'interact'='{self.interaction}',\n")
        script.write(f"'optic'='{self.optic_name}', 'comment'='{self.name}'" + "});\n")
        script.write(f"// --------------------------------------------------------\n\n")

    def as_stinput(self, file):

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


class Stage:
    """
    The Stage class stand to hold the data for the definition of a SolTrace Stage in the Geometry box.

    The Stage is defined regarding the Global Coordinate System (GCS) and has a coordinate system attached to it:
    the Stage Coordinate System (SCS).
    The SCS is defined regarding the GCS by the direction of its z-axis and a rotation about this axis (z-rot).
    The z-axis of the SCS is defined by the vector from the SCS ecs_origin to the SCS aim-point, so that:
    z = 'scs_aim_pt' - 'scs_origin'. Of course, these points and vectors are defined in the GCS.

    """

    def __init__(self, name: str, elements: list, scs_origin=array([0, 0, 0]), scs_aim_pt=array([0, 0, 1]), z_rot=0,
                 active=True, virtual=False, rays_multi_hit=True, trace_through=False):
        """

        :param name:
        :param elements: A list with Element objects within the Stage.
        :param scs_origin:
        :param scs_aim_pt:
        :param z_rot:
        :param active:
        :param virtual:
        :param rays_multi_hit:
        :param trace_through:
        """

        for el in elements:
            if type(el) != Element:
                raise "A non Element object was added in the elements argument of the Stage instance"

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
        VT = 1 if virtual else 0
        MH = 1 if rays_multi_hit else 0
        TT = 1 if trace_through else 0

        self.st_parameters += ['VIRTUAL', VT] + ['MULTIHIT', MH] + ['ELEMENTS', len(elements)] + ['TRACETHROUGH', TT]

    def as_script(self, script):

        if len(self.elements) == 0:
            raise "The Stage has no elements to interact with the rays. Please, add Elements to the elements argument"

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

    def as_stinput(self, file):
        file.write(f"STAGE\tXYZ")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write("\n")
        file.write(f"{self.name}\n")

        for el in self.elements:
            el.as_stinput(file=file)


class Geometry:

    def __init__(self, stages: list):
        self.stages = stages

    def as_script(self, script):
        script.write(f"// ---- Setting the Geometry box --------------------------------------------------------- \n\n")
        script.write(f"clearstages();\n")

        for stg in self.stages:
            stg.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

    def as_stinput(self, file):
        file.write(f"STAGE LIST COUNT \t{len(self.stages)}\n")

        for stg in self.stages:
            stg.as_stinput(file=file)


class Trace:
    """
    This class stands to hold all data needed to configurate and trace a SolTrace ray-tracing simulation.

    """

    def __init__(self, rays: float, cpus: int, seed=-1, sunshape=True, optical_errors=True, point_focus=False,
                 simulate=True):
        """

        :param rays:
        :param cpus:
        :param seed:
        :param sunshape:
        :param optical_errors:
        :param point_focus:
        """

        self.rays = int(rays)
        self.max_rays = 100 * self.rays
        self.cpus = int(cpus)
        self.sunshape = 'true' if sunshape else 'false'
        self.errors = 'true' if optical_errors else 'false'
        self.seed = seed
        self.point_focus = 'true' if point_focus else 'false'

        self.simulate = simulate

    def as_script(self, script):
        script.write('// ---- Setting the Ray Tracing simulation -------------------------------------------------\n\n')

        script.write('traceopt({ \n')
        script.write(f"'rays'={self.rays}, 'maxrays'={self.max_rays},\n")
        script.write(f"'include_sunshape'={self.sunshape}, 'optical_errors'={self.errors},\n")
        script.write(f"'cpus'={self.cpus}, 'seed'={self.seed}, 'point_focus'={self.point_focus}" + "});\n")

        if self.simulate:
            script.write('trace();\n')

        script.write(f"//-----------------------------------------------------------------------------------------\n\n")


class ElementStats:

    def __init__(self, stats_name: str, stage_index: int, element_index: int,
                 dni=1000, x_bins=15, y_bins=15, final_rays=True):
        """
        This class holds the data to select an Element from a Stage and to export its rays (flux) stats data after a
        simulation and then export it to a json file -- see elementstats() LK function.
        This json file is easily read as a Python dictionary.

        :param stage_index: The index number of the Stage (the first Stage has an index of 0).
        :param element_index: The index number of the element within the Stage (the first Element has an index of 0).
        :param stats_name: The name of the json file to be exported with the element stats.

        :param dni: The solar direct normal irradiance, in W/m2.
        :param x_bins: The number of grid elements in the x-axis to split the element surface.
        :param y_bins: The number of grid elements in the y-axis to split the element surface.
        """

        self.name = stats_name
        self.stg_index = abs(stage_index)
        self.ele_index = abs(element_index)

        self.dni = abs(dni)
        self.x_bins = abs(x_bins)
        self.y_bins = abs(y_bins)
        self.final_rays = 'true' if final_rays else 'false'

        self.file_full_path = None

    def as_script(self, script, file_path: Path, soltrace_version=2012):
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

        # Setting the SolTrace current working directory as the file_path.
        script.write("cwd('" + str([str(file_path)])[2:-2] + "');\n")

        # For the SolTrace 3.1.0 version #############################################################
        # SolTrace 2012 version does not have the LK function 'json_file()' ##########################
        if soltrace_version != 2012:
            script.write(f"json_file('{self.name}_stats.json', absorber_data);\n")
            script.write(f"//--------------------------------------------------------------------\n\n")

        # For the 2012 SolTrace version ##############################################################
        else:
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
            script.write('float_keys = ["min_flux", "bin_size_x", "power_per_ray", "bin_size_y", "peak_flux",\n')
            script.write('"ave_flux", "radius", "num_rays"];\n')
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


########################################################################################################################
########################################################################################################################


# def surface_as_spline_file(x_values: array, y_values: array, file_name: str, file_path: Path):
#
#     file_full_path = Path(file_path, f'{file_name}.csi')
#
#     file = open(file_full_path, 'w')
#     file.write()
#
#     return file_full_path


########################################################################################################################
# Sun functions ########################################################################################################

# def sun2soltrace(sun: Sun):
#
#     return SoltraceSun(sun_dir=sun.sun_vector, profile=sun.sun_shape.profile, size=sun.sun_shape.size)


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Optics functions #####################################################################################################


def reflective_surface(name: str, rho: float, slope_error: float, spec_error: float):

    front = OpticInterface(name=name, reflectivity=rho, transmissivity=0,
                           slp_error=slope_error, spec_error=spec_error, front=True)
    back = OpticInterface(name=name, reflectivity=0, transmissivity=0,
                          slp_error=slope_error, spec_error=spec_error, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def secondary_surface(name: str, rho: float, slope_error: float, spec_error: float):
    front = OpticInterface(name=name, reflectivity=rho, transmissivity=0,
                           slp_error=slope_error, spec_error=spec_error, front=True)
    back = OpticInterface(name=name, reflectivity=rho, transmissivity=0,
                          slp_error=slope_error, spec_error=spec_error, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def flat_absorber_surface(name: str, alpha: float):
    front = OpticInterface(name=name, reflectivity=1 - alpha, transmissivity=0, front=True)
    back = OpticInterface(name=name, reflectivity=1, transmissivity=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def absorber_tube_surface(name: str, alpha: float):
    front = OpticInterface(name=name, reflectivity=1 - alpha, transmissivity=0, front=True)
    back = OpticInterface(name=name, reflectivity=1 - alpha, transmissivity=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def transmissive_surface(name: str, tau: float, nf: float, nb: float):

    front = OpticInterface(name=name, reflectivity=0, transmissivity=tau, real_refractive_index=nf, front=True,
                           spec_error=0.92, slp_error=0.2)
    back = OpticInterface(name=name, reflectivity=0, transmissivity=1, real_refractive_index=nb, front=False,
                          spec_error=0.92, slp_error=0.2)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def cover_surfaces(tau: float, name='cover', refractive_index=1.52):

    # tau_s = tau**0.5

    out_surf = transmissive_surface(name=f'{name}_outer_cover', tau=tau, nf=refractive_index, nb=1.0)
    inn_surf = transmissive_surface(name=f'{name}_inner_cover', tau=1, nf=1.0, nb=refractive_index)

    return out_surf, inn_surf

#######################################################################################################################
#######################################################################################################################

########################################################################################################################
# Geometry functions ###################################################################################################


def flat_element(name: str, ecs_origin: array, ecs_aim_pt: array, width: array, length: float,
                 optic: OpticalSurface, reflect=True, enable=True):

    aperture = list([0] * 9)
    aperture[0:3] = 'r', width, length

    surface = list([0] * 9)
    surface[0] = 'f'

    elem = Element(name=name, ecs_origin=ecs_origin, ecs_aim_pt=ecs_aim_pt, z_rot=0,
                   optic=optic, aperture=aperture, surface=surface,
                   reflect=reflect, enable=enable)

    return elem


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Script functions #####################################################################################################

def create_script(file_path, file_name='optic'):
    """
    This function creates a lk script file with the inputted name. This file will be created at the current work
    directory if a full path is not inputted at the file_name parameter -- the standard directory in Python.

    :param file_path: THe Path where the script should be created.
    :param file_name: The name of the lk script file to be created.

    :return: It returns an opened LK file where the lines of code will be writen.
    """

    full_file_path = Path(file_path, f"{file_name}.lk")
    script = open(full_file_path, 'w')
    script.write('// ----------------- This set of commands will configure a SolTrace LK script ----------------\n\n\n')

    return script


def soltrace_script(file_path: Path, file_name: str, sun: SoltraceSun, optics: Optics, geometry: Geometry,
                    trace: Trace, stats: ElementStats):

    """
    This functions creates a full SolTrace LK script file.

    :param file_path: The path of the LK script file.
    :param file_name: The name of the LK script file.
    :param sun: A Sun object, refers to the SolTrace Sun bax.
    :param optics: An Optics object, refers to the SolTrace Optics box
    :param geometry: A Geometry object, refers to the SolTrace Geometry box, with Stages and Elements.
    :param trace: A Trace object, refers to the SolTrace Trace box
    :param stats: A ElementStats objects, refers to the rays and flux stats of an Element within a Stage.

    :return: The full path of the LK script.
    """

    # Check if the file path is an existing directory #####
    # If not, it creates the directory.
    if not file_path.is_dir():
        os.makedirs(file_path)
    ######################################################

    # Created the script file and keep it open ###########

    # The function 'create_script' uses this code to create the file.
    script_full_path = Path(file_path, f"{file_name}.lk")

    script = create_script(file_path=file_path,
                           file_name=file_name)
    ######################################################

    # Write the SolTrace objects as the correspondent script codes ###########################################
    # It also close the script file in order to enable its usage by other programs (e.g., the SolTrace exe

    sun.as_script(script=script)
    optics.as_script(script=script)
    geometry.as_script(script=script)
    trace.as_script(script=script)
    stats.as_script(script=script, file_path=file_path)
    script.close()
    ###########################################################################################################

    return script_full_path


def run_soltrace(lk_file_full_path: Path, soltrace_path=Path(f"C:\\SolTrace\\2012.7.9\\SolTrace.exe")):

    """
    This function opens the SolTrace version from 2012 and runs the LK script.

    :param lk_file_full_path: The full path of the LK file.
    :param soltrace_path: The full path of the SolTrace executable file.
    """

    cmd = f"{soltrace_path}" + ' -s ' + f"{lk_file_full_path} -h"
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def read_element_stats(full_file_path: Path):
    """
    This function reads the 'json' ElementStats file and returns it as a dictionary.

    :param full_file_path: The full path of the exported json file with the Element stats.

    :return: A dictionary with the element stats.
    """

    with open(full_file_path, encoding='utf-8') as data_file:
        stats = json.loads(data_file.read())

    return stats


def create_csi_file(curve_pts: array, knots_derivatives: array, file_path: Path, file_name: str):
    """
    This function creates the file used by SolTrace to construct an element by a rotationally symmetric cubic spline.
    The arguments of this function must be in meters, as SolTraces work with such unit.

    :param curve_pts: The curve points as [x, y] point-arrays.
    :param knots_derivatives: The first derivatives at the first and last points of the curve.
    :param file_path: The path in which to create the spline file.
    :param file_name: The name of the spline file.

    :return: The full path of the spline file.
    """

    # Curve points and knots derivatives ######
    x_values, y_values = curve_pts.T
    df_1, df_n = knots_derivatives
    ###########################################

    # Creating the rotationally symmetric cubic spline file path #####
    # A 'csi' extension file for SolTrace to correctly read it
    full_file_path = Path(file_path, f"{file_name}.csi")
    ##################################################################

    # Creating file and writing the data into it ################################################################
    file = open(full_file_path, 'w')
    # The first line must contain the number of points which defines the surface
    file.write(f"{x_values.shape[0]}\n")
    for x, y in zip(x_values, y_values):
        # write in the file the point coordinates values in meters
        file.write(f"{x}\t{y}\n")

    # the last line should contain the first derivatives at both edge knots.
    file.write(f"{df_1}\t{df_n}")  # writes the first derivatives at both edges knots
    file.close()  # closes the file
    ############################################################################################################

    return full_file_path


# def script_change_element_aim(script, element_index: int, aim_vector: array):
#     ax, ay, az = aim_vector
#     script.write(f"elementopt({element_index}, " + "{" + f"'ax'={ax}, 'ay'={ay}, 'az'={az}" + "});\n")
#
#     pass


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Input files functions ################################################################################################


def create_stinput(file_path, file_name='optic'):
    full_file_path = Path(file_path, f"{file_name}.stinput")
    file = open(full_file_path, 'w')
    file.write("# SOLTRACE VERSION 3.1.0 INPUT FILE\n")

    return file


def soltrace_file(file_path: Path, file_name: str, sun: SoltraceSun, optics: Optics, geometry: Geometry):
    if not file_path.is_dir():
        os.makedirs(file_path)

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
        print("The path passed as argument is not of a file")

#######################################################################################################################
#######################################################################################################################


class ElementFlux:
    """
    This class stands to hold the data from an element stats file.

    The elementstats() LK function returns an LK table -- which is similar to a Python dictionary -- with the rays data
    from a selected Element within a Stage.
    See the class ElementStats() and the function read_element_stats() in this module.

    """

    def __init__(self, stats_file: Path):

        # An attribute to hold the file path ####
        self.file_path = stats_file
        #########################################

        # Reading the stats as a dictionary #########
        stats = read_element_stats(self.file_path)
        #############################################

        # The size of the bins in x and y axes ################################
        self.x_bin = stats['bin_size_x']
        self.y_bin = stats['bin_size_y']

        # radial size of the x-bin, in radians
        self.radius = stats['radius']
        self.r_bin = (180/pi) * (self.x_bin / self.radius) if self.radius > 0 else 0

        # The area of a bin, in m2
        self.bin_area = self.x_bin * self.y_bin
        ######################################################################

        # The axis coordinates of the center of the bins ########
        self.x_values = array(stats['xvalues'])
        self.y_values = array(stats['yvalues'])

        # Radial coordinates of the bins, in radians
        self.r_values = (180/pi) * (self.x_values / self.radius) \
            if self.radius > 0 \
            else zeros(self.x_values.shape[0])
        #########################################################

        # The power that each ray carry #############
        self.ray_power = stats['power_per_ray']
        #############################################

        # The total flux #################################
        # In Watts
        self.flux = stats['num_rays'] * self.ray_power
        ##################################################

        # Rays per bin #######################################
        self.rays = array(stats['flux'])
        ######################################################

        # Flux per bin #######################################
        # Values are in Watts
        self.flux_map = self.rays * self.ray_power

        ######################################################

        # SolTrace metrics for the flux distribution #########
        self.average_flux = stats['ave_flux']
        self.sigma_flux = stats['sigma_flux']
        self.uniformity = stats['uniformity']
        ######################################################

    def concentration_map(self, dni: float):
        """
        A method to calculate the concentration map in the Element.
        Actually, it returns the flux concentration factor per bin.

        :param dni: The direct normal irradiance, in W/m2.

        :return: An array of arrays.
        """

        return self.flux_map / self.bin_area / dni
