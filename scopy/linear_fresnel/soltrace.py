from datetime import datetime
from pathlib import Path

from numpy import array, arange, zeros
from tqdm import tqdm

from scopy import OpticalProperty
from scopy.linear_fresnel import LFR
from scopy.sunlight import RadialSource, sun_direction
from soltracepy import Trace, Optics, Stage, Geometry, ElementStats, soltrace_script, run_soltrace, read_element_stats


def lfr_raytracing_optical_efficiency(lfr: LFR,
                                      theta_t: float, theta_l: float,
                                      aim: array,
                                      length: float,
                                      sun_shape: RadialSource,
                                      primaries_property: OpticalProperty.reflector,
                                      absorber_property: OpticalProperty.flat_absorber,
                                      trace_options: Trace,
                                      files_path: Path, script_name: str,
                                      x_bins: int = 15, y_bins: int = 15):

    # By definition, if one of the incidence angles are +- 90º, the efficiency is set to zero.
    if abs(theta_t) == 90. or abs(theta_l) == 90.:
        optical_efficiency = 0.
    # When incidences are not horizontal
    else:

        # The Sun incidence direction from the transversal and longitudinal incidence angles ####
        sun_dir = sun_direction(theta_t=theta_t, theta_l=theta_l)
        #########################################################################################

        # Transforming Optical.Property objects as Soltrace OpticalSurface objects ###
        primaries_optical_surface = primaries_property.to_soltrace()
        absorber_optical_surface = absorber_property.to_soltrace()
        ##############################################################################

        # Creating the Elements objects: primary mirrors and the flat receiver ########################
        primary_mirrors = lfr.field.to_soltrace(rec_aim=aim, sun_dir=sun_dir,
                                                optic=primaries_optical_surface, length=length)
        absorber = lfr.receiver.as_soltrace_element(length=length, optic=absorber_optical_surface)
        elements = primary_mirrors + absorber
        ###############################################################################################

        # Creating SolTrace objects for main input Boxes ##############################################
        sun = sun_shape.to_soltrace(sun_dir=sun_dir)
        optics = Optics(properties=[primaries_optical_surface, absorber_optical_surface])
        stage = Stage(name='linear_fresnel', elements=elements)
        geometry = Geometry(stages=[stage])

        stats = ElementStats(stats_name=f'absorber_flux_{theta_t}_{theta_l}',
                             stage_index=0, element_index=len(elements) - 1,
                             dni=1000., x_bins=int(x_bins), y_bins=int(y_bins), final_rays=True)
        ##############################################################################################

        # # Checking if trace options follows the sun shape settings ###########################
        # if eff_source.profile == 'collimated':
        #     trace_options.sunshape = False
        # else:
        #     trace_options.sunshape = True
        #
        # if primaries_property.spec_error != 0. or primaries_property.slope_error != 0.:
        #     trace_options.errors = True
        # else:
        #     trace_options.errors = False
        ########################################################################################

        # Creating and writing the codes in a LK script file #########################################
        script_full_path = soltrace_script(file_path=files_path,
                                           file_name=f'{script_name}_{theta_t}_{theta_l}',
                                           sun=sun, optics=optics,
                                           geometry=geometry,
                                           trace=trace_options, stats=stats)
        ##############################################################################################

        # Running the script file with the SolTrace 2012.7.9 version from the prompt #################
        if not stats.file_full_path.is_file():
            run_soltrace(script_full_path)
        ##############################################################################################

        # Reading the stats json file with the main absorber flux data #########################################
        # It then calculates and returns the optical efficiency
        # Here, optical efficiency is defined as the ratio between the absorber flux at the receiver and the
        # incident flux at the mirrors aperture as if the sun was perpendicular to each mirror.

        # OBS: 'linear_fresnel' module measures are in millimeters, and SolTrace works with meters.

        absorber_stats = read_element_stats(stats.file_full_path)
        absorber_flux = absorber_stats['power_per_ray'] * array(absorber_stats['flux']).flatten().sum()
        optical_efficiency = absorber_flux / (stats.dni * lfr.field.widths.sum() * length * 1e-6)
        #########################################################################################################

    return optical_efficiency


def lfr_factorized_analysis(lfr: LFR,
                            aim: array,
                            length: float,
                            sun_shape: RadialSource,
                            primaries_property: OpticalProperty.reflector,
                            absorber_property: OpticalProperty.flat_absorber,
                            trace_options: Trace,
                            files_path: Path, script_name: str,
                            x_bins: int = 15, y_bins: int = 15,
                            symmetric=False):

    lv = 0. if symmetric else -90.
    transversal_angles = arange(lv, 95., 5.)
    longitudinal_angles = arange(0., 95., 5.)

    print(f'Starting transversal optical analysis at {datetime.now()}')
    transversal_efficiencies = [lfr_raytracing_optical_efficiency(lfr=lfr, theta_t=tt, theta_l=0.,
                                                                  aim=aim, length=length,
                                                                  sun_shape=sun_shape,
                                                                  primaries_property=primaries_property,
                                                                  absorber_property=absorber_property,
                                                                  trace_options=trace_options,
                                                                  x_bins=x_bins, y_bins=y_bins,
                                                                  files_path=files_path, script_name=script_name)
                                for tt in tqdm(transversal_angles)]

    print(f'Starting longitudinal optical analysis at {datetime.now()}')
    longitudinal_efficiencies = [lfr_raytracing_optical_efficiency(lfr=lfr, theta_t=0., theta_l=tl,
                                                                   aim=aim, length=length,
                                                                   sun_shape=sun_shape,
                                                                   primaries_property=primaries_property,
                                                                   absorber_property=absorber_property,
                                                                   trace_options=trace_options,
                                                                   x_bins=x_bins, y_bins=y_bins,
                                                                   files_path=files_path, script_name=script_name)
                                 for tl in tqdm(longitudinal_angles)]

    transversal_data = zeros(shape=(transversal_angles.shape[0], 2))
    transversal_data.T[:] = transversal_angles, transversal_efficiencies

    longitudinal_data = zeros(shape=(longitudinal_angles.shape[0], 2))
    longitudinal_data.T[:] = longitudinal_angles, longitudinal_efficiencies

    return transversal_data, longitudinal_data


def lfr_biaxial_analysis(lfr: LFR, aim: array, length: float,
                         sun_shape: RadialSource,
                         primaries_property: OpticalProperty.reflector,
                         absorber_property: OpticalProperty.flat_absorber,
                         trace_options: Trace,
                         files_path: Path, script_name: str,
                         x_bins: int = 15, y_bins: int = 15,
                         symmetric=False):

    lv = 0. if symmetric else -90.
    angles_list = array([[x, y] for x in arange(lv, 95., 5.) for y in arange(0., 95., 5.)])

    print(f'Starting bi-axial optical analysis at {datetime.now()}')
    efficiencies = [lfr_raytracing_optical_efficiency(lfr=lfr, theta_t=pair[0], theta_l=pair[1],
                                                      aim=aim, length=length,
                                                      sun_shape=sun_shape,
                                                      primaries_property=primaries_property,
                                                      absorber_property=absorber_property,
                                                      trace_options=trace_options,
                                                      x_bins=x_bins, y_bins=y_bins,
                                                      files_path=files_path, script_name=script_name)

                    for pair in tqdm(angles_list)]

    biaxial_data = zeros(shape=(angles_list.shape[0], 3))
    biaxial_data.T[:] = angles_list.T[0], angles_list.T[1], efficiencies

    return biaxial_data


def lfr_acceptance_analysis(lfr: LFR,
                            aim: array,
                            theta_t: float,
                            sun_shape: RadialSource,
                            primaries_property: OpticalProperty.reflector,
                            absorber_property: OpticalProperty.flat_absorber,
                            trace_options: Trace,
                            files_path: Path, script_name: str,
                            lv=0.6, dt=0.1,
                            x_bins: int = 15, y_bins: int = 15):

    # Concentrator length #######################################
    # Acceptance analysis are in the transversal plane
    # Thus, it does not matter the length of the concentrator
    length = 4000.
    #############################################################

    # The on-axis optical efficiency (flux at the receiver) ##########################################
    on_axis_eta = lfr_raytracing_optical_efficiency(lfr=lfr, aim=aim, length=length,
                                                    theta_t=theta_t, theta_l=0.,
                                                    sun_shape=sun_shape,
                                                    primaries_property=primaries_property,
                                                    absorber_property=absorber_property,
                                                    trace_options=trace_options,
                                                    files_path=files_path, script_name=script_name,
                                                    x_bins=int(x_bins), y_bins=int(y_bins))
    ##################################################################################################

    # SolTrace objects that does not change in the off-axis incidences ###############################

    # Optical properties
    primaries_optical_surface = primaries_property.to_soltrace()
    absorber_optical_surface = absorber_property.to_soltrace()
    optics = Optics(properties=[primaries_optical_surface, absorber_optical_surface])

    # The Geometry box (primaries and absorber)
    sun_dir = sun_direction(theta_t=theta_t, theta_l=0.)
    primary_mirrors = lfr.field.to_soltrace(rec_aim=aim, sun_dir=sun_dir,
                                            optic=primaries_optical_surface, length=length)
    absorber = lfr.receiver.as_soltrace_element(length=length, optic=absorber_optical_surface)
    elements = primary_mirrors + absorber

    stage = Stage(name='linear_fresnel', elements=elements)
    geometry = Geometry(stages=[stage])
    ##################################################################################################

    # Lists to hold the transmission factor and off-axis incidences data #####
    norm_eta = [1.]
    off_axis_incidence = [0.]
    ##########################################################################

    # Loop for positive off-axis incidences ############################################################################
    k = 1
    while norm_eta[-1] >= lv:

        ###########################################################
        # Off-axis incidence direction ############################
        inc_angle = theta_t + k * dt
        sun_dir = sun_direction(theta_t=inc_angle,
                                theta_l=0.)

        # SolTrace sun object for the off-axis incidence
        sun = sun_shape.to_soltrace(sun_dir=sun_dir)
        ###########################################################

        # ElementStats object for the off-axis incidence ########################
        stats = ElementStats(stats_name=f'absorber_flux_{inc_angle}',
                             stage_index=0,
                             element_index=len(elements) - 1,
                             dni=1000.,
                             x_bins=int(x_bins), y_bins=int(y_bins),
                             final_rays=True)
        #########################################################################

        # Creates and run the LK script file for the off-axis incidence ############
        script_full_path = soltrace_script(file_path=files_path,
                                           file_name=f'{script_name}_{inc_angle}',
                                           sun=sun, optics=optics,
                                           geometry=geometry,
                                           trace=trace_options, stats=stats)
        run_soltrace(script_full_path)
        ############################################################################

        # Reads the stats file and calculate the optical efficiency #######################################
        # Read the absorber stats file
        absorber_stats = read_element_stats(stats.file_full_path)

        # Calculates the optical efficiency
        absorber_flux = absorber_stats['power_per_ray'] * array(absorber_stats['flux']).flatten().sum()
        optical_efficiency = absorber_flux / (stats.dni * lfr.field.widths.sum() * length * 1e-6)
        ###################################################################################################

        # Updating the list of transmission and incidence angles ########
        norm_eta.append(optical_efficiency / on_axis_eta)
        off_axis_incidence.append(inc_angle - theta_t)
        #################################################################

        # updating off-axis incidence ####
        k += 1
        ##################################
    ####################################################################################################################

    # Loop for negative off-axis incidences ############################################################################
    k = 1
    while norm_eta[0] >= lv:

        ###########################################################
        # Off-axis incidence direction ############################
        inc_angle = theta_t - k * dt
        sun_dir = sun_direction(theta_t=inc_angle,
                                theta_l=0.)

        # SolTrace sun object for the off-axis incidence
        sun = sun_shape.to_soltrace(sun_dir=sun_dir)
        ###########################################################

        # ElementStats object for the off-axis incidence ########################
        stats = ElementStats(stats_name=f'absorber_flux_{inc_angle}',
                             stage_index=0,
                             element_index=len(elements) - 1,
                             dni=1000.,
                             x_bins=int(x_bins), y_bins=int(y_bins),
                             final_rays=True)
        #########################################################################

        # Creates and run the LK script file for the off-axis incidence ############
        script_full_path = soltrace_script(file_path=files_path,
                                           file_name=f'{script_name}_{inc_angle}',
                                           sun=sun, optics=optics,
                                           geometry=geometry,
                                           trace=trace_options, stats=stats)
        run_soltrace(script_full_path)
        ############################################################################

        # Reads the stats file and calculate the optical efficiency #######################################
        # Read the absorber stats file
        absorber_stats = read_element_stats(stats.file_full_path)

        # Calculates the optical efficiency
        absorber_flux = absorber_stats['power_per_ray'] * array(absorber_stats['flux']).flatten().sum()
        optical_efficiency = absorber_flux / (stats.dni * lfr.field.widths.sum() * length * 1e-6)
        ###################################################################################################

        # Updating the list of transmission and incidence angles ########
        norm_eta.insert(0, optical_efficiency / on_axis_eta)
        off_axis_incidence.insert(0, inc_angle - theta_t)
        #################################################################

        # updating off-axis incidence ####
        k += 1
        ##################################
    ####################################################################################################################

    # Returning the data as arrays data points ###############
    transmission_data = zeros(shape=(len(norm_eta), 2))
    transmission_data.T[:] = off_axis_incidence, norm_eta
    ##########################################################

    return transmission_data
