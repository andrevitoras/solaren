

import subprocess
from pathlib import Path


#######################################################################################################################
#######################################################################################################################


def change_parameters(template_path: Path, file_path: Path, file_name: str, keys: list, values: list):
    """
    This function modify the desired keywords in a dck file to the desired values
    :param template_path: The full path of the template dck file to be modified.
    :param file_path: the path of where the generated dck file should be created
    :param file_name: the variable_name of the generated dck file
    :param keys: the tagged keywords that will be changed
    :param values: the values that will replace the tagged keywords.
    :return:
    """

    # check if the template input file is a 'dck' file
    if str(template_path)[-3:] != "dck":
        raise "A 'dck' file was not given as the template argument"

    # check if the number of keys to be replaced equals to the number of given values
    if len(keys) != len(values):
        raise "The number of keys to be replaced is different from the number of given values"

    # opens the template file and copy its content to a variable 'file' and closes the template file
    with open(template_path, 'r') as template:
        file_e = template.read()

    # check for keys and replace it with the new values.
    for old, new in zip(keys, values):
        file_e = file_e.replace(old, str(new))

    file_full_path = Path(file_path, f"{file_name}.dck")
    # created the output file with the replaced values.
    with open(file_full_path, 'w') as file_out:
        file_out.write(file_e)

    return file_full_path


def run_trnsys(file_full_path: Path, hide=True, version=17):
    """
    This function runs a TRNSYS dck file.
    :param file_full_path:
    :param hide:
    :param version:

    :return:
    """

    if version == 17:
        trnsys_path = 'C:\\Trnsys17\\Exe\\TRNExe.exe'
    elif version == 18:
        trnsys_path = 'C:\\TRNSYS18\\Exe\\TrnEXE.exe'

    else:
        raise ValueError('Please, select a valible TRNSYS version: 17 or 18')

    if hide:
        cmd = f'{trnsys_path} {file_full_path} /h'
    else:
        cmd = f'{trnsys_path} {file_full_path}'

    subprocess.run(cmd, shell=True, check=True, capture_output=True)

########################################################################################################################


# def sf_cost_reduction_analysis(csp_data: DataFrame, pvts_data: DataFrame, csp_cost_model: TS_CostModel, location: str,
#                                nbr_points=50):
#
#     sf_area_base_load = 546000
#
#     csp_df = csp_data[csp_data['Location'] == location]
#     csp_lowest_lcoe = csp_df['LCOE (USD/MWhe)'].min()
#
#     pvts_df = pvts_data[pvts_data['Location'] == location]
#     pvts_lowest_lcoe = pvts_df['LCOE (USD/MWhe)'].min()
#
#     csp_base_settings = csp_df[csp_df['LCOE (USD/MWhe)'] == csp_lowest_lcoe]
#     sm = csp_base_settings['Solar Multiple']
#     flh = csp_base_settings['Storage FLH']
#     sf2grid = csp_base_settings['SF2Grid (MWhe)']
#     tes2grid = csp_base_settings['TES2Grid (MWhe)']
#
#     ts_system = TS(flh=flh,
#                    turbine_gross=115.0,
#                    turbine_net=100.0,
#                    pb_eff=0.428)
#
#     new_cost_model = deepcopy(csp_cost_model)
#
#     output_data = zeros(shape=(nbr_points, 2))
#     for i, sf_new_cost in enumerate(linspace(start=csp_cost_model.sf_ec, stop=0, num=nbr_points)):
#
#         new_cost_model.sf_ec = sf_new_cost
#
#         power_plant = PTPP(sf_area=sm * sf_area_base_load,
#                            ts=ts_system,
#                            ts_cost_model=new_cost_model)
#
#         new_lcoe = power_plant.economic_assessment(sf2grid=sf2grid,
#                                                    tes2grid=tes2grid,
#                                                    pv_degradation=0.4 / 100,
#                                                    ts_degradation=0.4 / 100,
#                                                    discount_rate=7 / 100,
#                                                    inflation_rate=4 / 100,
#                                                    n=25)
#
#         output_data[i] = sf_new_cost, new_lcoe
#
#     return output_data, pvts_lowest_lcoe
#
#
# def csp_cost_to_pvts_lcoe(csp_cost_variation: array, pvts_lowest_lcoe: float):
#
#     csp_base_sf_cost = csp_cost_variation.T[0][0]
#
#     sf_lcoe_function = interp1d(x=csp_cost_variation.T[0], y=csp_cost_variation.T[1], kind='linear')
#
#     def y(x): return sf_lcoe_function(x) - pvts_lowest_lcoe
#
#     csp_eq_sf_cost = bisect(f=y, a=csp_cost_variation.T[0][0], b=csp_cost_variation.T[0][-1])
#
#     relative_cost_reduction = 100 * (csp_base_sf_cost - csp_eq_sf_cost) / csp_base_sf_cost
#
#     return round(relative_cost_reduction, 1)
#
#
# def sf_reduction_per_location(location: str, csp_data: DataFrame, pvts_data: DataFrame, csp_cost_model: TS_CostModel):
#
#     csp_cost_variation, pvts_lowest_lcoe = sf_cost_reduction_analysis(location=location,
#                                                                          csp_data=csp_data, pvts_data=pvts_data,
#                                                                          csp_cost_model=csp_cost_model)
#
#     relative_cost_reduction = csp_cost_to_pvts_lcoe(csp_cost_variation=csp_cost_variation,
#                                                    pvts_lowest_lcoe=pvts_lowest_lcoe)
#
#     return relative_cost_reduction
#
#
# def sf_cost_reductions(csp_data: DataFrame, pvts_data: DataFrame, csp_cost_model: TS_CostModel):
#
#     locations = list(set(csp_data['Location']))
#
#     output = zeros(shape=(len(locations), 3))
#
#     for i, loca in enumerate(locations):
#
#         csp_df = csp_data[csp_data['Location'] == loca]
#
#         dni = csp_df['DNI [kWh/m2.yr]'].mean()
#         ghi = csp_df['GHI [kWh/m2.yr]'].mean()
#
#         sf_reduction = sf_reduction_per_location(location=loca,
#                                                  csp_data=csp_data,
#                                                  pvts_data=pvts_data,
#                                                  csp_cost_model=csp_cost_model)
#
#         output[i] = ghi, dni/ghi, sf_reduction
#
#     return output
