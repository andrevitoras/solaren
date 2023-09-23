"""
Created by André Santos: andrevitoras@gmail.com / avas@uevora.pt
"""
import json
from copy import deepcopy
from pathlib import Path

from numpy import zeros, sqrt, array, ndarray
from pandas import DataFrame

from niopy.geometric_transforms import R

########################################################################################################################
########################################################################################################################


def closest(my_list, my_number):
    return min(my_list, key=lambda x: abs(x - my_number))


def chop(expr, *, maxi=1.e-6):
    return [i if abs(i) > maxi else 0 for i in expr]


def arrays_to_contour(data: DataFrame, x_col: str, y_col: str, z_col: str):

    df = data.sort_values(by=[x_col, y_col])
    x = df[x_col].unique()
    y = df[y_col].unique()
    z = df[z_col].values.reshape(len(x), len(y)).T

    return x, y, z


def read_trnsys_tmy2(file):

    lines = open(file, 'r').readlines()
    headers = ['Time', 'DNI [W/m2]', 'GHI [W/m2]', 'Solar Zenith [degrees]', 'Solar Azimuth [degrees]']

    data = zeros(shape=(len(lines) - 1, len(headers)))

    for i, file_line in enumerate(lines):

        if i == 0:
            continue
        else:
            data[i - 1][:] = [float(elem) for elem in file_line.split()]

    df = DataFrame(data, columns=headers)

    return df


def rmse(predictions: array, targets: array):
    return sqrt(((predictions - targets) ** 2).mean())


def dic2json(d: dict, file_path: Path = None, file_name: str = None):

    dict_to_export = deepcopy(d)
    keys = d.keys()

    for k in keys:
        if isinstance(d[k], dict):
            dict_to_export[k] = dic2json(d=d[k])
        elif isinstance(d[k], ndarray):
            dict_to_export[k] = d[k].tolist()
        else:
            dict_to_export[k] = d[k]

    if file_path is not None and file_name is not None:

        file_full_path = Path(file_path, f"{file_name}.json")
        with open(file_full_path, 'w') as file:
            json.dump(dict_to_export, file)

    return dict_to_export


def plot_line(a, b):
    return array([a[0], b[0]]), array([a[-1], b[-1]])


def rotate2D(points: array, center: array, tau: float):

    rm = R(alpha=tau)
    translated_pts = points - center
    rotated_pts = rm.dot(translated_pts.T).T + center

    return rotated_pts
