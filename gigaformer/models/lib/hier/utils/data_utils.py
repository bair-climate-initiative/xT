# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import numpy as np
import torch

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
DEFAULT_PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for level in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(level)] = (
            NAME_TO_VAR[var] + "_" + str(level)
        )

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    "NorthAmerica": {"lat_range": (15, 65), "lon_range": (220, 300)},  # 8x14
    "SouthAmerica": {"lat_range": (-55, 20), "lon_range": (270, 330)},  # 14x10
    "Europe": {"lat_range": (30, 65), "lon_range": (0, 40)},  # 6x8
    "SouthAsia": {"lat_range": (-15, 45), "lon_range": (25, 110)},  # 10, 14
    "EastAsia": {"lat_range": (5, 65), "lon_range": (70, 150)},  # 10, 12
    "Australia": {"lat_range": (-50, 10), "lon_range": (100, 180)},  # 10x14
    "Global": {"lat_range": (-90, 90), "lon_range": (0, 360)},  # 32, 64
}


def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region["lat_range"]
    lon_range = region["lon_range"]
    lat = lat[::-1]  # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (
        (lat_matrix >= lat_range[0])
        & (lat_matrix <= lat_range[1])
        & (lon_matrix >= lon_range[0])
        & (lon_matrix <= lon_range[1])
    )
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (
                (i >= h_from)
                & (i + p - 1 <= h_to)
                & (j >= w_from)
                & (j + p - 1 <= w_to)
            ):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        "patch_ids": valid_patch_ids,
        "min_h": min_h,
        "max_h": max_h,
        "min_w": min_w,
        "max_w": max_w,
    }


def parse_variable_groups(variables, variable_groups):
    """
    Args:
        variables: List of all variables
        variable_groups: dict[str][List[List[str]]]
    Outputs:
        outL dict[str][List[List[idx]]]
    """
    variable_group_dict = {}
    for gp_name, gp_variables in variable_groups.items():
        variable_group_dict[gp_name] = [
            np.array([variables.index(x) for x in lst]) for lst in gp_variables
        ]
    return variable_group_dict


class VariableGroup:
    def __init__(self, variable_group_dict) -> None:
        self.variable_group_dict = variable_group_dict

    def __call__(self, x) -> Any:
        """
        inputs: .... X C X H X W
        outputs: .... X C_GND X 1 X H X W,
                 .... X C_LEVEL X Pressure_Level X H X W
        """
        if not self.variable_group_dict:
            return x  # do nothing
        out_dict = {}
        for gp_name, gp_variables in self.variable_group_dict.items():
            out_dict[gp_name] = torch.stack(
                [x[..., lst, :, :] for lst in gp_variables], -4
            )

        return out_dict


def parse_lookbacks(lookbacks):
    if type(lookbacks) == int:
        assert lookbacks <= 0
        return list(range(lookbacks, 1))
    return lookbacks