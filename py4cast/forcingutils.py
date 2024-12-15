import datetime as dt
from typing import List

import numpy as np
import torch

"""
This module provides a colllection of utility functions to compute forcing
Forcing functions:
- get_year_hour_forcing()
- generate_toa_radiation_forcing()
Useful functions:
 - compute_day_of_years()
 - compute_hours_of_day()
 - compute_seconds_from_start_of_year()
"""


def compute_day_of_years(
    date: dt.datetime, output_terms: List[dt.timedelta]
) -> np.array:
    """
    Compute the day of the year for the specified terms from the date.
    First january is 1, not 0.
    """
    days = []

    for term in output_terms:
        date_tmp = date + term
        starting_year = dt.datetime(date_tmp.year, 1, 1)
        days.append((date_tmp - starting_year).days + 1)

    return np.asarray(days)


def compute_hours_of_day(
    date: dt.datetime, output_terms: List[dt.timedelta]
) -> np.array:
    """
    Compute the hour of the day for the specified terms from the date.
    """
    hours = []
    for term in output_terms:
        date_tmp = date + term
        hours.append(date_tmp.hour + date_tmp.minute / 60)
    return np.asarray(hours)


def compute_seconds_from_start_of_year(
    date: dt.datetime, output_terms: List[dt.timedelta]
) -> np.array:
    """
    Compute how many seconds have elapsed since the beginning of the year for the specified terms.
    """
    start_of_year = dt.datetime(date.year, 1, 1)
    return np.asarray(
        [(date + term - start_of_year).total_seconds() for term in output_terms]
    )


def get_year_hour_forcing(
    date: dt.datetime, output_terms: List[dt.timedelta]
) -> torch.Tensor:
    """
    Get the forcing term dependent of the date for each terms.
    """
    hours_of_day = compute_hours_of_day(date, output_terms)
    seconds_from_start_of_year = compute_seconds_from_start_of_year(date, output_terms)

    days_in_year = 366 if date.year % 4 == 0 else 365
    seconds_in_year = days_in_year * 24 * 60 * 60

    hour_angle = (torch.Tensor(hours_of_day) / 12) * torch.pi  # (sample_len,)
    year_angle = (
        (torch.Tensor(seconds_from_start_of_year) / seconds_in_year) * 2 * torch.pi
    )  # (sample_len,)
    datetime_forcing = torch.stack(
        (
            torch.sin(hour_angle),
            torch.cos(hour_angle),
            torch.sin(year_angle),
            torch.cos(year_angle),
        ),
        dim=1,
    )  # (N_t, 4)
    datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
    return datetime_forcing


def generate_toa_radiation_forcing(
    lat: torch.Tensor,
    lon: torch.Tensor,
    date_utc: dt.datetime,
    output_terms: List[dt.timedelta],
) -> torch.Tensor:
    """
    Get the forcing term of the solar irradiation for each terms.
    """

    day_of_years = compute_day_of_years(date_utc, output_terms)
    hours_of_day = compute_hours_of_day(date_utc, output_terms)

    # Hour angle, convert UTC hours into solar hours
    hours_lcl = torch.Tensor(hours_of_day).unsqueeze(-1).unsqueeze(-1) + lon / 15
    omega = 15 * (hours_lcl - 12)
    omega_rad = np.radians(omega)

    # Eq. 1.6.3 in Solar Engineering of Thermal Processes, Photovoltaics and Wind 5th ed.
    # Solar constant
    E0 = 1366

    # Eq. 1.6.1a in Solar Engineering of Thermal Processes, Photovoltaics and Wind 5th ed.
    # unit(23.45) = degree
    dec = 23.45 * torch.sin(
        2 * np.pi * (284 + torch.Tensor(day_of_years)) / 365
    ).unsqueeze(-1).unsqueeze(-1)

    dec_rad = np.radians(dec)

    # Latitude
    phi = torch.Tensor(lat)
    phi_rad = np.radians(phi)

    # Eq. 1.6.2 with beta=0 in Solar Engineering of Thermal Processes, Photovoltaics and Wind 5th ed.
    cos_sza = torch.sin(phi_rad) * torch.sin(dec_rad) + torch.cos(phi_rad) * torch.cos(
        dec_rad
    ) * torch.cos(omega_rad)

    # 0 if the sun is after the sunset.
    toa_radiation = torch.fmax(torch.tensor(0), E0 * cos_sza).unsqueeze(-1)

    return toa_radiation
