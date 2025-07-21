from typing import Tuple

import numpy as np
import pyproj


def compute_ellipse(area: float, ar: float, theta: float, loncent: float, latcent: float,
                    num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, pyproj.Proj]:
    r"""
    Computes the coordinates of a rotated ellipse based on geophysical parameters and
    projects it onto a polar stereographic coordinate system centered near the North Pole.

    :param area: Area of the ellipse in square kilometers.
    :type area: float
    :param ar: Aspect ratio of the ellipse (major axis / minor axis).
    :type ar: float
    :param theta: Orientation angle of the major axis in radiant, measured counter-clockwise from east.
    :type theta: float
    :param loncent: Longitude of the ellipse center in degrees.
    :type loncent: float
    :param latcent: Latitude of the ellipse center in degrees.
    :type latcent: float
    :param num_points: Number of points to use for generating the ellipse perimeter. Default is 200.
    :type num_points: int, optional

    :raises ValueError:
    :raises TypeError:

    :return: Triple containing the projected x and y coordinates (in meters) of the rotated ellipse and projection.
    :rtype: tuple of np.ndarray
    """
    if area <= 0:
        raise ValueError("Area must be a positive number.")
    if not isinstance(area, (int, float)):
        raise TypeError("Area must be a positive number.")
    if ar <= 0:
        raise ValueError("Aspect ratio must be a positive number.")
    if not isinstance(ar, (int, float)):
        raise TypeError("Aspect ratio must be a positive number.")

    # Calculate semi-major (a) and semi-minor (b) axes
    b = np.sqrt(area / (np.pi * ar))  # Minor axis length [km]
    a = ar * b  # Major axis length [km]

    # Create points for the ellipse in x-y coordinate system
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(t)  # [km]
    y = b * np.sin(t)  # [km]

    # Rotate the ellipse by theta degrees
    theta_rad = theta + np.radians(90)  # radiant, account for 0° in x-y not equal 0° North Polar Projection
    x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)  # [km]
    y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)  # [km]

    # Define stereographic projection centered on the ellipse
    proj_pyproj = pyproj.Proj(proj='stere',
                              lat_0=90,  # latcent
                              lon_0=0,  # loncent
                              lat_ts=60,  # latcent
                              ellps='WGS84')
    # Convert center to x-y
    x_center, y_center = proj_pyproj(loncent, latcent)

    # Translate ellipse to the center in meters
    x_final = x_center + x_rot * 1000  # if x_rot is in km
    y_final = y_center + y_rot * 1000
    return x_final, y_final, proj_pyproj