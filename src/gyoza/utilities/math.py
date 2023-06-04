import numpy as np
from typing import Tuple

def swirl(x:np.ndarray, y: np.ndarray, x0:float = 0, y0: float = 0, radius: float = 5, rotation: float = 0, strength: float = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a swirl operation on given x and y coordinates.
    
    Inputs:
    - x, y: Coordinates of points that shall be swirled.
    - x0, y0: The origin of the swirl.
    - radius: The extent of the swirl. Small values indicate local swirl, large values indicate global swirl.
    - rotation: Adds a rotation angle to the swirl.
    - strength: Indicates the strength of swirl.

    Outputs:
    - x_new, y_new: The transformed coordinates.
    """
    
    # Polar coordinates of each point
    theta = np.arctan2((y-y0), (x-x0))
    rho = np.sqrt((x-x0)**2 + (y-y0)**2)
    
    # Swirl
    r = np.log(2)*radius/5
    new_theta = rotation + strength * np.exp(-rho/r) + theta

    # Cartesian coordinates
    x_new = rho * np.cos(new_theta)
    y_new = rho * np.sin(new_theta)

    # Outputs
    return x_new, y_new