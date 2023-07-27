import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import OrderedDict

def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def polar_to_cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def archimedian_spiral(xs, alpha):
    phi = xs

    # Transform
    rho = alpha * phi 

    # Convert to cartesian
    xs, ys = polar_to_cartesian(rho=rho, phi=phi)

    return xs, ys

def logarithmic_spiral(xs, alpha, beta):
    phi = xs

    # Transform
    rho = alpha * np.exp(beta*phi) 

    # Convert to cartesian
    xs, ys = polar_to_cartesian(rho=rho, phi=phi)

    return xs, ys

def rotate(xs, ys, theta):
    # Convert to polar
    rho, phi = cartesian_to_polar(x=xs, y=ys)

    # Rotate
    phi = phi + theta

    # Convert to cartesian
    xs, ys = polar_to_cartesian(rho=rho, phi=phi)

    return xs, ys

def tangent(f, f_prime, x_0):
    return lambda x: x*f_prime(x_0) + f(x_0) - x_0*f_prime(x_0)

def normal(f, f_prime, x_0):
    return lambda x: (-1/f_prime(x_0))*(x-x_0) + f(x_0) 

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

def __make_color_palette__() -> np.ndarray:
    """Generates interpolations between the colors red, green and blue.
    
    :return: color_palette (:class:`np.ndarray`) - An array of shape [55, 3], listing colors in RGB format."""

    # from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    color_palette = np.zeros([ncols, 3])
    col = 0
    # RY
    color_palette[0:RY, 0] = 255
    color_palette[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    color_palette[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    color_palette[col:col+YG, 1] = 255
    col += YG
    # GC
    color_palette[col:col+GC, 1] = 255
    color_palette[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    color_palette[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    color_palette[col:col+CB, 2] = 255
    col += CB
    # BM
    color_palette[col:col+BM, 2] = 255
    color_palette[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    color_palette[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    color_palette[col:col+MR, 0] = 255
    
    # Output
    return color_palette 

color_palette = __make_color_palette__()
"""A convenience variable, storing the color palette that is computed by :py:meth:`__make_color_palette__`."""

def make_radial_line(radius: float, rotation: float, point_count: int) -> np.ndarray:
    """Generates a straight line with ``point_count`` many points that has one endpoint at the origin and the other endpoint on the 
    circle defined by defined by ``radius`` and ``rotation``.

    :param radius: The radius of the circle from which lines are generated.
    :type radius: float
    :param rotation: The angle of rotation of the line in radians. Movement is clockwise.
    :type rotation: float
    :param point_count: The number of points on the line.
    :type point_count: int
    :return: x, y (:class:`np.ndarray`) - The coordinates of line with shape [``point_count``, 2].
    """

    # Generate horizontal line
    x = np.arange(start=0, stop=radius+radius/point_count, step=radius/point_count, dtype=np.float32)
    y = np.zeros(x.shape, dtype=np.float32)
    line = np.concatenate([x[:,np.newaxis], y[:,np.newaxis]], axis=1); del x, y

    # Rotate
    rotaton_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    line = np.dot(line, rotaton_matrix)

    # Unpack
    x = line[:,0]; y = line[:,1]

    # Outputs
    return x, y

def make_2_dimensional_gaussian(mu: np.ndarray, sigma: np.ndarray, shape: List[int]) -> np.ndarray:
    """Generates a 2 dimensional Gaussian distribution.

    :param mu: The two means for the Gaussian variables. Assumed to be of shape [2].
    :type mu: np.ndarray
    :param sigma: The covariance matrix. Assumed to be of shape [2,2].
    :type sigma: np.ndarray
    :param shape: The desired shape of the output.
    :type shape: _type_, optional

    :return: 
        - X (:class:`numpy.ndarray`) - Coordiates of the grid of the two variables. Shape == [ ``shape`` [0]* ``shape`` [1],2].
        - p (:class:`numpy.ndarray`) - The probabilities associated with the coordinates ``X``. Shape == [ ``shape`` [0]* ``shape`` [1]].
        - D (:class:`numpy.ndarray`) - A matrix that arranges ``p`` with desired ``shape``.
        """


    # Generate x, y coordinates 
    x = np.linspace(-3, 3, shape[1])
    y = np.linspace(-3, 3, shape[0])
    xv, yv = np.meshgrid(x, y)
    X = np.concatenate([np.reshape(xv,[-1,1]), np.reshape(yv, [-1,1])], axis=1) # Shape == [shape[0]*shape[1],2] 
    
    # Compute probability
    numerator = np.exp(-0.5*np.sum((X-mu).dot(np.linalg.inv(sigma)) * (X-mu), axis=1))
    denominator = np.sqrt((2*np.pi)**2 * np.linalg.det(sigma))
    p = numerator / denominator

    D = np.reshape(p, [shape[1], shape[0]])

    # Outputs
    return X, p, D

def make_color_wheel(pixels_per_inch: int, pixel_count: int = 128, swirl_strength: float = 0, gaussian_variance: float = 1) -> np.ndarray:
    """Generates an image of a color wheel with swirl

    :param dpi: The density of pixels per inch on the user machine.
    :type dpi: int
    :param pixel_count: The desired width and height of ``image`` in pixels, defaults to 128
    :type pixel_count: int, optional
    :param swirl_strength: The strength of swirl applied to the color wheel. Sensible values are in the range [0,10]. The sign is 
        ignored. Defaults to 0
    :type swirl_strength: float, optional
    :param saturation: The saturation of the colors. valid values are in range [0,1], where 0 corresponds to a white image and 1 to
        a fully satured image. Defaults to 1.
    :type saturation: float, optional
    :return: image (:class:`np.ndarray`) - The image of shape [pixel_count, pixel_count, 4] where 4 are the channels.
    """

    # Make radial lines
    x_s = [None] * len(color_palette); y_s = [None] * len(color_palette)
    for c in range(len(color_palette)):
        # Make straight line
        x, y = make_radial_line(radius=1, rotation=c*2*np.pi/(len(color_palette)), point_count=2+(int)(2*swirl_strength))
        
        # Add swirl
        if swirl_strength != 0: x,y = swirl(x=x,y=y,radius=5, rotation=0, strength=swirl_strength)
            
        # Save to array
        x_s[c] = x; y_s[c] = y

    # Draw wedges
    figure = plt.figure(figsize=(pixel_count/(2*pixels_per_inch), pixel_count/(2*pixels_per_inch)), dpi=pixels_per_inch)
    plt.axis('off')
    for c, color in enumerate(color_palette):
        # A wedges has two main lines
        x_c = x_s[c]; y_c = y_s[c]
        x_d = x_s[(c+1) % len(color_palette)]; y_d = y_s[(c+1) % len(color_palette)]
        
        # Draw on figure
        plt.fill(np.concatenate([x_c,np.flip(x_d)]), np.concatenate([y_c,np.flip(y_d)]), color=tuple(color/255), linewidth=1/pixels_per_inch)

    plt.xlim([-1,1]); plt.ylim([-1,1])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    # Export as image
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    plt.close()

    # Apply gaussian saturation
    _, _, gaussian = make_2_dimensional_gaussian(mu=np.zeros([2]), sigma=gaussian_variance*np.eye(2), shape=[height, width])
    gaussian = (gaussian-np.min(gaussian))/np.max(gaussian) # Now ranges between 0 and 1
    gaussian = np.array(255 *gaussian, dtype=np.uint8)
    image[:,:,3] = gaussian # The alpha channel 
    
    # Outputs
    return image

note_to_frequency = OrderedDict([
('C0', 16.35),  ('C#0', 17.32),  ('D0', 18.35),  ('D#0',19.45),   ('E0', 20.60),  ('F0', 21.83),  ('F#0', 23.12),  ('G0', 24.50),  ('G#0', 25.96),  ('A0', 27.50),  ('A#0',29.14),   ('B0', 30.87),
('C1', 32.70),  ('C#1',34.65),   ('D1', 36.71),  ('D#1', 38.89),  ('E1', 41.20),  ('F1', 43.65),  ('F#1', 46.25),  ('G1', 49.00),  ('G#1', 51.91),  ('A1', 55.00),  ('A#1', 58.27),  ('B1', 61.74),
('C2', 65.41),  ('C#2', 69.30),  ('D2', 73.42),  ('D#2', 77.78),  ('E2', 82.41),  ('F2', 87.31),  ('F#2', 92.50),  ('G2', 98.00),  ('G#2', 103.83), ('A2', 110.00), ('A#2', 116.54), ('B2', 123.47),
('C3', 130.81), ('C#3', 138.59), ('D3', 146.83), ('D#3', 155.56), ('E3', 164.81), ('F3', 174.61), ('F#3', 185.00), ('G3', 196.00), ('G#3', 207.65), ('A3', 220.00), ('A#3', 233.08), ('B3', 246.94),
('C4',261.63),  ('C#4',277.18),  ('D4',293.66),  ('D#4', 311.13), ('E4', 329.63), ('F4', 349.23), ('F#4',369.99),  ('G4',392.00),  ('G#4',415.30),  ('A4', 440.00), ('A#4',466.16),  ('B4', 493.88),
('C5',523.25),  ('C#5',554.37),  ('D5', 587.33), ('D#5',622.25),  ('E5', 659.25), ('F5', 698.46), ('F#5',739.99),  ('G5', 783.99), ('G#5', 830.61), ('A5', 880.00), ('A#5', 932.33), ('B5', 987.77),
('C6', 1046.50),('C#6', 1108.73),('D6', 1174.66),('D#6', 1244.51),('E6', 1318.51),('F6', 1396.91),('F#6', 1479.98),('G6', 1567.98),('G#6', 1661.22),('A6', 1760.00),('A#6', 1864.66),('B6', 1975.53),
('C7', 2093.00),('C#7', 2217.46),('D7', 2349.32),('D#7', 2489.02),('E7', 2637.02),('F7', 2793.83),('F#7', 2959.96),('G7', 3135.96),('G#7', 3322.44),('A7', 3520.00),('A#7', 3729.31),('B7', 3951.07),
('C8', 4186.01),('C#8', 4434.92),('D8', 4698.63),('D#8', 4978.03),('E8', 5274.04),('F8', 5587.65),('F#8', 5919.91),('G8', 6271.93),('G#8', 6644.88),('A8', 7040.00),('A#8', 7458.62),('B8', 7902.13)
])

major_steps = [0,2,4,5,7,9,11,12]
"""The steps one needs to take from a given base note to reach all notes on its diatonic major scale."""
minor_steps = [0,2,3,5,7,8,10,12]
"""The steps one needs to take from a given base note to reach all notes on its diatonic minor scale."""

# Idea: train: play minor scale in one half of the spectrum, play major scale in other half.
# test: swop them around to see whether model learned major/ minor and spectral range
# One factor is octave represented as expontentially increasing freqs
# Other factor is scale which is a categroical variable. For each scale we show its half of the octaves
# in all combinations, hence independent variables.