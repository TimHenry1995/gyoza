import numpy as np

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
