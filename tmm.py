from cmath import sin, asin
import numpy as np
from materials import Material


def tmm(materials, wavelength, angle, polarization):
    """
    Calculates fraction of incident power transmitted/reflected for a multi-layer slab bounded on either side by air.
    Assumes slabs are infinite in the x and y directions and bounding layers are semi-infinite in z.
    Calculations are based on the transfer matrix approach; this program builds the matrix M based on the dynamical
    boundary matrices D and propagation matrices P for each slab.
    The output values are pulled from the matrix M.

    :param materials: List of Material objects defining the multi-layer slab.
    :param wavelength: Vacuum wavelength of the wave in nanometres.
    :param angle: Incident angle of the wave.
    :param polarization: Polarization of incoming wave. Can be either 'TE' or 'TM'.
    :return: Fraction of incident power transmitted, fraction of incident power reflected.
    """
    outer_material = Material('Air')    # semi-infinite slabs of air outside the structure
    prev_material = outer_material
    D_0 = prev_material.dynamical_matrix(wavelength, angle, polarization)
    M = np.linalg.inv(D_0)
    for material in materials:
        angle = get_refraction_angle(prev_material, material, wavelength, angle)
        D_i = material.dynamical_matrix(wavelength, angle, polarization)
        P_i = material.propagation_matrix(wavelength, angle)
        M = M.dot(D_i).dot(P_i).dot(np.linalg.inv(D_i))
        prev_material = material
    angle = get_refraction_angle(prev_material, outer_material, wavelength, angle)
    D_out = outer_material.dynamical_matrix(wavelength, angle, polarization)
    M = M.dot(D_out)

    T = np.abs((1 / M[0, 0])) ** 2
    R = np.abs((M[1, 0] / M[0, 0])) ** 2
    return T, R


def get_refraction_angle(material_1, material_2, wavelength, theta_1):
    """
    Uses Snell's Law to get the angle of the propagating wave in material_2 after refracting from material_1 with angle
    theta_1.

    :param material_1: Material light is coming from.
    :param material_2: Material light is refracting into.
    :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
    :param theta_1: Complex angle of the propagating wave in material_1.
    :return: Complex angle of the propagating wave in material_2.
    """
    n1 = material_1.get_n(wavelength)
    n2 = material_2.get_n(wavelength)
    theta_2 = asin(n1*sin(theta_1)/n2)
    return theta_2
