import numpy as np
np.seterr(all='raise')
import matplotlib.pyplot as plt
from materials import Material
import math

"""
    Goal of this code is, given a list of materials and their thicknesses, as well as an incident electric field and its
    angle, build the transfer matrix that governs the transmitted and reflected fields. Output will be a plot of the
    reflection and transmission as a function of wavelength and incident angle.
"""

print('ELEC 856 Assignment 1')
print('Reflectance and Transmittance Spectra for Multi-material slabs')


def get_refraction_angle(n1, n2, theta1):
    return np.arcsin(n1 * np.sin(theta1) / n2).real


def dynamical_matrix(eps, mu, theta):
    return {'TE': np.array([[1, 1],
                            [np.sqrt(eps/mu)*np.cos(theta), -np.sqrt(eps/mu)*np.cos(theta)]]),
            'TM': np.array([[np.cos(theta), np.cos(theta)],
                            [np.sqrt(eps/mu), -np.sqrt(eps/mu)]])}


def propagation_matrix(k, d):
    return np.array([[np.exp(1j * k * d), 0],
                     [0, np.exp(-1j * k * d)]])


def get_wavenumber(n, theta, wavelength):
    return 2 * math.pi * n * np.cos(theta) / wavelength


def tmm(wavelength, angle, mats, polarization):
    outer_material = Material('air')
    prev_mat = outer_material
    D_0 = dynamical_matrix(prev_mat.get_eps(wavelength), prev_mat.get_mu(wavelength), angle)[polarization]
    M = np.linalg.inv(D_0)
    for mat in mats:
        angle = get_refraction_angle(prev_mat.get_n(wavelength), mat.get_n(wavelength), angle)
        D_i = dynamical_matrix(mat.get_eps(wavelength), mat.get_mu(wavelength), angle)[polarization]
        P_i = propagation_matrix(get_wavenumber(mat.get_n(wavelength), angle, wavelength), mat.d)
        M = M.dot(D_i).dot(P_i).dot(np.linalg.inv(D_i))
        prev_mat = mat
    angle = get_refraction_angle(prev_mat.get_n(wavelength), outer_material.get_n(wavelength), angle)
    D_out = dynamical_matrix(outer_material.get_eps(wavelength), outer_material.get_mu(wavelength), angle)[polarization]
    M = M.dot(D_out)
    T = np.abs((1 / M[0, 0])) ** 2
    R = np.abs((M[1, 0] / M[0, 0])) ** 2
    return T, R


MODELS = ['Drude', 'Johnson', 'Palik']
POLARIZATIONS = ['TE', 'TM']
WAVELENGTH_MIN = 400e-9
WAVELENGTH_MAX = 1700e-9
ANGLE_MIN = 0
ANGLE_MAX = np.pi/2
N_POINTS = 100

wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N_POINTS)
angles = np.linspace(ANGLE_MIN, ANGLE_MAX, N_POINTS)

USER_PARAMETERS = False
CUSTOM_MATERIALS = USER_PARAMETERS
metal_model = MODELS[2]
number_of_slabs = 5
show_plot = True
if USER_PARAMETERS:
    metal_model = MODELS[int(input('Which model for metals:'
                                   '\n\t[1] Analytic Drude-Lorenz'
                                   '\n\t[2] Numerical (Johnson-Christy)'
                                   '\n\t[3] Numerical (Palik)'
                                   '\n')) - 1]
    number_of_slabs = int(input('Number of materials: '))
    show_plot = input('Show Plots? (T/F) ') is 'T'

if CUSTOM_MATERIALS:
    material_strings = input('Enter materials:').split(',')
    thicknesses = []
    for mat_str in material_strings:
        thicknesses.append(float(input('{} thickness (in nm): '.format(mat_str.capitalize()))) * 10**-9)
    materials = [
        Material(material_strings[i], model=metal_model, thickness=thicknesses[i]) for i in range(len(material_strings))]
else:
    MATERIALS_FN = 'test_materials.txt'
    txt_file = np.loadtxt(MATERIALS_FN, dtype=str, delimiter=',')
    materials = [
        Material(line[0], model=metal_model, thickness=float(line[1]) * 10**-9) for line in txt_file]

reflectance = {POLARIZATIONS[0]: np.zeros(shape=(N_POINTS, N_POINTS)),
               POLARIZATIONS[1]: np.zeros(shape=(N_POINTS, N_POINTS))}
transmittance = {POLARIZATIONS[0]: np.zeros(shape=(N_POINTS, N_POINTS)),
                 POLARIZATIONS[1]: np.zeros(shape=(N_POINTS, N_POINTS))}

Wavelengths, Angles = np.meshgrid(wavelengths, angles)

for p in POLARIZATIONS:
    for i in range(N_POINTS):
        for j in range(N_POINTS):
            t, r = tmm(Wavelengths[i, j], Angles[i, j], materials, p)
            transmittance[p][i, j] = t
            reflectance[p][i, j] = r


if show_plot:
    '3D Plot'
    fig = plt.figure('Surface')
    ax1 = plt.axes(projection='3d')
    #ax1 = fig.add_subplot(211, label='TE', projection='3d')
    #ax2 = fig.add_subplot(212, label='TM', projection='3d')
    ax1.plot_wireframe(Wavelengths, Angles, transmittance['TE'])
    ax1.plot_wireframe(Wavelengths, Angles, reflectance['TE'])
    #ax2.plot_wireframe(Wavelengths, Angles, transmittance['TM'])

    # plt.figure('TE')
    # plt.title('R/T for continuous TE wave')
    # plt.plot(wavelengths*1e9, transmittance['TE'], '-C1')
    # plt.plot(wavelengths*1e9, reflectance['TE'], '--C1')
    # plt.plot(wavelengths*1e9, [reflectance['TE'][i]+transmittance['TE'][i] for i in range(len(transmittance['TE']))])
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Magnitude')
    # plt.legend(['T', 'R'])
    #
    # plt.figure('TM')
    # plt.title('R/T for continuous TM wave')
    # plt.plot(wavelengths*1e9, transmittance['TM'], '-C2')
    # plt.plot(wavelengths*1e9, reflectance['TM'], '--C2')
    # plt.plot(wavelengths*1e9, [reflectance['TM'][i]+transmittance['TM'][i] for i in range(len(transmittance['TM']))])
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Magnitude')
    # plt.legend(['T', 'R'])

    plt.show()



