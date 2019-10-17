import numpy as np
import matplotlib.pyplot as plt
from materials import Material
import math

"""
    Goal of this code is, given a list of materials and their thicknesses, as well as an incident electric field and its
    angle, build the transfer matrix that governs the transmitted and reflected fields. Output will be a plot of the
    reflection and transmission as a function of wavelength.
"""


def propagation_matrix(k, d):
    return np.array([[np.exp(-1j * k * d), 0],
                    [0, np.exp(1j * k * d)]])  # check for accuracy


def interface_matrix(eps, mu, theta):
    return {'TE': np.array([[1, 1],
                            [np.sqrt(eps/mu)*np.cos(theta), -np.sqrt(eps/mu)*np.cos(theta)]]),
            'TM': np.array([[np.cos(theta), np.cos(theta)],
                            [np.sqrt(eps/mu), -np.sqrt(eps/mu)]])}


def get_refraction_angle(n1, n2, theta1):
    return np.arcsin(n1*np.sin(theta1)/n2)


def get_wavenumber(n, theta, wavelength):
    return 2 * math.pi * n * np.cos(theta) / wavelength


print('ELEC 856 Assignment 1')
print('Reflectance and Transmittance Spectra for Multi-material slabs')


show_plot = True
models = ["Drude", "Johnson", "Palik"]
metal_model = models[int(input('Metal Model 1 (Drude-Lorenz), 2 (Johnson-Christy), or 3 (Palik)?'))-1]
custom_params = False

polarization = ['TE', 'TM']
incident_angle = 0
number_of_slabs = 5
wavelength_min = 400e-9
wavelength_max = 1700e-9

if custom_params:
    incident_angle = float(input('Incident Angle (in º): '))
    number_of_slabs = int(input('Number of materials: '))

material_strings = input('Enter materials:').split(',')
thicknesses = []
for mat_str in material_strings:
    thicknesses.append(float(input('{} thickness (in nm): '.format(mat_str.capitalize()))) * 10**-9)

materials = [
    Material(material_strings[i],  thickness=thicknesses[i]) for i in range(len(material_strings))] #model=metal_model,
wavelengths = np.linspace(wavelength_min, wavelength_max, 500)
reflectance = {polarization[0]: [], polarization[1]: []}
transmittance = {polarization[0]: [], polarization[1]: []}

for p in polarization:
    for wvl in wavelengths:
        angle = incident_angle
        prev_mat = Material('air')
        D0 = interface_matrix(prev_mat.get_eps(wvl), prev_mat.get_mu(wvl), angle)[p]
        M = np.linalg.inv(D0)
        for mat in materials:
            angle = get_refraction_angle(prev_mat.get_n(wvl), mat.get_n(wvl), angle)
            D = interface_matrix(mat.get_eps(wvl), mat.get_mu(wvl), angle)[p]
            P = propagation_matrix(get_wavenumber(mat.get_n(wvl), angle, wvl), mat.d)
            M = M.dot(D).dot(P).dot(np.linalg.inv(D))
            prev_mat = mat
        M = M.dot(D0)
        t = np.abs((1/M[0, 0]))**2
        r = np.abs((M[1, 0] / M[0, 0]))**2
        transmittance[p].append(t)
        reflectance[p].append(r)


if show_plot:
    ''
    plt.figure('TE')
    plt.title('R/T for continuous TE wave')
    plt.plot(wavelengths*1e9, transmittance['TE'], '-C1')
    plt.plot(wavelengths*1e9, reflectance['TE'], '--C1')
    plt.plot(wavelengths*1e9, [reflectance['TE'][i]+transmittance['TE'][i] for i in range(len(transmittance['TE']))])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Magnitude')
    plt.legend(['T', 'R'])

    ''
    plt.figure('TM')
    plt.title('R/T for continuous TM wave')
    plt.plot(wavelengths*1e9, transmittance['TM'], '-C2')
    plt.plot(wavelengths*1e9, reflectance['TM'], '--C2')
    plt.plot(wavelengths*1e9, [reflectance['TM'][i]+transmittance['TM'][i] for i in range(len(transmittance['TM']))])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Magnitude')
    plt.legend(['T', 'R'])

    ''
    plt.show()



