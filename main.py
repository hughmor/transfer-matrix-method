import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tmm import np, pi, tmm
from materials import Material, INDICES

"""
    ELEC 856 Nanophotonics Assignment 1
    Hugh Morison
    
    This script builds the transfer matrix for the light incident on a multi-material slab and plots the reflection and 
    transmission as a function of wavelength and incident angle for both TE and TM polarizations. 
    
    Materials are supplied through the file 'material_input.txt' where each line is of the form
    '[name],[thickness],[index]' where name and thickness are required, and index is only required for materials not
    listed in the file 'materials.py'. You can always supply a value for index to use a constant value for any
    material.
    
    If you choose either 'gold' or 'silver', the first line of 'material_input.txt' must be of the form 'model=[model]'
    where model is the model for the optical properties of the material and must be one of 'Drude', 'Johnson', or
    'Palik'.
    
    The 'Drude' model is a semi-analytical model of the permittivities, while the 'Johnson' and 'Palik' models use 
    complex refractive index values interpolated from the books by Johnson & Christy and Palik.
    
"""

# change these to control the settings for the wavelength and angle sweeps
WAVELENGTH_MIN = 400e-9
WAVELENGTH_MAX = 1700e-9
N_WAVELENGTHS = 500
ANGLE_MIN = 0
ANGLE_MAX = 85*pi/180
N_ANGLES = 10

# change these to control the plot settings
PLOT2D = True
ANGLE_NDX = N_ANGLES-1    # angle slice for 2D plot (must be between 0 and N_ANGLES-1)
PLOT3D = True

# don't change these
MODELS = {'Drude': 0, 'Johnson': 1, 'Palik': 2}
POLARIZATIONS = ['TE', 'TM']
MATERIALS_FILE = 'material_input.txt'
NANOMETRES = 1.0e-9


def readfrominput():
    """
    Reads input parameters from text file.

    :return: list of material specification strings, the supplied model to use for metals
    """
    txt_input = np.loadtxt(MATERIALS_FILE, dtype=str)
    metal_model = None
    if len(txt_input[0].split('=')) > 1:    # first line specifies model
        model_str = txt_input[0].split('=')[1].capitalize()
        if model_str not in MODELS.keys():
            raise ValueError('Invalid model supplied in input file.')
        metal_model = MODELS.get(model_str)
        material_list = txt_input[1:]
    else:    # no model given, should be no metals
        material_list = txt_input
    return material_list, metal_model


def getmaterials():
    """
    Converts input parameters to a list of Material objects.

    :return: list of Material objects
    """
    material_list, metal_model = readfrominput()
    materials = []
    for line in material_list:
        l = line.split(',')
        if len(l) == 2:
            if l[0].capitalize() in INDICES.keys():    # hardcoded material
                materials.append(Material(l[0], thickness=float(l[1])*NANOMETRES))
            elif metal_model is not None:    # gold or silver using model
                materials.append(Material(l[0], metal_model=metal_model, thickness=float(l[1])*NANOMETRES))
            else:
                raise ValueError('No metal model supplied.')
        elif len(l) == 3:    # custom material
            materials.append(Material(l[0], thickness=float(l[1])*NANOMETRES, index=float(l[2])))
        else:
            raise ValueError('Malformed line in input file')
    return materials


def showplots(wavelengths, angles, reflectance, transmittance, plot_3d=True, plot_2d=True, angle_slice=0):
    """
    Generates output plots. Can plot results in 2D, 3D, or both.

    :param wavelengths: 2D array of wavelengths from meshgrid
    :param angles: 2D array of angles from meshgrid
    :param reflectance: 2D array of reflectance values for each wavelength and angle
    :param transmittance: 2D array of transmittance values for each wavelength and angle
    :param plot_3d: whether to show the 3D plot; default is True
    :param plot_2d: whether to show the 2D plot; default is True
    :param angle_slice: number of the angle slice to plot; default is 0
    """
    if plot_3d:
        '3D Plot'
        fig_3d = plt.figure('Reflected & Transmitted Power vs. Wavelength & Angle')
        axes_3d = []
        for i, p in enumerate(POLARIZATIONS):
            axes_3d.append(fig_3d.add_subplot(1, 2, i+1, label=p, projection='3d'))
            surf1 = axes_3d[i].plot_surface(wavelengths*NANOMETRES, angles*180/pi, transmittance[p], label='T')
            surf1._facecolors2d = surf1._facecolors3d
            surf1._edgecolors2d = surf1._edgecolors3d
            surf2 = axes_3d[i].plot_surface(wavelengths*NANOMETRES, angles*180/pi, reflectance[p], label='R')
            surf2._facecolors2d = surf2._facecolors3d
            surf2._edgecolors2d = surf2._edgecolors3d
            axes_3d[i].set_xlabel('Wavelength (nm)')
            axes_3d[i].set_ylabel('Angles (º)')
            axes_3d[i].set_zlabel('Magnitude')
            axes_3d[i].legend()
    if plot_2d:
        '2D Plots'
        fig_2d = plt.figure('Reflected & Transmitted Power vs. Wavelength @ {}º'.format(angles[angle_slice, 0]))
        axes_2d = []
        for i, p in enumerate(POLARIZATIONS):
            axes_2d.append(fig_2d.add_subplot(1, 2, i+1, label=p))
            axes_2d[i].plot(wavelengths[0, :]/NANOMETRES, transmittance[p][angle_slice, :], '-C4')
            axes_2d[i].plot(wavelengths[0, :]/NANOMETRES, reflectance[p][angle_slice, :], '-C2')
            axes_2d[i].set_xlabel('Wavelength (nm)')
            axes_2d[i].set_ylabel('Magnitude')
            axes_2d[i].legend(['T', 'R', 'T+R'])

    plt.show()


def main():
    """
    Runs the program. Defines data structures, gets inputs, and calls transfer matrix code.

    :return: the values needed for plotting: wavelength meshgrid, angle meshgrid, reflectance, and transmittance
    """
    materials = getmaterials()

    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N_WAVELENGTHS)
    angles = np.linspace(ANGLE_MIN, ANGLE_MAX, N_ANGLES)
    wavelengths, angles = np.meshgrid(wavelengths, angles)

    reflectance = {p: np.zeros(shape=(N_ANGLES, N_WAVELENGTHS)) for p in POLARIZATIONS}
    transmittance = {p: np.zeros(shape=(N_ANGLES, N_WAVELENGTHS)) for p in POLARIZATIONS}

    for p in POLARIZATIONS:
        for i in range(N_ANGLES):
            for j in range(N_WAVELENGTHS):
                t, r = tmm(materials, wavelengths[i, j], angles[i, j], p)
                transmittance[p][i, j] = t
                reflectance[p][i, j] = r

    return wavelengths, angles, reflectance, transmittance


if __name__ == '__main__':
    results = main()
    if PLOT2D or PLOT3D:
        showplots(*results, plot_2d=PLOT2D, plot_3d=PLOT3D, angle_slice=ANGLE_NDX)

