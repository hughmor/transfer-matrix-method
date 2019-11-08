import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tmm import tmm, np, pi
from materials import Material, INDICES
mpl.style.use('seaborn-whitegrid')

# don't change these
MODELS = {'Drude': 0, 'Johnson': 1, 'Palik': 2}
POLARIZATIONS = ['TE', 'TM']
MATERIALS_FILE = 'material_input.txt'

# change these to control the default sweep settings
WAVELENGTH_MIN = 400.0  # nanometres
WAVELENGTH_MAX = 1700.0
N_WAVELENGTHS = 1000
ANGLE_MIN = 0.0
ANGLE_MAX = 45 * pi / 180
N_ANGLES = 100

# change these to control the default plot settings
PLOT2D = True
ANGLE_NDX = 0  # angle slice for 2D plot (must be between 0 and N_ANGLES-1)
PLOT3D = False


def tmm_sweep(**kwargs):
    """
    Runs the program. Defines data structures, gets inputs, and calls transfer matrix method sweeping over wavelengths,
    angles, and polarizations.

    All inputs can be altered but gets materials from 'material_input.txt' by default.
    """

    # get list of Material objects in stack
    if 'materials' in kwargs.keys():
        materials = kwargs['materials']
    else:    # not provided, get from text file
        materials = get_materials()

    # define 2 sweep dimensions and mesh grid
    wavelengths = np.linspace(kwargs.get('min_wavelength', WAVELENGTH_MIN),
                              kwargs.get('max_wavelength', WAVELENGTH_MAX),
                              kwargs.get('num_wavelengths', N_WAVELENGTHS))
    angles = np.linspace(kwargs.get('min_angle', ANGLE_MIN),
                         kwargs.get('max_angle', ANGLE_MAX),
                         kwargs.get('num_angles', N_ANGLES))
    wavelengths, angles = np.meshgrid(wavelengths, angles)

    # initialize output arrays
    reflectance = {p: np.zeros(shape=wavelengths.shape) for p in POLARIZATIONS}
    transmittance = {p: np.zeros(shape=wavelengths.shape) for p in POLARIZATIONS}

    # run 2 sweep dimensions for each polarization and store accordingly
    for p in POLARIZATIONS:
        for i in range(len(wavelengths[:, 0])):
            for j in range(len(angles[0, :])):
                t, r = tmm(materials, wavelengths[i, j], angles[i, j], p)
                transmittance[p][i, j] = t
                reflectance[p][i, j] = r

    return wavelengths, angles, reflectance, transmittance


def read_from_input():
    """
    Reads input parameters from text file.

    :return: List of material specification strings, Supplied model to use for metals
    """
    txt_input = np.loadtxt(MATERIALS_FILE, dtype=str)
    metal_model = None
    if len(txt_input[0].split('=')) > 1:    # first line specifies model
        model_str = txt_input[0].split('=')[1].capitalize()
        if model_str not in MODELS.keys():
            raise ValueError('Invalid model supplied in input file.')
        metal_model = model_str
        material_list = txt_input[1:]
    else:    # no model given, should be no metals
        material_list = txt_input
    return material_list, metal_model


def get_materials():
    """
    Converts input parameters to a list of Material objects.

    :return: List of Material objects
    """
    material_list, metal_model = read_from_input()
    materials = []
    for line in material_list:
        l = line.split(',')
        if len(l) == 2:    # '[name],[thickness]'
            if l[0].capitalize() in INDICES.keys():    # hardcoded material
                materials.append(Material(l[0], thickness=float(l[1])))
            elif metal_model is not None:    # gold or silver
                materials.append(Material(l[0], model=metal_model, thickness=float(l[1])))
            else:
                raise ValueError('No metal model supplied.')
        elif len(l) == 3:    # '[name],[thickness],[index]'
            materials.append(Material(l[0], thickness=float(l[1]), index=complex(l[2])))
        else:
            raise ValueError('Malformed line in input file.')
    return materials


def show_plots(wavelengths, angles, reflectance, transmittance, plot_3d=PLOT3D, plot_2d=PLOT2D, angle_slice=ANGLE_NDX):
    """
    Generates output plots. Can plot results in 2D, 3D, or both.

    :param wavelengths: 2D array of wavelengths from meshgrid.
    :param angles: 2D array of angles from meshgrid.
    :param reflectance: 2D array of reflectance values for each wavelength and angle.
    :param transmittance: 2D array of transmittance values for each wavelength and angle.
    :param plot_3d: Whether to show the 3D plot; default is False.
    :param plot_2d: Whether to show the 2D plot; default is True.
    :param angle_slice: Index of the angle slice to plot; default is 0.
    """
    if plot_3d:
        '3D Plot'
        fig_3d = plt.figure('Reflected & Transmitted Power vs. Wavelength & Angle')
        axes_3d = []
        for i, comp in enumerate([transmittance, reflectance]):
            axes_3d.append(fig_3d.add_subplot(1, 2, i+1, projection='3d'))
            surf1 = axes_3d[i].plot_surface(wavelengths, angles*180/pi, comp['TE'], label='TE')
            surf1._facecolors2d = surf1._facecolors3d
            surf1._edgecolors2d = surf1._edgecolors3d
            surf2 = axes_3d[i].plot_surface(wavelengths, angles*180/pi, comp['TM'], label='TM')
            surf2._facecolors2d = surf2._facecolors3d
            surf2._edgecolors2d = surf2._edgecolors3d
            axes_3d[i].set_xlabel('Wavelength (nm)')
            axes_3d[i].set_ylabel('Angles (º)')
            axes_3d[i].set_zlabel('Magnitude')
            axes_3d[i].legend()
    if plot_2d:
        '2D Plots'
        fig_2d = plt.figure('Reflected & Transmitted Power vs. Wavelength @ {:.2f}º'.format(angles[angle_slice, 0]))
        axes_2d = []
        for i, comp in enumerate([transmittance, reflectance]):
            axes_2d.append(fig_2d.add_subplot(1, 2, i+1))
            axes_2d[i].plot(wavelengths[0, :], comp['TE'][angle_slice, :])
            axes_2d[i].plot(wavelengths[0, :], comp['TM'][angle_slice, :])
            axes_2d[i].set_xlabel('Wavelength (nm)')
            axes_2d[i].set_ylabel('Magnitude')
            axes_2d[i].legend(POLARIZATIONS)

    plt.show()


if __name__ == '__main__':
    results = tmm_sweep()
    if PLOT2D or PLOT3D:
        show_plots(*results, plot_2d=PLOT2D, plot_3d=PLOT3D, angle_slice=ANGLE_NDX)

