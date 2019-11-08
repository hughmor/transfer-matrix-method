import pandas as pd
import numpy as np
from scipy.constants import h, hbar, c, e, epsilon_0, mu_0, pi, electron_mass
from cmath import cos, sqrt, exp, inf

PATH = 'data/'
FILENAMES = {
    'Johnson': PATH + 'materialconstants_j.xlsx',
    'Palik': PATH + 'materialconstants_p.xlsx'}

INDICES = {    # hardcoded constant indices for common materials
    "Silicon": 3.48,
    "Silica": 1.48,
    "Air": 1.0
}

ELECTRON_DENSITIES = {    # values for drude model
    "Gold": 5.9e28,
    "Silver": 5.86e28
}

CONDUCTIVITIES = {
    "Gold": 4.1e7,
    "Silver": 6.3e7
}


class Material:
    """
    The Material object contains the optical properties of one material in a multi-layer structure and can generate its
    propagation matrix and dynamical matrix according to the transfer matrix method.

    :param name: The name of the material.
    :param thickness: The thickness of the slab in nanometres.
    :param model: The model to use for metal properties. Defaults to 'Constant'.
    :param index: Constant refractive index.
    """

    def __init__(self, name, thickness=inf, model='Constant', index=None):
        self.name = name.capitalize()
        self.model = model.capitalize()
        self.d = thickness
        self.n = index
        if self.n is None:    # no constant index provided
            if self.model == 'Constant':    # if the index is constant and wasn't provided, should be hardcoded
                if self.name not in INDICES.keys():
                    raise ValueError('Constant model but no index was provided and not hardcoded.')
                self.n = INDICES[self.name]
            elif self.model == 'Palik' or self.model == 'Johnson':    # these models use data from excel files
                self.n_table = self.import_properties()
            else:    # model must be Drude-Lorenz, don't need to assign anything
                if not self.model == 'Drude':
                    raise ValueError('Invalid Model Supplied.')

    def dynamical_matrix(self, wavelength, theta, p):
        """
        Generates the dynamical boundary matrix for the Material at a specific wavelength, angle, and polarization.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param theta: Complex angle (measured from the normal) of the propagating wave.
        :param p: Polarization of the propagating wave. Can be 'TE' or 'TM'.
        :return: 2x2 dynamical matrix at the boundary of the Material.
        """
        eps = self.get_eps(wavelength)
        mu = self.get_mu(wavelength)
        if p == 'TE':
            return np.array([[1, 1],
                             [sqrt(eps / mu) * cos(theta), -sqrt(eps / mu) * cos(theta)]])
        elif p == 'TM':
            return np.array([[cos(theta), cos(theta)],
                             [sqrt(eps / mu), -sqrt(eps / mu)]])
        else:
            raise ValueError('Invalid Polarization')

    def propagation_matrix(self, wavelength, theta):
        """
        Generates the propagation matrix through the Material at a specific wavelength and angle.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param theta: Complex angle (measured from the normal) of the propagating wave.
        :return: 2x2 propagation matrix through the Material.
        """
        k = self.get_normal_wavevector(wavelength, theta)
        d = self.d
        return np.array([[exp(-1j * k * d), 0],
                        [0, exp(1j * k * d)]])

    def get_normal_wavevector(self, wavelength, theta):
        """
        Gets the normal component of the wavevector of the light traveling in the Material for a certain wavelength and
        angle.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param theta: Complex angle (measured from the normal) of the propagating wave.
        :return: Value of the normal component of the wavevector.
        """
        n = self.get_n(wavelength)
        k = 2 * pi * n * cos(theta) / wavelength
        return k

    def get_n(self, wavelength, energy=None):
        """
        Gets the complex refractive index of the Material according to its model.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's refractive index.
        """
        if self.model == 'Constant':
            n = self.n
        elif self.model == 'Drude':
            eps_r = self.get_eps(wavelength, energy=energy)/epsilon_0
            ndx = sqrt(eps_r.real + sqrt(eps_r.real**2 + eps_r.imag**2))/sqrt(2)
            kppa = sqrt(-eps_r.real + sqrt(eps_r.real**2 + eps_r.imag**2))/sqrt(2)
            n = ndx + 1j*kppa
        else:
            n = self.interpolate_n_value(wavelength, energy=energy)
        return n

    def get_eps(self, wavelength, energy=None):
        """
        Gets the complex permittivity of the Material according to its model.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's permittivity.
        """
        if self.model == 'Constant':
            eps_r = self.n**2
        elif self.model == 'Drude':
            eps_r = self.drude_model(wavelength, energy=energy)
        else:
            n = self.get_n(wavelength, energy=energy)
            eps_1 = n.real**2 - n.imag**2
            eps_2 = 2*n.real*n.imag
            eps_r = eps_1 + 1j*eps_2
        return epsilon_0 * eps_r

    def get_mu(self, wavelength, energy=None):
        """
        Gets the complex permeability of the Material according to its model.

        Currently materials are assumed to be non-magnetic so the function returns the vacuum permeability, but in the
        future this would look similar to Material.get_eps().

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's permeability.
        """
        return mu_0

    @staticmethod
    def get_energy(wavelength):
        """
        Gets energy from vacuum wavelength.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :return: Energy of the wave in eV.
        """
        return h * c / wavelength / e * 1.0e9

    def interpolate_n_value(self, wavelength, energy=None):
        """
        Performs a linear interpolation on the Material's properties table to get the refractive index from data.

        :param wavelength: Wavelength (in nanometres) at which to interpolate value.
        :param energy: Energy of the propagating wave in electron-volts. Overrides wavelength if provided.
        :return: Complex refractive index.
        """
        if energy is None:
            x = self.get_energy(wavelength)
        else:
            x = energy
        arg_closest = np.argmin(np.abs(self.n_table['energy'] - x))
        if self.n_table['energy'][arg_closest] - x < 0:
            arg_lower = arg_closest
            if arg_lower+1 == len(self.n_table['energy']):
                return self.n_table['n'][arg_lower]    # supplied energy is out of bounds of data
            arg_upper = arg_closest+1
        elif self.n_table['energy'][arg_closest] - x > 0:
            arg_upper = arg_closest
            if arg_upper == 0:
                return self.n_table['n'][arg_upper]    # supplied energy is out of bounds of data
            arg_lower = arg_closest-1
        else:
            return self.n_table['n'][arg_closest]
        x_0 = self.n_table['energy'][arg_lower]
        x_1 = self.n_table['energy'][arg_upper]
        y_0 = self.n_table['n'][arg_lower]
        y_1 = self.n_table['n'][arg_upper]
        return y_0*(1-(x-x_0)/(x_1-x_0)) + y_1*((x-x_0)/(x_1-x_0))      # linear interpolation

    def drude_model(self, wavelength, energy=None):
        """
        Gets the complex relative permittivity of the Material using the Drude-Lorenz model.

        :param wavelength: Vacuum wavelength of the propagating wave in nanometres.
        :param energy: Energy of the propagating wave in electron-volts. Overrides wavelength if provided.
        :return: Material's permittivity.
        """
        if energy is None:
            w = self.get_energy(wavelength)*e/hbar
        else:
            w = energy*e/hbar

        sigma = CONDUCTIVITIES[self.name]
        N = ELECTRON_DENSITIES[self.name]

        tau = electron_mass*sigma/(N*e**2)
        w_p = sqrt((N*e**2)/(epsilon_0*electron_mass))

        eps_1 = 1 - ((w_p*tau)**2)/(1+(w*tau)**2)
        eps_2 = (tau*w_p**2)/(w*(1+(w*tau)**2))
        return eps_1 + 1j*eps_2

    def import_properties(self):
        """ Imports material properties from Excel file. """
        sheet = pd.read_excel(FILENAMES[self.model], sheet_name=self.name.capitalize(), usecols='A:C')
        en = np.array(sheet['Energy'])
        n = np.array(sheet['n'])
        k = np.array(sheet['k'])
        if self.model == 'Palik':    # Palik has flipped data
            en = np.flip(en)    # hacky fix - should sort the 3 arrays by ascending energy
            n = np.flip(n)
            k = np.flip(k)
        cond = [not (a or b) for a, b in zip(np.isnan(n), np.isnan(k))]    # remove empty elements
        en = en[cond]
        n = n[cond]
        k = k[cond]
        props = {'energy': en, 'n': n + 1j*k}
        return props
