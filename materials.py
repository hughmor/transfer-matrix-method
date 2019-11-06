import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, c, e, epsilon_0, mu_0, pi, elementary_charge, electron_mass
from cmath import cos, sqrt, exp

PATH = 'data/'
FILENAMES = {
    'Johnson': PATH + 'materialconstants_j.xlsx',
    'Palik': PATH + 'materialconstants_p.xlsx'}

INDICES = {
    "Silicon": 3.48,
    "Silica": 1.48,
    "Air": 1.0
}

ELECTRON_DENSITIES = {
    "Gold": 5.9e28,
    "Silver": 5.86e28
}

CONDUCTIVITIES = {
    "Gold": 4.1e7,
    "Silver": 6.3e7
}


class Material:
    """
    The Material object contains the optical properties of one material in a multi-layer structure.

    :param name: The name of the material.
    :param metal_model: The model to use for metal properties. Defaults to 'Constant'.
    :param thickness: The thickness of the slab in nanometres.
    :param index: Constant refractive index.
    """

    def __init__(self, name, metal_model='Constant', thickness=-1, index=-1):
        self.name = name.capitalize()
        self.d = thickness
        self.model = metal_model.capitalize()
        self.n = index
        if self.n < 0:
            if self.model == 'Constant':
                self.n = INDICES[self.name]
            elif self.model == 'Drude':
                return
            elif self.model == 'Palik' or self.model == 'Johnson':
                self.properties = self.import_properties()
            else:
                print('ERROR: Invalid Model Supplied')
                return

    def dynamical_matrix(self, wavelength, theta, p):
        """
        Generates the dynamical boundary matrix for the Material at a specific wavelength, angle, and polarization.

        :param wavelength: Vacuum wavelength of the propagating wave.
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

    def get_normal_wavevector(self, wavelength, theta):
        """
        Gets the normal component of the wavevector of the light traveling in the Material for a certain wavelength and
        angle.

        :param wavelength: Vacuum wavelength of the propagating wave.
        :param theta: Complex angle (measured from the normal) of the propagating wave.
        :return:
        """
        n = self.get_n(wavelength)
        k = 2 * pi * n * cos(theta) / wavelength
        return k

    def propagation_matrix(self, wavelength, theta):
        """
        Generates the propagation matrix through the Material at a specific wavelength and angle.

        :param wavelength: Vacuum wavelength of the propagating wave.
        :param theta: Complex angle (measured from the normal) of the propagating wave.
        :return: 2x2 propagation matrix through the Material.
        """
        k = self.get_normal_wavevector(wavelength, theta)
        d = self.d
        return np.array([[exp(1j * k * d), 0],
                         [0, exp(-1j * k * d)]])

    def get_n(self, wvl, energy=None):
        """
        Gets the complex refractive index of the Material according to its model.

        :param wvl: Vacuum wavelength of the propagating wave.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's refractive index.
        """
        if self.model == 'Constant':
            n = self.n
        elif self.model == 'Drude':
            eps = self.get_eps(wvl, energy=energy)
            ndx = ((eps.real+(eps.real**2+eps.imag**2)**0.5)**0.5)/sqrt(2)
            kppa = ((-eps.real+(eps.real**2+eps.imag**2)**0.5)**0.5)/sqrt(2)
            n = ndx + 1j*kppa
        else:
            n = self.interpolate_property_value(wvl, energy=energy)
        return n

    def get_eps(self, wvl, energy=None):
        """
        Gets the complex permittivity of the Material according to its model.

        :param wvl: Vacuum wavelength of the propagating wave.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's permittivity.
        """
        if self.model == 'Constant':
            eps_r = self.n**2
        elif self.model == 'Drude':
            eps_r = self.drude_model(wvl, energy=energy)
        else:
            n = self.get_n(wvl, energy=energy)
            eps_1 = n.real**2 - n.imag**2
            eps_2 = 2*n.real*n.imag
            eps_r = eps_1 + 1j*eps_2
        return epsilon_0 * eps_r

    def get_mu(self, wvl, energy=None):
        """
        Gets the complex permeability of the Material according to its model.

        Currently materials are assumed to be non-magnetic so the function returns the vacuum permeability, but in the
        future this would look similar to Material.get_eps().

        :param wvl: Vacuum wavelength of the propagating wave.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Material's permeability.
        """
        return mu_0

    @staticmethod
    def get_energy(wvl):
        """
        Gets energy from vacuum wavelength.

        :param wvl: Vacuum wavelength of the propagating wave.
        :return: Energy of the wave.
        """
        return h * c / wvl

    def interpolate_property_value(self, wvl, energy=None):
        """
        Performs a linear interpolation on the Material's properties table to get the refractive index from data.

        :param wvl: Wavelength at which to interpolate value.
        :param energy: Energy of the propagating wave. Overrides wavelength if provided.
        :return: Complex refractive index.
        """
        if energy is None:
            x = self.get_energy(wvl)
        else:
            x = energy
        arg_closest = np.argmin(np.abs(self.properties['energy'] - x))
        if self.properties['energy'][arg_closest] - x < 0:
            arg_upper = arg_closest+1
            arg_lower = arg_closest
        elif self.properties['energy'][arg_closest] - x > 0:
            arg_upper = arg_closest
            arg_lower = arg_closest-1
        else:
            return self.properties['n'][arg_closest]
        x_0 = self.properties['energy'][arg_lower]
        x_1 = self.properties['energy'][arg_upper]
        y_0 = self.properties['n'][arg_lower]
        y_1 = self.properties['n'][arg_upper]
        return y_0*(1-(x-x_0)/(x_1-x_0)) + y_1*((x-x_0)/(x_1-x_0))      # linear interpolation

    def drude_model(self, wavelength, energy=None):
        """
        Gets the complex relative permittivity of the material.

        :param wavelength:
        :param energy:
        :return:
        """
        if energy is None:
            w = self.get_energy(wavelength)
        else:
            w = energy
        sigma = CONDUCTIVITIES[self.model]
        N = ELECTRON_DENSITIES[self.model]
        tau = electron_mass*sigma/(N*elementary_charge**2)
        w_p = sqrt((N*elementary_charge**2)/(epsilon_0*electron_mass))
        eps_1 = 1 - ((w_p*tau)**2)/(1+(w*tau)**2)
        eps_2 = (tau*w_p**2)/(w*(1+(w*tau)**2))
        return eps_1 + 1j*eps_2

    # deprecated: new drude model function is above. this is kept for comparison purposes
    def _drude_model(self, wavelength, energy=None):
        if energy is None:
            w = self.get_energy(wavelength)
        else:
            w = energy

        if self.name == 'Gold':
            gmf = 1
        elif self.name == 'Silver':
            gmf = 0
        else:
            raise ValueError('ERROR: Metal not supported')
        w_Ps = {0: 8.5546, 0.5: 9.0218, 1: 8.9234}
        Gamma_Ps = {0: 0.022427, 0.5: 0.16713, 1: 0.042389}
        eps_infs = {0: 1.7318, 0.5: 2.2838, 1: 2.2715}
        wg1s = {0: 4.0575, 0.5: 3.0209, 1: 2.6652}
        w01s = {0: 3.926, 0.5: 2.7976, 1: 2.3957}
        Gamma_1s = {0: 0.017723, 0.5: 0.18833, 1: 0.1788}
        A1s = {0: 51.217, 0.5: 22.996, 1: 73.251}
        w02s = {0: 4.1655, 0.5: 3.34, 1: 3.5362}
        Gamma_2s = {0: 0.18819, 0.5: 0.68309, 1: 0.35467}
        A2s = {0: 30.77, 0.5: 57.54, 1: 40.007}

        w_P = self._drude_helper(w_Ps, gmf)
        Gamma_P = self._drude_helper(Gamma_Ps, gmf)
        eps_inf = self._drude_helper(eps_infs, gmf)
        wg1 = self._drude_helper(wg1s, gmf)
        w01 = self._drude_helper(w01s, gmf)
        Gamma_1 = self._drude_helper(Gamma_1s, gmf)
        A1 = self._drude_helper(A1s, gmf)
        w02 = self._drude_helper(w02s, gmf)
        Gamma_2 = self._drude_helper(Gamma_2s, gmf)
        A2 = self._drude_helper(A2s, gmf)

        eps_CP1 = A1*(-np.sqrt(wg1-w01)*np.log(1-((w+1j*Gamma_1)/w01)**2)/(2*(w+1j*Gamma_1)**2) +
                      2*np.sqrt(wg1)*np.arctanh(np.sqrt((wg1-w01)/wg1))/(w+1j*Gamma_1)**2 -
                      np.sqrt(w+1j*Gamma_1-wg1)*np.arctan(np.sqrt((wg1-w01)/(w+1j*Gamma_1-wg1)))/(w+1j*Gamma_1)**2 -
                      np.sqrt(w+1j*Gamma_1-wg1)*np.arctanh(np.sqrt((wg1-w01)/(w+1j*Gamma_1+wg1)))/(w+1j*Gamma_1)**2)
        eps_CP2 = A2*np.log(1-((w+1j*Gamma_2)/w02)**2)/(2*(w+1j*Gamma_2)**2)
        return eps_inf - w_P**2/(w**2 + 1j*w*Gamma_P) + eps_CP1 + eps_CP2

    @staticmethod
    def _drude_helper(vals, gmf):
        return (2*vals[1] - 4*vals[0.5] + 2*vals[0])*gmf**2 + (-vals[1] + 4*vals[0.5] - 3*vals[0])*gmf + vals[0]

    def import_properties(self):
        """ Imports material properties from Excel file. """
        sheet = pd.read_excel(FILENAMES[self.model], sheet_name=self.name.capitalize(), usecols='A:C')
        en = np.array(sheet['Energy'])
        n = np.array(sheet['n'])
        k = np.array(sheet['k'])
        if self.model == 'Palik':   # hacky fix, should sort arrays by energy
            en = np.flip(en)
            n = np.flip(n)
            k = np.flip(k)
        cond = [not (a or b) for a, b in zip(np.isnan(n), np.isnan(k))]  # remove empty elements
        en = en[cond]
        n = n[cond]
        k = k[cond]
        props = {'energy': h * c / en / e, 'n': n + 1j*k}
        return props

    def plot_properties(self):
        """ Plots the complex refractive index values. """
        if self.model == 'Johnson' or self.model == 'Palik':
            plt.figure()
            plt.plot(self.properties['energy']*10**6, self.properties['n'].real, '-')
            plt.plot(self.properties['energy']*10**6, self.properties['n'].imag, '--')
            plt.xscale('log')
            plt.xlabel('Wavelength (micro-m)')
            plt.yscale('log')
            plt.ylabel('Value')
            plt.show()
        else:
            raise ValueError('Cant call plot_properties on Material that uses {} model'.format(self.model.lower()))