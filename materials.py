import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as consts


FILENAMES = {
    'Johnson': 'materialconstants_j.xlsx',
    'Palik': 'materialconstants_p.xlsx'}

INDICES = {
    "Silicon": 3.48,
    "Silica": 1.48,
    "Air": 1.0,
    "Aluminum gallium arsenide": None,
    'Test1': 1.2,
    'Test2': 1.5,
    'Test3': 1.3,
    'Test4': 1.25,
    'Test5': 1.1
}


class Material:
    def __init__(self, name, model='Constant', thickness=-1, index=-1):
        self.name = name.capitalize()
        self.d = thickness
        if self.name in INDICES.keys():
            self.model = 'Constant'
        else:
            self.model = model
        self.n = index
        if self.n < 0:
            if self.model is 'Constant':
                self.n = INDICES[self.name]
            elif self.model is 'Drude':
                return
            elif self.model is 'Palik' or self.model is 'Johnson':
                self.properties = self.import_properties()
            else:
                print('ERROR: Invalid Model Supplied')
                return

    def import_properties(self):
        sheet = pd.read_excel(FILENAMES[self.model], sheet_name=self.name.capitalize(), usecols='A:C')
        e = np.array(sheet['Energy'])
        n = np.array(sheet['n'])
        k = np.array(sheet['k'])
        if self.model is 'Palik':
            e = np.flip(e)
            n = np.flip(n)
            k = np.flip(k)
        props = {'energy': consts.h * consts.c / e / consts.e, 'n': n + 1j*k}
        return props

    def get_n(self, wvl):
        if self.model is 'Constant':
            return self.n
        elif self.model is 'Drude':
            return self.drude_model(wvl)
        else:
            return self.interpolate_value(wvl)

    def get_eps(self, wvl):
        n = self.get_n(wvl)
        return consts.epsilon_0 * n**2

    @staticmethod
    def get_mu(wvl):
        return consts.mu_0      # assuming non magnetic materials

    def get_energy(self, wvl):
        return consts.h * consts.c / wvl

    def interpolate_value(self, wvl, prop='n'):
        if prop is 'n':
            x = self.get_energy(wvl)
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
            return y_0*(1-(x-x_0)/(x_1-x_0)) + y_1*((x-x_0)/(x_1-x_0))      # linear interp
        else:
            print('BIG FUCK UP')
            return

    def drude_model(self, wavelength):
        eps_inf = -1
        eps_CP1 =-1
        eps_CP2 =-1
        w_P =-1
        w =-1
        Gamma_P =-1
        print('BIG FUCK UP')
        return eps_inf - (w_P**2/w^2 + 1j*w*Gamma_P) + eps_CP1 + eps_CP2

    def plot_properties(self):
        plt.figure()
        plt.plot(self.properties['energy']*10**6, self.properties['n'].real, '-')
        plt.plot(self.properties['energy']*10**6, self.properties['n'].imag, '--')
        plt.xscale('log')
        plt.xlabel('Wavelength (micro-m)')
        plt.yscale('log')
        plt.ylabel('Value')
        plt.show()

# gold,silver,silicon,silica,gold