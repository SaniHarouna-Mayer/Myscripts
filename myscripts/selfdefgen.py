from diffpy.srfit.fitbase import ProfileGenerator
import numpy as np


class ExpcosGenerator(ProfileGenerator):
    """
    baseline generator. simulate the baseline with a wave function:
    Gaussian(center, *std) * Sinwave(wavelength, cos_amp, sin_amp)
    *std means that the left right parts have different standard deviation.
    The attribute is the parameter in equation named with
    """
    def __init__(self, name):
        ProfileGenerator.__init__(self, name)
        self._newParameter("wavelength", 10.0)
        self._newParameter("center", 2.0)
        self._newParameter("cos_coef", .1)
        self._newParameter("sin_coef", .1)
        self._newParameter("left_std", 5.)
        self._newParameter("right_std", 5.0)
        return

    def __call__(self, x):
        a0 = self.wavelength.value
        a1 = self.center.value
        a2 = self.cos_coef.value
        a3 = self.sin_coef.value
        a4 = self.left_std.value
        a5 = self.right_std.value
        msk = x < a1
        x0 = x[msk]
        x1 = x[np.logical_not(msk)]
        env = np.concatenate([np.exp(-.5*((x0-a1)/a4)**2), np.exp(-.5*((x1-a1)/a5)**2)])
        osc = a2 * np.cos(2.*np.pi*x/a0) + a3 * np.sin(2.*np.pi*x/a0)
        y = env * osc
        return y
