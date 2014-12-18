import numpy as np

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Int, Float, Array


class Thrust(Component):

    def __init__(self, Ns):
        super(Thrust, self).__init__()

        # inputs
        self.add('Ns',    Int(0, iotype='in',   desc='number of elements'))

        self.add('yN',    Array(np.zeros(Ns+1), iotype='in', desc='node locations'))
        self.add('dr',    Array(np.zeros(Ns),   iotype='in', desc='length of each element'))
        self.add('r',     Array(np.zeros(Ns),   iotype='in', desc='radial location of each element'))

        self.add('ycmax', Float(0., iotype='in'))

        self.add('Cl',    Array(np.zeros(Ns), iotype='in', desc='lift coefficient distribution'))
        self.add('c',     Array(np.zeros(Ns), iotype='in', desc='chord distribution'))

        self.add('rho',   Float(0., iotype='in', desc='air density'))
        self.add('Omega', Float(0., iotype='in', desc='rotor angular velocity'))

        # outputs
        self.add('dT',        Array(np.zeros((Ns, 1)), iotype='out', desc='Thrust'))
        self.add('chordFrac', Array(np.zeros(Ns), iotype='out'))

    def execute(self):
        self.chordFrac = np.ones((self.Ns, 1))
        self.dT = np.zeros((self.Ns, 1))

        # Compute multiplyer for partial element
        for index, element in enumerate(self.yN):
            if element < self.ycmax:
                sTrans = index  # determine transitional partial element
        self.chordFrac[sTrans] = self.yN[sTrans+1] - self.ycmax  \
                               / (self.yN[sTrans+1] - self.yN[sTrans])

        # Compute thrust assuming small angles
        for s in range(self.Ns):
            self.dT[s] = self.chordFrac[s] * 0.5 * self.rho \
                       * (self.Omega * self.r[s])**2        \
                       * self.Cl[s] * self.c[s] * self.dr[s]


class ActuatorDiskInducedVelocity(Component):
    """ Compute induced velocity using annual-ring actuator disk theory
    """

    def __init__(self, Ns):
        super(ActuatorDiskInducedVelocity, self).__init__()

        # inputs
        self.add('Ns',  Int(0,    iotype='in', desc='number of elements'))

        self.add('r',   Array(np.zeros(Ns), iotype='in', desc='radial location of each element'))
        self.add('dr',  Array(np.zeros(Ns), iotype='in', desc='length of each element'))

        self.add('R',   Float(0., iotype='in', desc='rotor radius'))
        self.add('b',   Int(0,    iotype='in', desc='number of blades'))
        self.add('h',   Float(0., iotype='in', desc='height of rotor'))
        self.add('vc',  Float(0., iotype='in', desc='vertical velocity'))
        self.add('rho', Float(0., iotype='in', desc='air density'))

        self.add('dT',  Array(np.zeros((Ns, 1)), iotype='in', desc='thrust'))

        # outputs
        self.add('vi',  Array(np.zeros(Ns), iotype='out', desc='induced downwash distribution'))

    def execute(self):
        self.vi = np.zeros((self.Ns, 1))

        for s in range(self.Ns):
            sq = 0.25 * self.vc**2 + \
                 0.25 * self.b * self.dT[s] / (np.pi * self.rho * self.r[s] * self.dr[s])
            self.vi[s] = -0.5*self.vc + np.sqrt(sq)

        # Add ground effect Cheesemen & Benett's
        self.vi /= (1. + (self.R / self.h / 4.) ** 2)
