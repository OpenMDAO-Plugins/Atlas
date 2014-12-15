import numpy as np
from numpy import pi, arctan2

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Int, Float, Array, VarTree
from openmdao.lib.drivers.api import FixedPointIterator

from Atlas import AtlasConfiguration, DiscretizeProperties, \
                  Aero, Aero2, Structures, Fblade

from openmdao.util.log import enable_trace  # , disable_trace


class Results(Component):
    """ collect/calculate results from aerostructural computation
    """

    def __init__(self, Ns):
        super(Results, self).__init__()

        # initial values required to size arrays
        y0 = np.zeros(Ns+1)
        n0 = np.zeros(Ns)
        q0 = np.zeros((6*(Ns+1), 1))

        # inputs
        self.add('b',          Int(0,    iotype='in', desc='number of blades'))
        self.add('Ns',         Int(0,    iotype='in', desc='number of elements'))
        self.add('yN',         Array(y0, iotype='in', desc='node locations'))
        self.add('yE',         Array(n0, iotype='in', desc=''))
        self.add('cE',         Array(n0, iotype='in', desc='chord of each element'))
        self.add('Cl',         Array(n0, iotype='in', desc='lift coefficient distribution'))
        self.add('q',          Array(q0, iotype='in', desc='deformation'))
        self.add('phi',        Array(n0, iotype='in', desc=''))
        self.add('collective', Float(0., iotype='in', desc='collective angle in radians'))
        self.add('fblade',     VarTree(Fblade(Ns), iotype='in'))
        self.add('Mtot',       Float(0.0, iotype='in', desc='total mass'))

        # outputs
        self.add('di',         Array(n0, iotype='out', desc='dihedral angle'))
        self.add('alphaJig',   Array(n0, iotype='out', desc='aerodynamic jig angle'))
        self.add('Ttot',       Float(0., iotype='out', desc='total thrust'))
        self.add('Qtot',       Float(0., iotype='out', desc='total torque'))
        self.add('MomRot',     Float(0., iotype='out', desc='total moment'))
        self.add('Ptot',       Float(0., iotype='out', desc='total powers'))

    def execute(self):
        # Compute aerodynamic jig angle
        self.alphaJig = np.zeros(self.cE.shape)

        qq = np.zeros((6, self.Ns+1))
        for s in range(1, self.Ns+1):
            qq[:, s] = self.q[s*6:s*6+6].T

        Clalpha = 2*pi
        for s in range(0, len(self.yN)-1):
            self.alphaJig[s] = self.Cl[s] / Clalpha            \
                             - (qq[4, s] + qq[4, s+1]) / 2     \
                             + self.phi[s] - self.collective

        # Compute dihedral angle
        self.di = np.zeros((self.Ns, 1))
        for s in range(0, self.Ns):
            self.di[s] = arctan2(qq[2, s+1] - qq[2, s], self.yN[s+1] - self.yN[s])

        # Compute totals
        # (Note: reshaping is due to numpy vs MATLAB 1D array shapes.. should scrub this)
        self.Ttot   = np.sum(self.fblade.Fz.reshape(-1, 1) * np.cos(self.di)) * self.b * 4
        self.MomRot = np.sum(self.fblade.Fz.reshape(-1, 1) * self.yE.reshape(-1, 1))
        self.Qtot   = np.sum(self.fblade.Q)  * self.b * 4
        Pitot       = np.sum(self.fblade.Pi) * self.b * 4
        Pptot       = np.sum(self.fblade.Pp) * self.b * 4
        self.Ptot   = Pptot + Pitot  # non-covered centre

        # print self.parent.name, '\t', 'Ptot:', self.Ptot


class Switch(Component):
    """ select the appropriate source for blade force data
    """

    def __init__(self, Ns):
        super(Switch, self).__init__()

        # inputs
        self.add('fblade_initial', VarTree(Fblade(Ns), iotype='in'))
        self.add('fblade_updated', VarTree(Fblade(Ns), iotype='in'))

        # outputs
        self.add('fblade',          VarTree(Fblade(Ns), iotype='out'))

        self.initial = True
        self.force_execute = True

    def execute(self):
        if self.initial:
            self.fblade = self.fblade_initial
            self.initial = False
        else:
            self.fblade = self.fblade_updated


class AeroStructural(Assembly):
    """ Performs an aerodynamic and structural computation on a single
        configuration given the full set of design parameters. The aerodynamic
        computation returns the thrust, torque, power, and induced velocity. The
        structural computation first computes the mass of the helicopter based on
        the structural description of the spars and chord lengths. It then
        computes the deformation of the spars, the strains, and the resulting
        factor of safety for each of the failure modes.
    """

    def __init__(self, Ns):
        super(AeroStructural, self).__init__()

        # configuration
        self.add('config', AtlasConfiguration(Ns))

        # Take parameterized properties and get property for each element
        self.add('discrete', DiscretizeProperties(Ns))
        self.connect('config.Ns',           'discrete.Ns')
        self.connect('config.ycmax',        'discrete.ycmax')
        self.connect('config.R',            'discrete.R')
        self.connect('config.c',            'discrete.c_in')
        self.connect('config.Cl',           'discrete.Cl_in')
        self.connect('config.Cm',           'discrete.Cm_in')
        self.connect('config.t',            'discrete.t_in')
        self.connect('config.xtU',          'discrete.xtU_in')
        self.connect('config.xtL',          'discrete.xtL_in')
        self.connect('config.xEA',          'discrete.xEA_in')
        self.connect('config.yWire',        'discrete.yWire')
        self.connect('config.d',            'discrete.d_in')
        self.connect('config.theta',        'discrete.theta_in')
        self.connect('config.nTube',        'discrete.nTube_in')
        self.connect('config.nCap',         'discrete.nCap_in')
        self.connect('config.lBiscuit',     'discrete.lBiscuit_in')

        # First run Aero calc with simple blade-element model. Lift is
        # accurate with this simple model since Cl is pre-defined
        self.add('aero', Aero(Ns))
        self.connect('config.b',            'aero.b')
        self.connect('config.R',            'aero.R')
        self.connect('config.Ns',           'aero.Ns')
        self.connect('discrete.yN',         'aero.yN')
        self.connect('config.dr',           'aero.dr')
        self.connect('config.r',            'aero.r')
        self.connect('config.h',            'aero.h')
        self.connect('config.ycmax[0]',     'aero.ycmax')
        self.connect('config.rho',          'aero.rho')
        self.connect('config.visc',         'aero.visc')
        self.connect('config.vw',           'aero.vw')
        self.connect('config.vc',           'aero.vc')
        self.connect('config.Omega',        'aero.Omega')
        self.connect('discrete.cE',         'aero.c')
        self.connect('discrete.Cl',         'aero.Cl')
        self.connect('discrete.d',          'aero.d')
        self.connect('config.yWire',        'aero.yWire')
        self.connect('config.zWire',        'aero.zWire')
        self.connect('config.tWire',        'aero.tWire')
        self.connect('discrete.Cm',         'aero.Cm')
        self.connect('discrete.xtU',        'aero.xtU')
        self.connect('discrete.xtL',        'aero.xtL')

        # Then run Aero calc with more accurate Vortex method
        self.add('aero2', Aero2(Ns))
        self.connect('config.b',            'aero2.b')
        self.connect('config.R',            'aero2.R')
        self.connect('config.Ns',           'aero2.Ns')
        self.connect('discrete.yN',         'aero2.yN')
        self.connect('config.dr',           'aero2.dr')
        self.connect('config.r',            'aero2.r')
        self.connect('config.h',            'aero2.h')
        self.connect('config.ycmax[0]',     'aero2.ycmax')
        self.connect('config.rho',          'aero2.rho')
        self.connect('config.visc',         'aero2.visc')
        self.connect('config.vw',           'aero2.vw')
        self.connect('config.vc',           'aero2.vc')
        self.connect('config.Omega',        'aero2.Omega')
        self.connect('discrete.cE',         'aero2.c')
        self.connect('discrete.Cl',         'aero2.Cl')
        self.connect('discrete.d',          'aero2.d')
        self.connect('config.yWire',        'aero2.yWire')
        self.connect('config.zWire',        'aero2.zWire')
        self.connect('config.tWire',        'aero2.tWire')
        self.connect('discrete.Cm',         'aero2.Cm')
        self.connect('discrete.xtU',        'aero2.xtU')
        self.connect('discrete.xtL',        'aero2.xtL')
        self.connect('config.anhedral',     'aero2.anhedral')

        # Structures calc
        self.add('struc', Structures(Ns))
        self.connect('config.flags',        'struc.flags')
        self.connect('discrete.yN',         'struc.yN')
        self.connect('discrete.d',          'struc.d')
        self.connect('discrete.theta',      'struc.theta')
        self.connect('discrete.nTube',      'struc.nTube')
        self.connect('discrete.nCap',       'struc.nCap')
        self.connect('discrete.lBiscuit',   'struc.lBiscuit')
        self.connect('config.Jprop',        'struc.Jprop')
        self.connect('config.b',            'struc.b')
        self.connect('discrete.cE',         'struc.cE')
        self.connect('discrete.xEA',        'struc.xEA')
        self.connect('discrete.xtU',        'struc.xtU')
        self.connect('config.dQuad',        'struc.dQuad')
        self.connect('config.thetaQuad',    'struc.thetaQuad')
        self.connect('config.nTubeQuad',    'struc.nTubeQuad')
        self.connect('config.lBiscuitQuad', 'struc.lBiscuitQuad')
        self.connect('config.RQuad',        'struc.RQuad')
        self.connect('config.hQuad',        'struc.hQuad')
        self.connect('config.ycmax[0]',     'struc.ycmax')
        self.connect('config.yWire',        'struc.yWire')
        self.connect('config.zWire',        'struc.zWire')
        self.connect('config.tWire',        'struc.tWire')
        self.connect('config.TWire',        'struc.TWire')
        self.connect('config.TEtension',    'struc.TEtension')
        self.connect('config.mElseRotor',   'struc.mElseRotor')
        self.connect('config.mElseCentre',  'struc.mElseCentre')
        self.connect('config.mElseR',       'struc.mElseR')
        self.connect('config.R',            'struc.R')
        self.connect('config.mPilot',       'struc.mPilot')
        self.connect('config.presLoad',     'struc.presLoad')

        # converge aero and structures via fixed point iteration
        self.add('switch', Switch(Ns))
        self.connect('aero.Fblade',         'switch.fblade_initial')
        self.connect('aero2.Fblade',        'switch.fblade_updated')
        self.connect('switch.fblade',       'struc.fblade')
        # self.connect('struc.q',             'aero2.q')  # via constraint

        self.add('iterate', FixedPointIterator())
        # self.iterate.max_iteration = 2  # 2 passes to emulate MATLAB code
        self.iterate.tolerance = 1e-10
        self.iterate.add_parameter('aero2.q', low=-1e999, high=1e999)
        self.iterate.add_constraint('aero2.q = struc.q')

        # make sure we have a valid parameter value for first iteration
        q_dim = 6*(self.config.Ns+1)
        self.aero2.q = np.zeros((q_dim, 1))

        # calculate results
        self.add('results', Results(Ns))
        self.connect('config.b',            'results.b')
        self.connect('config.Ns',           'results.Ns')
        self.connect('discrete.yN',         'results.yN')
        self.connect('discrete.yE',         'results.yE')
        self.connect('discrete.cE',         'results.cE')
        self.connect('discrete.Cl',         'results.Cl')
        self.connect('struc.q',             'results.q')
        self.connect('struc.Mtot',          'results.Mtot')
        self.connect('aero2.phi',           'results.phi')
        self.connect('config.collective',   'results.collective')
        self.connect('aero2.Fblade',        'results.fblade')

        self.driver.workflow.add('config')
        self.driver.workflow.add('discrete')
        self.driver.workflow.add('aero')
        self.driver.workflow.add('iterate')
        self.driver.workflow.add('results')


if __name__ == "__main__":
    # enable_trace()

    top = AeroStructural()
    top.run()

    print 'AeroStructural Results'
    print '----------------------'
    print 'Ttot   =', top.results.Ttot
    print 'MomRot =', top.results.MomRot
    print 'Qtot   =', top.results.Qtot
    print 'Ptot   =', top.results.Ptot
