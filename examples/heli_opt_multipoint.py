import time

from openmdao.main.api import Assembly, set_as_top
from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Float, Array
from openmdao.lib.drivers.api import SLSQPdriver

try:
    from pyopt_driver import pyopt_driver
except ImportError:
    pyopt_driver = None

from openmdao.util.log import enable_trace, disable_trace

import numpy as np
from numpy import pi

from Atlas import AtlasConfiguration, AeroStructural


class ConfigLow(AtlasConfiguration):
    """ Atlas configuration for optimization of low altitude case
    """

    def __init__(self, Ns):
        super(ConfigLow, self).__init__(Ns)

        # add optimizer controlled inputs
        self.add('Omega_opt', Float(0., iotype='in', desc='rotor angular velocity'))
        self.add('H_opt',     Float(0., iotype='in', desc='height of aircraft'))

    def execute(self):
        super(ConfigLow, self).execute()

        # use optimizer provided values
        self.Omega = self.Omega_opt
        self.h     = self.H_opt + self.zWire


class AeroStructuralLow(AeroStructural):
    """ AeroStructural assembly for low altitude case in multipoint optimization
    """

    def __init__(self, Ns):
        super(AeroStructuralLow, self).__init__(Ns)

        # replace config with optimizer driven config
        self.replace('config', ConfigLow(Ns))

        # create passthroughs for variables used by the optimizer
        self.create_passthrough('config.Omega_opt')
        self.create_passthrough('config.H_opt')
        self.create_passthrough('struc.Mtot')
        self.create_passthrough('results.Ttot')
        self.create_passthrough('results.Ptot')


class ConfigHigh(AtlasConfiguration):
    """ Atlas configuration for optimization of high altitude case
    """

    def __init__(self, Ns):
        super(ConfigHigh, self).__init__(Ns)

        # add optimizer controlled inputs
        self.add('Omega_opt', Float(0., iotype='in', desc='rotor angular velocity'))
        self.add('Cl0_opt',   Float(0., iotype='in', desc=''))
        self.add('Cl1_opt',   Float(0., iotype='in', desc=''))
        self.add('H_opt',     Float(0., iotype='in', desc='height of aircraft'))
        self.add('TWire_opt', Float(0., iotype='in', desc=''))

    def execute(self):
        super(ConfigHigh, self).execute()

        # use optimizer provided values
        self.Omega = self.Omega_opt
        self.Cl[0] = self.Cl0_opt
        self.Cl[1] = self.Cl1_opt
        self.h     = self.H_opt + self.zWire
        self.TWire = [self.TWire_opt]


class AeroStructuralHigh(AeroStructural):
    """ AeroStructural assembly for high altitude case in multipoint optimization
    """

    def __init__(self, Ns):
        super(AeroStructuralHigh, self).__init__(Ns)

        # replace config with optimizer driven config
        self.replace('config', ConfigHigh(Ns))

        # create passthroughs for variables used by the optimizer
        self.create_passthrough('config.Omega_opt')
        self.create_passthrough('config.Cl0_opt')
        self.create_passthrough('config.Cl1_opt')
        self.create_passthrough('config.H_opt')
        self.create_passthrough('config.TWire_opt')
        self.create_passthrough('struc.Mtot')
        self.create_passthrough('results.Ttot')
        self.create_passthrough('results.Ptot')


class ConfigWind(AtlasConfiguration):
    """ Atlas configuration for optimization of wind case
    """

    def __init__(self, Ns):
        super(ConfigWind, self).__init__(Ns)

        # add optimizer controlled inputs
        self.add('Omega_opt',  Float(0., iotype='in', desc='rotor angular velocity'))
        self.add('OmegaRatio', Float(0., iotype='in', desc=''))

        self.add('Cl_opt',     Array(np.zeros(Ns), iotype='in', desc=''))

        self.add('H_opt',      Float(0., iotype='in', desc='height of aircraft'))
        self.add('TWire_opt',  Float(0., iotype='in', desc=''))
        self.add('vw_opt',     Float(0., iotype='in', desc='wind velocity'))

    def execute(self):
        super(ConfigWind, self).execute()

        # use optimizer provided values
        self.Omega = (self.Omega_opt**3 * self.OmegaRatio)**(1./3.)
        self.Cl    = self.Cl_opt
        self.h     = self.H_opt + self.zWire
        self.TWire = [self.TWire_opt]
        self.vw    = self.vw_opt

        # FIXME: the following two flags are ignored
        self.flags.FreeWake = 0  # momentum theory
        self.flags.AeroStr  = 0  # assume flat wing (no deformation)


class AeroStructuralWind(AeroStructural):
    """ AeroStructural assembly for wind case in multipoint optimization
    """

    def __init__(self, Ns):
        super(AeroStructuralWind, self).__init__(Ns)

        # replace config with optimizer driven config
        self.replace('config', ConfigWind(Ns))

        # create passthroughs for variables used by the optimizer
        self.create_passthrough('config.Omega_opt')
        self.create_passthrough('config.OmegaRatio')

        self.create_passthrough('config.Cl_opt')
        self.create_passthrough('config.H_opt')
        self.create_passthrough('config.TWire_opt')
        self.create_passthrough('config.vw_opt')

        self.create_passthrough('struc.Mtot')
        self.create_passthrough('results.Ttot')
        self.create_passthrough('results.Ptot')


class ConfigGravity(AtlasConfiguration):
    """ Atlas configuration for optimization of gravity case
    """

    def __init__(self, Ns):
        super(ConfigGravity, self).__init__(Ns)

        # add optimizer controlled inputs
        self.add('Omega_opt',  Float(0., iotype='in', desc='rotor angular velocity'))
        self.add('OmegaRatio', Float(0., iotype='in'))

        self.add('Cl_opt',     Array(np.zeros(Ns), iotype='in'))

        self.add('H_opt',      Float(0., iotype='in', desc='height of aircraft'))
        self.add('TWire_opt',  Float(0., iotype='in', desc=''))

    def execute(self):
        super(ConfigGravity, self).execute()

        # use optimizer provided values
        self.Omega = (self.Omega_opt**3 * self.OmegaRatio)**(1./3.)
        self.Cl    = self.Cl_opt
        self.h     = self.H_opt + self.zWire
        self.TWire = [self.TWire_opt]

        # gravity and wire forces only
        self.flags.Load = 1

        # FIXME: the following two flags are ignored
        self.flags.FreeWake = 0  # momentum theory
        self.flags.AeroStr  = 0  # assume flat wing (no deformation)


class AeroStructuralGravity(AeroStructural):
    """ AeroStructural assembly for gravity case in multipoint optimization
    """

    def __init__(self, Ns):
        super(AeroStructuralGravity, self).__init__(Ns)

        # replace config with optimizer driven config
        self.replace('config', ConfigGravity(Ns))

        # create passthroughs for variables used by the optimizer
        self.create_passthrough('config.Omega_opt')
        self.create_passthrough('config.OmegaRatio')

        self.create_passthrough('config.Cl_opt')
        self.create_passthrough('config.H_opt')
        self.create_passthrough('config.TWire_opt')

        self.create_passthrough('struc.Mtot')
        self.create_passthrough('results.Ttot')
        self.create_passthrough('results.Ptot')


class Multipoint(Assembly):
    """ Assembly for multipoint AeroStructural optimization.

        Evaluates AeroStructural for four cases:
            low altitude
            high altitude
            wind
            gravity only
    """

    def __init__(self, Ns):
        super(Multipoint, self).__init__()

        # configuration inputs
        self.add('alt_low',    Float(0., iotype='in', desc='low altitude'))
        self.add('alt_high',   Float(0., iotype='in', desc='high altitude'))
        self.add('alt_ratio',  Float(0., iotype='in', desc='proportion of time near ground'))

        self.add('TWire_high', Float(0., iotype='in', desc=''))
        self.add('TWire_wind', Float(0., iotype='in', desc=''))
        self.add('TWire_grav', Float(0., iotype='in', desc=''))

        self.add('OmegaRatio', Float(0., iotype='in', desc=''))

        self.add('vw',         Float(0., iotype='in', desc='wind velocity'))

        self.add('Cl_max',     Array(np.zeros(Ns), iotype='in', desc=''))

        # optimizer parameters
        self.add('Omega_low',  Float(0., iotype='in', desc='rotor angular velocity, low altitude'))
        self.add('Omega_high', Float(0., iotype='in', desc='rotor angular velocity, high altitude'))
        self.add('Cl0_high',   Float(0., iotype='in', desc=''))
        self.add('Cl1_high',   Float(0., iotype='in', desc=''))

        # outputs
        self.add('P',          Float(0., iotype='out', desc=''))

        # low altitude
        self.add('low', AeroStructuralLow(Ns))

        self.connect('Omega_low', 'low.Omega_opt')
        self.connect('alt_low',   'low.H_opt')

        self.create_passthrough('low.Mtot', 'Mtot_low')
        self.create_passthrough('low.Ttot', 'Ttot_low')

        # high altitude
        # need a different rotor speed and lift distribution at altitude
        self.add('high', AeroStructuralHigh(Ns))

        self.connect('Omega_high', 'high.Omega_opt')
        self.connect('Cl0_high',   'high.Cl0_opt')
        self.connect('Cl1_high',   'high.Cl1_opt')
        self.connect('alt_high',   'high.H_opt')
        self.connect('TWire_high', 'high.TWire_opt')

        self.create_passthrough('high.Mtot', 'Mtot_high')
        self.create_passthrough('high.Ttot', 'Ttot_high')

        # wind case
        self.add('wind', AeroStructuralWind(Ns))

        self.connect('Omega_high', 'wind.Omega_opt')
        self.connect('OmegaRatio', 'wind.OmegaRatio')
        self.connect('Cl_max',     'wind.Cl_opt')
        self.connect('alt_high',   'wind.H_opt')
        self.connect('TWire_wind', 'wind.TWire_opt')
        self.connect('vw',         'wind.vw_opt')

        # gravity case
        self.add('grav', AeroStructuralGravity(Ns))

        self.connect('Omega_high', 'grav.Omega_opt')
        self.connect('OmegaRatio', 'grav.OmegaRatio')
        self.connect('Cl_max',     'grav.Cl_opt')
        self.connect('alt_high',   'grav.H_opt')
        self.connect('TWire_grav', 'grav.TWire_opt')

        # total power
        self.connect('alt_ratio*low.Ptot + (1 - alt_ratio)*high.Ptot', 'P')

        self.driver.workflow.add(['low', 'high', 'wind', 'grav'])


class HeliOptM(Assembly):
    """ Multipoint aero-structural optimization """

    def __init__(self, Ns):
        super(HeliOptM, self).__init__()

        # add an optimizer and a multi-point AeroStructural assembly
        if pyopt_driver and 'SNOPT' in pyopt_driver._check_imports():
            self.add("driver", pyopt_driver.pyOptDriver())
            self.driver.optimizer = "SNOPT"
            self.driver.options = {
                # any changes to default SNOPT options?
            }
        else:
            print 'SNOPT not available, using SLSQP'
            self.add('driver', SLSQPdriver())

        # Set force_fd to True. This will force the derivative system to treat
        # the whole model as a single entity to finite difference it and force
        # the system decomposition to put all of it into an opaque system.
        #
        # Full-model FD is preferable anyway because:
        # 1. There are no derivatives defined for any comps
        # 2. There are a lot of interior connections that would likely make
        #    it much slower if you allow openmdao to finite difference the
        #    subassemblies like it normally does.
        self.driver.gradient_options.force_fd = True

        self.add('mp', Multipoint(Ns))

        self.mp.alt_low   = 0.5       # low altitude
        self.mp.alt_high  = 3.5       # high altitude
        self.mp.alt_ratio = 35./60.   # proportion of time near ground

        self.mp.TWire_high = 900
        self.mp.TWire_wind = 2100
        self.mp.TWire_grav = 110

        self.mp.OmegaRatio = 2

        self.mp.vw = 0/3.6   # zero

        self.mp.Cl_max = [1.4, 1.35, 1.55, 0., 0., 0., 0., 0., 0., 0.]  # max control

        # objective: minimize total power
        self.driver.add_objective('mp.P')

        # parameter: rotor speed
        self.driver.add_parameter('mp.Omega_low',
                                  low=0.15*2*pi, high=0.25*2*pi)
        self.mp.Omega_low = 0.20*2*pi  # initial value

        self.driver.add_parameter('mp.Omega_high',
                                  low=0.15*2*pi, high=0.19*2*pi)
        self.mp.Omega_high = 0.17*2*pi  # initial value

        # parameter: lift distribution at high altitude
        self.driver.add_parameter('mp.Cl0_high',
                                  low=0.8, high=1.4)
        self.mp.Cl0_high = 1.

        self.driver.add_parameter('mp.Cl1_high',
                                  low=0.8, high=1.3)
        self.mp.Cl1_high = 1.

        # constraint: lift >= weight
        self.driver.add_constraint('mp.Mtot_low*9.8-mp.Ttot_low<=0')
        self.driver.add_constraint('mp.Mtot_high*9.8-mp.Ttot_high<=0')

        # TODO: optional constraints
        #    if flags.ConFail:
        #       Structural Failure in Rotor Spar (ConFail)
        #       Buckling failure of spar (ConFailBuck)
        #       Tensile failure in wire (ConFailWire)
        #
        #    if flags.ConDef:
        #       Constraints on Maximum Deformation (ConDelta)
        #
        #    if flags.MultiPoint && flags.ConJigCont:
        #       Consistent jig twist (ConAlphaJig)
        #
        #    if flags.MultiPoint && flags.ConWireCont
        #       Wire stretch consistency (conWire)

        # Optimization Constraints  (not used... yet)
        vrCon = VariableTree()
        vrCon.MaxDelta    = -0.1
        vrCon.MinDelta    = 0.1
        vrCon.FOSmat      = 0.55    # 1.3
        vrCon.FOSbuck     = 0.5     # 1.3
        vrCon.FOSquadbuck = 5.
        vrCon.FOStorbuck  = 0.5     # 1.5
        vrCon.FOSwire     = 0.5     # 2


if __name__ == '__main__':
    # enable_trace()

    opt = set_as_top(HeliOptM(10))

    print 'Starting multipoint optimization at %s ...' % time.strftime('%X')
    time1 = time.time()
    opt.run()
    time2 = time.time()
    print 'Optimization complete at %s (elapsed time: %5.2f minutes)' \
        % (time.strftime('%X'), ((time2-time1)/60))

    print

    print 'Objective:  P =', opt.mp.P

    print 'Constraint: Low Weight-Lift =',  opt.mp.Mtot_low*9.8-opt.mp.Ttot_low
    print 'Constraint: High Weight-Lift =', opt.mp.Mtot_high*9.8-opt.mp.Ttot_high

    print 'Parameter:  Omega (Low) =',  opt.mp.Omega_low
    print 'Parameter:  Omega (High) =', opt.mp.Omega_high
