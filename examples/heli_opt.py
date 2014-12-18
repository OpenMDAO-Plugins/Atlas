from openmdao.main.api import Assembly, set_as_top
from openmdao.lib.datatypes.api import Float
from openmdao.lib.drivers.api import SLSQPdriver

try:
    from pyopt_driver import pyopt_driver
except ImportError:
    pyopt_driver = None

from numpy import pi

from Atlas import AtlasConfiguration, AeroStructural


class ConfigOpt(AtlasConfiguration):
    """ Atlas configuration for single point optimization """

    def __init__(self, Ns):
        super(ConfigOpt, self).__init__(Ns)

        # inputs for optimizer
        self.add('Omega_opt', Float(0., iotype='in', desc='rotor angular velocity'))

    def execute(self):
        super(ConfigOpt, self).execute()

        # use optimizer provided value for Omega
        self.Omega = self.Omega_opt


class AeroStructuralOpt(AeroStructural):
    """ AeroStructural assembly for single point optimization """

    def __init__(self, Ns):
        super(AeroStructuralOpt, self).__init__(Ns)

        # replace config with optimizer driven config
        self.replace('config', ConfigOpt(Ns))

        # create passthroughs for variables used by the optimizer
        self.create_passthrough('config.Omega_opt')
        self.create_passthrough('struc.Mtot')
        self.create_passthrough('results.Ttot')
        self.create_passthrough('results.Ptot')


class HeliOpt(Assembly):
    """ Single point aero-structural optimization """

    def __init__(self, Ns):
        super(HeliOpt, self).__init__()

        # add an optimizer and an AeroStructural assembly
        if pyopt_driver and 'SNOPT' in pyopt_driver._check_imports():
            self.add("driver", pyopt_driver.pyOptDriver())
            self.driver.optimizer = "SNOPT"
            self.driver.options = {
                # any changes to default SNOPT options?
            }
        else:
            print 'SNOPT not available, using SLSQP'
            self.add('driver', SLSQPdriver())

        self.add('aso', AeroStructuralOpt(Ns))

        # objective: minimize total power
        self.driver.add_objective('aso.Ptot')

        # parameter: rotor speed
        self.driver.add_parameter('aso.Omega_opt',
                                  low=0.15*2*pi, high=0.25*2*pi)
        self.aso.Omega_opt = 0.2*2*pi  # initial value

        # constraint: lift >= weight
        self.driver.add_constraint('aso.Mtot*9.8-aso.Ttot<=0')

        # TODO: optional constraints
        #
        #    if flags.ConFail:
        #       Structural Failure in Rotor Spar (ConFail)
        #       Buckling failure of spar (ConFailBuck)
        #       Tensile failure in wire (ConFailWire)
        #
        #    if flags.ConDef:
        #       Constraints on Maximum Deformation (ConDelta)


if __name__ == '__main__':
    import pylab as plt
    from makeplot import plot_single

    from openmdao.lib.casehandlers.api import JSONCaseRecorder

    opt = set_as_top(HeliOpt(10))
    opt.recorders.append(JSONCaseRecorder(out='heli_opt.json'))
    opt.run()

    # for reference, MATLAB solution:
    #    Omega:   1.0512
    #    Ptot:  421.3185
    print 'Parameter:  Omega       =', opt.aso.config.Omega
    print 'Constraint: Weight-Lift =', (opt.aso.Mtot*9.8-opt.aso.Ttot)
    print 'Objective:  Ptot        =', opt.aso.Ptot

    from openmdao.lib.casehandlers.api import CaseDataset
    dataset = CaseDataset('heli_opt.json', 'json')
    data = dataset.data.by_case().fetch()
    case = data[-1]

    plot_single(case)
