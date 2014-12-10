
from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Array

import numpy as np

from Atlas import Thrust, ActuatorDiskInducedVelocity, LiftDrag
from Atlas import VortexRing
# from Atlas import VortexRingC as VortexRing  # use cython compiled version


class Aero(Assembly):

    def __init__(self, Ns):
        super(Aero, self).__init__()

        # initial values required to size arrays
        a0 = np.zeros(1)
        y0 = np.zeros(Ns+1)
        n0 = np.zeros(Ns)

        # inputs
        self.add('b',     Int(0,    iotype="in",   desc="number of blades"))
        self.add('R',     Float(0., iotype='in', desc='rotor radius'))
        self.add('Ns',    Int(0,    iotype='in',   desc='number of elements'))
        self.add('yN',    Array(y0, iotype="in", desc='node locations'))
        self.add('dr',    Array(n0, iotype="in", desc="length of each element"))
        self.add('r',     Array(n0, iotype="in", desc="radial location of each element"))
        self.add('h',     Float(0., iotype="in", desc="height of rotor"))

        self.add('ycmax', Float(0., iotype="in"))

        self.add('rho',   Float(0., iotype='in', desc='air density'))
        self.add('visc',  Float(0., iotype='in', desc='air viscosity'))
        self.add('vw',    Float(0., iotype='in', desc='wind velocity'))
        self.add('vc',    Float(0., iotype='in', desc='vertical velocity'))
        self.add('Omega', Float(0., iotype='in', desc='rotor angular velocity'))

        self.add('c',     Array(n0, iotype='in', desc='chord distribution'))
        self.add('Cl',    Array(n0, iotype='in', desc='lift coefficient distribution'))
        self.add('d',     Array(n0, iotype='in', desc='spar diameter distribution'))

        self.add('yWire', Array(a0, iotype='in', desc='location of wire attachment along span'))
        self.add('zWire', Float(0., iotype='in', desc='depth of wire attachement'))
        self.add('tWire', Float(0., iotype='in', desc='thickness of wire'))

        self.add('Cm',    Array(n0, iotype='in', desc=''))
        self.add('xtU',   Array(n0, iotype='in', desc='fraction of laminar flow on the upper surface'))
        self.add('xtL',   Array(n0, iotype='in', desc='fraction of laminar flow on the lower surface'))

        # configure
        self.add('thrust', Thrust(Ns))
        self.connect('Ns',        'thrust.Ns')
        self.connect('yN',        'thrust.yN')
        self.connect('dr',        'thrust.dr')
        self.connect('r',         'thrust.r')
        self.connect('ycmax',     'thrust.ycmax')
        self.connect('Cl',        'thrust.Cl')
        self.connect('c',         'thrust.c')
        self.connect('rho',       'thrust.rho')
        self.connect('Omega',     'thrust.Omega')

        self.add('induced', ActuatorDiskInducedVelocity(Ns))
        self.connect('Ns',        'induced.Ns')
        self.connect('r',         'induced.r')
        self.connect('dr',        'induced.dr')
        self.connect('R',         'induced.R')
        self.connect('b',         'induced.b')
        self.connect('h',         'induced.h')
        self.connect('vc',        'induced.vc')
        self.connect('rho',       'induced.rho')
        self.connect('thrust.dT', 'induced.dT')

        self.add('lift_drag', LiftDrag(Ns))
        self.connect('yN',         'lift_drag.yN')
        self.connect('Ns',         'lift_drag.Ns')
        self.connect('dr',         'lift_drag.dr')
        self.connect('r',          'lift_drag.r')
        self.connect('rho',        'lift_drag.rho')
        self.connect('visc',       'lift_drag.visc')
        self.connect('vw',         'lift_drag.vw')
        self.connect('vc',         'lift_drag.vc')
        self.connect('Omega',      'lift_drag.Omega')
        self.connect('c',          'lift_drag.c')
        self.connect('Cl',         'lift_drag.Cl')
        self.connect('d',          'lift_drag.d')
        self.connect('yWire',      'lift_drag.yWire')
        self.connect('zWire',      'lift_drag.zWire')
        self.connect('tWire',      'lift_drag.tWire')
        self.connect('Cm',         'lift_drag.Cm')
        self.connect('xtU',        'lift_drag.xtU')
        self.connect('xtL',        'lift_drag.xtL')
        self.connect('induced.vi', 'lift_drag.vi')
        self.connect('thrust.chordFrac', 'lift_drag.chordFrac')

        self.create_passthrough('induced.vi')
        self.create_passthrough('lift_drag.phi')
        self.create_passthrough('lift_drag.Re')
        self.create_passthrough('lift_drag.Cd')
        self.create_passthrough('lift_drag.Fblade')

        self.driver.workflow.add('thrust')
        self.driver.workflow.add('induced')
        self.driver.workflow.add('lift_drag')


class Aero2(Assembly):

    def __init__(self, Ns):
        super(Aero2, self).__init__()

        # initial values required to size arrays
        a0 = np.zeros(1)
        y0 = np.zeros(Ns+1)
        n0 = np.zeros(Ns)
        q0 = np.zeros((6*(Ns+1), 1))

        # inputs
        self.add('b',        Int(0,    iotype="in",   desc="number of blades"))
        self.add('R',        Float(0., iotype='in', desc='rotor radius'))
        self.add('Ns',       Int(0,    iotype='in',   desc='number of elements'))
        self.add('yN',       Array(y0, iotype="in", desc='node locations'))
        self.add('dr',       Array(n0, iotype="in", desc="length of each element"))
        self.add('r',        Array(n0, iotype="in", desc="radial location of each element"))
        self.add('h',        Float(0., iotype="in", desc="height of rotor"))

        self.add('ycmax',    Float(0., iotype="in"))

        self.add('rho',      Float(0., iotype='in', desc='air density'))
        self.add('visc',     Float(0., iotype='in', desc='air viscosity'))
        self.add('vw',       Float(0., iotype='in', desc='wind velocity'))
        self.add('vc',       Float(0., iotype='in', desc='vertical velocity'))
        self.add('Omega',    Float(0., iotype='in', desc='rotor angular velocity'))

        self.add('c',        Array(n0, iotype='in', desc='chord distribution'))
        self.add('Cl',       Array(n0, iotype='in', desc='lift coefficient distribution'))
        self.add('d',        Array(n0, iotype='in', desc='spar diameter distribution'))

        self.add('yWire',    Array(a0, iotype='in', desc='location of wire attachment along span'))
        self.add('zWire',    Float(0., iotype='in', desc='depth of wire attachement'))
        self.add('tWire',    Float(0., iotype='in', desc='thickness of wire'))

        self.add('Cm',       Array(n0, iotype='in', desc=''))
        self.add('xtU',      Array(n0, iotype='in', desc='fraction of laminar flow on the upper surface'))
        self.add('xtL',      Array(n0, iotype='in', desc='fraction of laminar flow on the lower surface'))

        self.add('q',        Array(q0, iotype='in', desc='deformation'))

        self.add('anhedral', Float(iotype='in'))

        # configure
        self.add('thrust', Thrust(Ns))
        self.connect('Ns',        'thrust.Ns')
        self.connect('yN',        'thrust.yN')
        self.connect('dr',        'thrust.dr')
        self.connect('r',         'thrust.r')
        self.connect('ycmax',     'thrust.ycmax')
        self.connect('Cl',        'thrust.Cl')
        self.connect('c',         'thrust.c')
        self.connect('rho',       'thrust.rho')
        self.connect('Omega',     'thrust.Omega')

        self.add('induced', VortexRing(Ns))
        self.connect('yN',        'induced.yN')
        self.connect('Ns',        'induced.Ns')
        self.connect('b',         'induced.b')
        self.connect('h',         'induced.h')
        self.connect('vc',        'induced.vc')
        self.connect('rho',       'induced.rho')
        self.connect('thrust.dT', 'induced.dT')
        self.connect('Omega',     'induced.Omega')
        self.connect('q',         'induced.q')
        self.connect('anhedral',  'induced.anhedral')

        self.add('lift_drag', LiftDrag(Ns))
        self.connect('yN',         'lift_drag.yN')
        self.connect('Ns',         'lift_drag.Ns')
        self.connect('dr',         'lift_drag.dr')
        self.connect('r',          'lift_drag.r')
        self.connect('rho',        'lift_drag.rho')
        self.connect('visc',       'lift_drag.visc')
        self.connect('vw',         'lift_drag.vw')
        self.connect('vc',         'lift_drag.vc')
        self.connect('Omega',      'lift_drag.Omega')
        self.connect('c',          'lift_drag.c')
        self.connect('Cl',         'lift_drag.Cl')
        self.connect('d',          'lift_drag.d')
        self.connect('yWire',      'lift_drag.yWire')
        self.connect('zWire',      'lift_drag.zWire')
        self.connect('tWire',      'lift_drag.tWire')
        self.connect('Cm',         'lift_drag.Cm')
        self.connect('xtU',        'lift_drag.xtU')
        self.connect('xtL',        'lift_drag.xtL')
        self.connect('induced.vi', 'lift_drag.vi')
        self.connect('thrust.chordFrac', 'lift_drag.chordFrac')

        self.create_passthrough('induced.vi')
        self.create_passthrough('lift_drag.phi')
        self.create_passthrough('lift_drag.Re')
        self.create_passthrough('lift_drag.Cd')
        self.create_passthrough('lift_drag.Fblade')

        self.driver.workflow.add('thrust')
        self.driver.workflow.add('induced')
        self.driver.workflow.add('lift_drag')
