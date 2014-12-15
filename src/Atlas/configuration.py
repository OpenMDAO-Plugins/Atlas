import numpy as np

from math import pi, sqrt

from openmdao.main.api import Component, VariableTree
from openmdao.main.datatypes.api import Int, Float, Array, Str, Enum, VarTree

from properties import JointProperties


class Flags(VariableTree):
    Opt          = Int(1, desc='0 - single run, 1 - optimization')
    ConFail      = Int(0, desc='1 - use structural failure as a constraint on optimization')
    ConWireCont  = Int(0, desc='1 - use wire length continuity as a constraint to set appropriate wire forces in multi-point optimizations')
    ConJigCont   = Int(0, desc='1 - use jig continuity')
    ConDef       = Int(0, desc='1 - constraints on maximum deformation of the rotor')
    MultiPoint   = Int(4, desc='0 - single point optimization, 1 - 4 point optimization (h=0.5, h=3, wind case, gravity load)')
    Quad         = Int(1, desc='0 - prop drive, 1 - quad rotor')
    FreeWake     = Int(1, desc='0 - momentum theory, 1 - free vortex ring wake')
    PlotWake     = Int(0, desc='0 - dont plot wake, 1 - plot wake ')
    DynamicClimb = Int(0, desc='0 - vc imposes downward velocity, 1 - vc represents climb (final altitude depends on Nw)')
    Cover        = Int(0, desc='0 - no cover over root rotor blades, 1 - cover')
    Load         = Int(0, desc='0 - normal run, 1 - gravity forces only, 2 - prescribed load from pLoad')
    Cdfit        = Int(1, desc='0 - analytic model for drag coefficient, 1 - curve fit on BE airfoils')
    GWing        = Int(1, desc='0 - Daedalus style wing, 1 - Gossamer style wing (changes amount of laminar flow)')
    AeroStr      = Int(1, desc='0 - Assume flat wing, 1 - take deformation into account')
    Movie        = Int(0, desc='0 - dont save animation, 1 - save animation')
    wingWarp     = Int(0, desc='0 - no twist constraint, >0 - twist constraint at wingWarp')

    CFRPType     = Str('NCT301-1X HS40 G150 33 +/-2%RW', desc='type of carbon fibre reinforced polymer')

    WireType     = Enum('Pianowire', ('Pianowire', 'Vectran'), desc='Material to be used for lift wire')


class PrescribedLoad(VariableTree):
    y = Float(9.9999, desc='Point load location')
    pointZ = Float(0.15*9.8, desc='N')
    pointM = Float(0, desc='Nm')
    distributedX = Float(0, desc='N/m')
    distributedZ = Float(0, desc='N/m')
    distributedM = Float(0, desc='Nm/m')


class AtlasConfiguration(Component):
    """ Atlas configuration
    """

    def __init__(self, Ns):
        super(AtlasConfiguration, self).__init__()

        # outputs (manual configuration)
        self.add('flags', VarTree(Flags(), iotype='out'))

        self.add('b',     Int(2,      iotype='out', desc='number of blades'))
        self.add('Ns',    Int(Ns,     iotype='out', desc='number of elements'))

        self.add('R',     Float(10.0, iotype='out', desc='rotor radius'))
        self.add('H',     Float(0.5,  iotype='out', desc='height of aircraft'))

        self.add('ycmax', Array(np.array([1.4656, 3.2944]), iotype='out'))

        self.add('rho',   Float(1.18,       iotype='out', desc='air density'))
        self.add('vw',    Float(0.0,        iotype='out', desc='wind velocity'))
        self.add('vc',    Float(0.0,        iotype='out', desc='vertical velocity'))
        self.add('visc',  Float(1.78e-5,    iotype='out', desc='air viscosity'))
        self.add('Omega', Float(0.165*2*pi, iotype='out', desc='rotor angular velocity'))

        self.add('c', Array(np.array([0, 0.8, 1.4, 0.4, 0.36, 0., 0., 0., 0., 0.]),  # ideal curve
                    iotype='out', desc='chord distribution'))
        self.add('d', Array(np.array([3.442, 1.99, 1.239, 0., 0., 0., 0., 0., 0., 0.])*2.54/100,  # inches to meters
                    iotype='out', desc='spar diameter distribution'))
        self.add('Cl', Array(np.array([1.5, 1.43, 1.23, 0., 0., 0., 0., 0., 0., 0.]),  # 0.5 m No canard lift distribution (as flown)
                    iotype='out', desc='lift coefficient distribution'))
        self.add('Cm', Array(np.array([-0.15, -0.12, -0.12, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc=''))
        self.add('t', Array(np.array([0.14, 0.14, 0.14, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc=''))
        self.add('xEA', Array(np.array([0.27, 0.33, 0.24, 0., 0., 0., 0., 0., 0., 0.]),  # percent chord
                    iotype='out', desc=''))
        self.add('xtU', Array(np.array([0.15, 7, 0.15, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc='fraction of laminar flow on the upper surface'))
        self.add('xtL', Array(np.array([0.30, 7, 0.30, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc='fraction of laminar flow on the lower surface'))

        self.add('theta', Array(np.array([20, 20, 20, 0., 0., 0., 0., 0., 0., 0.])*pi/180,  # deg to rad
                    iotype='out', desc='wrap angle'))
        self.add('nTube', Array(np.array([4, 4, 4, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc='number of tube layers'))
        self.add('nCap', Array(np.array([0, 0, 0, 0., 0., 0., 0., 0., 0., 0.]),
                    iotype='out', desc='number of cap strips'))
        self.add('lBiscuit', Array(np.array([12, 12, 6, 0., 0., 0., 0., 0., 0., 0.])*2.54/100,  # inches to meters
                    iotype='out', desc='unsupported biscuit length'))

        self.add('dQuad',        Float(4*2.54/100,  iotype='out', desc='diameter of quad rotor struts'))
        self.add('thetaQuad',    Float(35*pi/180,   iotype='out', desc='wrap angle of quad rotor struts'))
        self.add('nTubeQuad',    Int(4,             iotype='out', desc='number of CFRP layers in quad rotor struts'))
        self.add('lBiscuitQuad', Float(12*2.54/100, iotype='out', desc=''))
        self.add('hQuad',        Float(3.0,         iotype='out', desc='height of quad-rotor truss'))

        self.add('collective',   Float(0*pi/180, iotype='out', desc='collective angle in radians'))
        self.add('etaP',         Float(0.0,      iotype='out', desc=''))

        self.add('yWire',        Array([5.8852], iotype='out', desc='location of wire attachment along span'))  # actual spars
        self.add('zWire',        Float(1.0,      iotype='out', desc='depth of wire attachement'))
        self.add('tWire',        Float(.0016,    iotype='out', desc='thickness of wire'))  # steel
        self.add('TWire',        Array([1100.],  iotype='out', desc=''))
        self.add('TEtension',    Float(50.0,     iotype='out', desc=''))

        self.add('anhedral',     Float(0.8*pi/180, iotype='out'))

        self.add('mElseRotor',   Float(5.11,    iotype='out', desc=''))
        self.add('mElseCentre',  Float(6.487+3, iotype='out', desc=''))
        self.add('mElseR',       Float(0.032,   iotype='out', desc=''))
        self.add('mPilot',       Float(71.0,    iotype='out', desc='mass of pilot (kg)'))

        # outputs (calculated)
        self.add('yN',    Array(np.zeros(Ns+1), iotype='out', desc='node locations'))
        self.add('dr',    Array(np.zeros(Ns),   iotype='out', desc='length of each element'))
        self.add('r',     Array(np.zeros(Ns),   iotype='out', desc='radial location of each element'))

        self.add('RQuad', Float(iotype='out', desc='distance from centre of helicopter to centre of quad rotors'))
        self.add('h',     Float(iotype='out', desc='height of rotor'))

        self.add('Jprop',   VarTree(JointProperties(), iotype='out'))
        self.add('presLoad', VarTree(PrescribedLoad(), iotype='out'))

        # force execution, since there are no 'inputs'
        self.force_execute = True

    def execute(self):
        self.yN = np.linspace(0, self.R, self.Ns+1)

        self.dr = np.zeros(self.Ns)
        self.r = np.zeros(self.Ns)

        for s in range(self.Ns):
            self.dr[s] = self.yN[s+1] - self.yN[s]     # length of each element
            self.r[s] = 0.5*(self.yN[s] + self.yN[s+1])

        self.RQuad = sqrt(2*self.R**2) + 0.05

        self.h = self.H + self.zWire

        # Properties at joint location for buckling analysis
        self.Jprop.d = self.d[1]
        self.Jprop.theta = self.theta[1]
        self.Jprop.nTube = int(self.nTube[1])
        self.Jprop.nCap  = int(self.nCap[1])
        self.Jprop.lBiscuit = self.lBiscuit[1]
