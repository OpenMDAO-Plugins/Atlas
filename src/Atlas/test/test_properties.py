from Atlas import DiscretizeProperties, wire_properties, SparProperties, ChordProperties
import numpy as np
import unittest

from openmdao.util.testutil import assert_rel_error


def relative_err(x, y):
    return (np.abs(x-y)/np.linalg.norm(x)).max()


def absolute_err(x, y):
    return np.abs(x-y).max()


class AtlasTestProperties(unittest.TestCase):

    def test_wire_properties(self):
        props = wire_properties['Pianowire']

        self.assertAlmostEquals(props['RHO'], 7.85e3, 4)
        self.assertAlmostEquals(props['E'], 2.10e11, 3)
        self.assertAlmostEquals(props['ULTIMATE'], 2.62e9, 3)

        props = wire_properties['Vectran']

        self.assertAlmostEquals(props['RHO'], 1.1065e3, 4)
        self.assertAlmostEquals(props['E'], 3.921e10, 3)
        self.assertAlmostEquals(props['ULTIMATE'], 9.828e8, 3)

    def test_discretizeProperties(self):
        comp = DiscretizeProperties(10)

        # set inputs
        comp.Ns       = 10
        comp.ycmax    = np.array([1.4656, 3.2944])
        comp.R        = 10
        comp.c_in     = np.array([0, 0.8000, 1.4000, 0.4000, 0.3600])
        comp.Cl_in    = np.array([1.5000, 1.4300, 1.2300])
        comp.Cm_in    = np.array([-0.1500, -0.1200, -0.1200])
        comp.t_in     = np.array([0.1400, 0.1400, 0.1400])
        comp.xtU_in   = np.array([0.1500, 7.0000, 0.1500])
        comp.xtL_in   = np.array([0.3000, 7.0000, 0.3000])
        comp.xEA_in   = np.array([0.2700, 0.3300, 0.2400])
        comp.yWire    = np.array([5.8852, ])
        comp.d_in     = np.array([0.0874, 0.0505, 0.0315])
        comp.theta_in = np.array([0.3491, 0.3491, 0.3491])
        comp.nTube_in = np.array([4, 4, 4])
        comp.nCap_in  = np.array([0, 0, 0])
        comp.lBiscuit_in = np.array([0.3048, 0.3048, 0.1524])

        # run
        comp.run()

        tol = 5e-4
        cE = np.array([0.2729, 1.3903, 1.1757, 1.0176, 0.8818,
                       0.7602, 0.6507, 0.5528, 0.4666, 0.3925])
        self.assertLess(relative_err(cE, comp.cE), tol)

        cN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertLess(absolute_err(cN, comp.cN), tol)

        c100 = np.array([0.0273, 0.0819, 0.1365, 0.1910, 0.2456, 0.3002, 0.3548,
                         0.4094, 0.4640, 0.5186, 0.5731, 0.6277, 0.6823, 0.7369,
                         1.4045, 1.3766, 1.3507, 1.3263, 1.3033, 1.2815, 1.2606,
                         1.2406, 1.2213, 1.2026, 1.1845, 1.1669, 1.1498, 1.1331,
                         1.1168, 1.1007, 1.0850, 1.0696, 1.0545, 1.0396, 1.0249,
                         1.0104, 0.9961, 0.9821, 0.9682, 0.9545, 0.9409, 0.9275,
                         0.9143, 0.9012, 0.8882, 0.8754, 0.8628, 0.8502, 0.8378,
                         0.8255, 0.8134, 0.8014, 0.7894, 0.7777, 0.7660, 0.7545,
                         0.7430, 0.7317, 0.7205, 0.7095, 0.6985, 0.6877, 0.6770,
                         0.6663, 0.6559, 0.6455, 0.6352, 0.6251, 0.6151, 0.6052,
                         0.5954, 0.5857, 0.5761, 0.5667, 0.5574, 0.5482, 0.5391,
                         0.5301, 0.5213, 0.5125, 0.5039, 0.4954, 0.4871, 0.4788,
                         0.4707, 0.4627, 0.4548, 0.4470, 0.4393, 0.4318, 0.4244,
                         0.4171, 0.4099, 0.4029, 0.3960, 0.3891, 0.3825, 0.3759,
                         0.3694, 0.3631])
        self.assertLess(relative_err(c100, comp.c100), tol)

        Cl = np.array([1.5000, 1.4987, 1.4604, 1.4239, 1.3940,
                       1.3642, 1.3344, 1.3046, 1.2747, 0.8299])
        self.assertLess(relative_err(Cl, comp.Cl), tol)

        Cm = np.array([-0.1500, -0.1494, -0.1330, -0.1200, -0.1200,
                       -0.1200, -0.1200, -0.1200, -0.1200, -0.1200])
        self.assertLess(relative_err(Cm, comp.Cm), tol)

        t = np.array([0.1400, 0.1400, 0.1400, 0.1400, 0.1400,
                      0.1400, 0.1400, 0.1400, 0.1400, 0.1400])
        self.assertLess(relative_err(t, comp.t), tol)

        xtU = np.array([0.0500, 0.1500, 0.1500, 0.1500, 0.1500,
                        0.1500, 0.1500, 0.1500, 0.1500, 0.1500])
        self.assertLess(relative_err(xtU, comp.xtU), tol)

        xtL = np.array([0.0500, 0.3000, 0.3000, 0.3000, 0.3000,
                        0.3000, 0.3000, 0.3000, 0.3000, 0.3000])
        self.assertLess(relative_err(xtL, comp.xtL), tol)

        xEA = np.array([0.2700, 0.2711, 0.3039, 0.3272, 0.3138,
                        0.3004, 0.2870, 0.2736, 0.2601, 0.2467])
        self.assertLess(relative_err(xEA, comp.xEA), tol)

        d = np.array([0.0843, 0.0780, 0.0718, 0.0655, 0.0592,
                      0.0530, 0.0477, 0.0431, 0.0384, 0.0338])
        self.assertLess(relative_err(d, comp.d), tol)

        theta = np.array([0.3491, 0.3491, 0.3491, 0.3491, 0.3491,
                          0.3491, 0.3491, 0.3491, 0.3491, 0.3491])
        self.assertLess(relative_err(theta, comp.theta), tol)

        nTube = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.assertLess(relative_err(nTube, comp.nTube), tol)

        nCap = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertLess(absolute_err(nCap, comp.nCap), tol)

        lBiscuit = np.array([0.3048, 0.3048, 0.3048, 0.3048, 0.3048,
                             0.3048, 0.2820, 0.2450, 0.2080, 0.1709])
        self.assertLess(relative_err(lBiscuit, comp.lBiscuit), tol)

        yN = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertLess(relative_err(yN, comp.yN), tol)

        yE = np.array([0.5000, 1.5000, 2.5000, 3.5000, 4.5000,
                       5.5000, 6.5000, 7.5000, 8.5000, 9.5000])
        self.assertLess(relative_err(yE, comp.yE), tol)

    def test_sparProperties(self):
        comp = SparProperties(10)

        comp.yN = np.array([0, 14.5057])
        comp.d = np.array([0.1016, ])
        comp.theta = np.array([0.6109, ])
        comp.nTube = np.array([4, ])
        comp.nCap = np.array([0, 0])
        comp.lBiscuit = np.array([0.3048, ])
        #comp.CFRPType = 1
        comp.CFRPType = 'NCT301-1X HS40 G150 33 +/-2%RW'

        comp.run()

        tol = 0.0001
        assert_rel_error(self, comp.EIx[0], 23704.383, tol)
        assert_rel_error(self, comp.EIz[0], 23704.383, tol)
        assert_rel_error(self, comp.EA[0], 18167620.0, tol)
        assert_rel_error(self, comp.GJ[0], 2.2828e4, tol)
        assert_rel_error(self, comp.mSpar[0], 4.7244, tol)

    def test_chordProperties(self):
        comp = ChordProperties(10)
        comp.yN = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        comp.c = np.array([0.2729, 1.3903, 1.1757, 1.0176, 0.8818, 0.7602, 0.6507, 0.5528, 0.4666, 0.3925])
        comp.d = np.array([0.0843, 0.0780, 0.0718, 0.0655, 0.0592, 0.0530, 0.0477, 0.0431, 0.0384, 0.0338])
        comp.GWing = 1
        comp.xtU = np.array([0.0500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500])

        comp.run()

        tol = 0.001
        expected_mChord = [0.018497, 0.167168, 0.133574, 0.110904, 0.092785, 0.077596, 0.064695, 0.053795, 0.044734, 0.037342]
        expected_xCGChord = [0.37481, 0.29252, 0.29447, 0.29634, 0.29838, 0.30073, 0.30350, 0.30677, 0.31059, 0.31498]
        for i, (e_mChord, e_xCGChord) in enumerate(zip(expected_mChord, expected_xCGChord)):
            assert_rel_error(self, comp.mChord[i], e_mChord, tol)
            assert_rel_error(self, comp.xCGChord[i], e_xCGChord, tol)


if __name__ == "__main__":
    unittest.main()
