import unittest
import numpy as np
from rainroute import CFG, fuzzy_risk, decode, GENES

class TestRainRoute(unittest.TestCase):
    def test_fuzzy_risk_bounds(self):
        """Test that fuzzy risk is mathematically bounded between 0 and 1."""
        p = {n: (lo + hi)/2 for n, lo, hi in GENES}
        p["d_high"] = max(p["d_high"], p["d_med"] + 0.05)
        risk = fuzzy_risk(np.array([0.1, 0.5]), np.array([0.0, 0.05]), p)
        self.assertTrue(np.all(risk >= 0))
        self.assertTrue(np.all(risk <= 1))

    def test_decode_logic(self):
        """Test genetic algorithm chromosome decoding bounds."""
        x = np.ones(len(GENES))
        p = decode(x)
        self.assertEqual(p["margin"], int(round(GENES[-1][2])))

if __name__ == '__main__':
    unittest.main()
