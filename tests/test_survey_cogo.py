#!/usr/bin/env python3
"""
Unit Tests: PRIME Survey & COGO Geodesic Engine
===============================================
Verifies:
1. WGS-84 Vincenty forward and inverse geodetic precision
2. Metes-and-bounds deed parser and polygon closure calculation
3. PRIME-Net symbolic acreage and ALTA/NSPS compliance verification
"""

import unittest
import math
from survey_app.cogo import (
    vincenty_direct,
    vincenty_inverse,
    relative_bearing,
    haversine_distance_and_bearing,
    polygon_geodesic_area_and_perimeter,
)
from survey_app.metes_bounds_parser import (
    MetesBoundsParser,
    parse_dms_to_degrees,
    quadrant_bearing_to_azimuth,
)
from survey_app.primenet_verifier import PrimeNetSurveyVerifier


class TestSurveyCogo(unittest.TestCase):
    def setUp(self):
        self.pob_lat = 30.2672
        self.pob_lon = -97.7431
        self.parser = MetesBoundsParser()
        self.verifier = PrimeNetSurveyVerifier()

    def test_dms_to_degrees(self):
        """Tests conversion of DMS to decimal degrees."""
        # 45 deg 30 min 00 sec = 45.5 deg
        self.assertAlmostEqual(parse_dms_to_degrees("45", "30", "0"), 45.5, places=5)
        # 12 deg 15 min 36 sec = 12 + 15/60 + 36/3600 = 12.26 deg
        self.assertAlmostEqual(parse_dms_to_degrees("12", "15", "36"), 12.26, places=5)

    def test_quadrant_bearing_to_azimuth(self):
        """Tests quadrant bearing to 0-360 azimuth conversion."""
        self.assertEqual(quadrant_bearing_to_azimuth("N", 45.0, "E"), 45.0)
        self.assertEqual(quadrant_bearing_to_azimuth("S", 45.0, "E"), 135.0)
        self.assertEqual(quadrant_bearing_to_azimuth("S", 45.0, "W"), 225.0)
        self.assertEqual(quadrant_bearing_to_azimuth("N", 45.0, "W"), 315.0)

    def test_vincenty_roundtrip_precision(self):
        """Verifies forward Vincenty direct + inverse returns sub-millimeter precision."""
        target_dist_m = 250.0  # 250 meters
        target_az = 65.5      # Azimuth 65.5 degrees

        lat2, lon2, rev_az = vincenty_direct(self.pob_lat, self.pob_lon, target_az, target_dist_m)
        calc_dist, calc_az, _ = vincenty_inverse(self.pob_lat, self.pob_lon, lat2, lon2)

        # Distance error must be < 0.001 meters (1 millimeter)
        self.assertLess(abs(calc_dist - target_dist_m), 0.001)
        # Bearing error must be < 0.0001 degrees
        self.assertLess(abs(calc_az - target_az), 0.0001)

    def test_relative_bearing_navigation(self):
        """Verifies relative turn angle calculation for compass HUD."""
        # Target is at 90° (East), device points 0° (North) -> Turn right 90°
        self.assertEqual(relative_bearing(90.0, 0.0), 90.0)
        # Target is at 270° (West), device points 0° (North) -> Turn left -90°
        self.assertEqual(relative_bearing(270.0, 0.0), -90.0)
        # Target straight ahead at 45°, device points 45° -> 0°
        self.assertEqual(relative_bearing(45.0, 45.0), 0.0)

    def test_metes_bounds_traversal_and_closure(self):
        """Verifies parsing of a rectangular traverse and closure calculation."""
        deed_text = """
        BEGINNING at an iron pin found;
        THENCE North 00° 00' 00" East, a distance of 100.00 feet to an iron rod found;
        THENCE North 90° 00' 00" East, a distance of 200.00 feet to a rebar set;
        THENCE South 00° 00' 00" East, a distance of 100.00 feet to a stone monument;
        THENCE South 90° 00' 00" West, a distance of 200.00 feet to the POINT OF BEGINNING.
        """
        calls = self.parser.parse_text_calls(deed_text)
        self.assertEqual(len(calls), 4)

        poly = self.parser.compute_polygon_from_pob(self.pob_lat, self.pob_lon, calls)
        self.assertEqual(len(poly["corners"]), 5)  # POB + 4 corners
        self.assertAlmostEqual(poly["total_perimeter_feet"], 600.0, delta=0.1)
        # Closure error should be nearly 0 (< 0.01 feet)
        self.assertLess(poly["misclosure_feet"], 0.01)

    def test_primenet_acreage_and_alta_audit(self):
        """Verifies symbolic acreage and ALTA compliance checks."""
        # 1 acre = 43,560 sq ft
        res = self.verifier.verify_acreage(43560.0 * 3.5)
        self.assertEqual(res["acres_decimal"], 3.5)
        self.assertEqual(res["acres_fraction"], "7/2")
        self.assertTrue(res["exact_verified"])

        # Test ALTA Urban Standard (minimum 1:10,000)
        # 0.05 ft misclosure over 1,000 ft perimeter = 1:20,000 -> PASS
        audit_pass = self.verifier.audit_survey_closure(0.05, 1000.0, "urban")
        self.assertTrue(audit_pass["alta_nsps_compliant"])
        self.assertIn("PASSED", audit_pass["status"])

        # 0.5 ft misclosure over 1,000 ft perimeter = 1:2,000 -> FAIL
        audit_fail = self.verifier.audit_survey_closure(0.5, 1000.0, "urban")
        self.assertFalse(audit_fail["alta_nsps_compliant"])


if __name__ == "__main__":
    unittest.main()
