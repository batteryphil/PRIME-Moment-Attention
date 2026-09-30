#!/usr/bin/env python3
"""
Metes-and-Bounds Legal Description Parser
==========================================
Parses surveyor calls from deed descriptions into decimal azimuths, distances (meters),
and calculates all corner coordinates from a Point of Beginning (POB).
Also computes the traverse closure error (misclosure ratio).
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional
from survey_app.cogo import vincenty_direct, vincenty_inverse, FEET_TO_METERS, METERS_TO_FEET


# Unit conversion factors to meters
UNIT_FACTORS_METERS = {
    "feet": FEET_TO_METERS,
    "foot": FEET_TO_METERS,
    "ft": FEET_TO_METERS,
    "'": FEET_TO_METERS,
    "meter": 1.0,
    "meters": 1.0,
    "m": 1.0,
    "rod": 16.5 * FEET_TO_METERS,
    "rods": 16.5 * FEET_TO_METERS,
    "pole": 16.5 * FEET_TO_METERS,
    "poles": 16.5 * FEET_TO_METERS,
    "perch": 16.5 * FEET_TO_METERS,
    "perches": 16.5 * FEET_TO_METERS,
    "chain": 66.0 * FEET_TO_METERS,
    "chains": 66.0 * FEET_TO_METERS,
    "ch": 66.0 * FEET_TO_METERS,
}


def parse_dms_to_degrees(deg_str: str, min_str: Optional[str] = None, sec_str: Optional[str] = None) -> float:
    """Converts Degrees, Minutes, Seconds to decimal degrees."""
    d = float(deg_str)
    m = float(min_str) if min_str is not None and min_str != "" else 0.0
    s = float(sec_str) if sec_str is not None and sec_str != "" else 0.0
    return d + (m / 60.0) + (s / 3600.0)


def quadrant_bearing_to_azimuth(quad1: str, angle_deg: float, quad2: str) -> float:
    """
    Converts quadrant bearing (e.g. N 45° E, S 30° W) to whole-circle azimuth (0° to 360°).
    quad1: 'N' or 'S'
    quad2: 'E' or 'W'
    """
    q1 = quad1.upper()[0]
    q2 = quad2.upper()[0]
    angle = angle_deg % 360.0

    if q1 == 'N' and q2 == 'E':
        return angle
    elif q1 == 'S' and q2 == 'E':
        return 180.0 - angle
    elif q1 == 'S' and q2 == 'W':
        return 180.0 + angle
    elif q1 == 'N' and q2 == 'W':
        return (360.0 - angle) % 360.0
    else:
        return angle


class MetesBoundsParser:
    """
    Robust regex engine for extracting surveyor calls from deed text.
    Handles varied historical and modern formats:
    - N 45° 30' 15" E 150.25 ft
    - S 12-15-00 W, 320.0 feet
    - North 89 degrees 15 minutes East 250.00'
    - N45E 100ft
    - Thence South 88°30'20" East, a distance of 412.50 feet to a 1/2" iron rod set;
    """

    # Comprehensive regex matching surveyor bearings and distances
    CALL_PATTERN = re.compile(
        r"(?:THENCE\s+)?"
        r"(?P<q1>[NS]|North|South)\s*"
        r"(?P<deg>\d+(?:\.\d+)?)\s*(?:°|deg|degrees|\-|\s)\s*"
        r"(?:(?P<min>\d+(?:\.\d+)?)\s*(?:'|min|minutes|\-|\s)\s*)?"
        r"(?:(?P<sec>\d+(?:\.\d+)?)\s*(?:\"|sec|seconds|\s)\s*)?"
        r"(?P<q2>[EW]|East|West)"
        r"[,\s]+"
        r"(?:(?:a\s+)?distance\s+of\s+)?\$?"
        r"(?P<dist>\d+(?:,\d+)*(?:\.\d+)?)\s*"
        r"(?P<unit>feet|foot|ft|'|meters?|m|chains?|ch|rods?|poles?|perches?)?"
        r"(?:[,\s]+(?:to\s+(?:a|an)\s+)?(?P<monument>[^;\.\n]+))?",
        re.IGNORECASE
    )

    def parse_text_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses all traverse calls in text into structured records."""
        calls = []
        clean_text = text.replace("\r", " ")

        for match in self.CALL_PATTERN.finditer(clean_text):
            q1 = match.group("q1")[0].upper()
            q2 = match.group("q2")[0].upper()
            deg_str = match.group("deg")
            min_str = match.group("min")
            sec_str = match.group("sec")

            angle = parse_dms_to_degrees(deg_str, min_str, sec_str)
            azimuth = quadrant_bearing_to_azimuth(q1, angle, q2)

            raw_dist = float(match.group("dist").replace(",", ""))
            unit_str = (match.group("unit") or "feet").lower()
            factor = UNIT_FACTORS_METERS.get(unit_str, FEET_TO_METERS)
            dist_meters = raw_dist * factor
            dist_feet = dist_meters * METERS_TO_FEET

            monument = match.group("monument")
            monument_clean = monument.strip() if monument else "Survey Corner"

            raw_call = match.group(0).strip()
            bearing_str = f"{q1} {int(deg_str)}°"
            if min_str:
                bearing_str += f"{int(float(min_str))}'"
            if sec_str:
                bearing_str += f"{round(float(sec_str), 1)}\""
            bearing_str += f" {q2}"

            calls.append({
                "raw_call": raw_call,
                "bearing_str": bearing_str,
                "q1": q1,
                "q2": q2,
                "angle_deg": angle,
                "azimuth": azimuth,
                "distance_feet": dist_feet,
                "distance_meters": dist_meters,
                "monument": monument_clean,
            })

        return calls

    def compute_polygon_from_pob(
        self,
        pob_lat: float,
        pob_lon: float,
        calls: List[Dict[str, Any]],
        pob_monument: str = "Point of Beginning (POB)"
    ) -> Dict[str, Any]:
        """
        Computes all sequential corner coordinates from POB using WGS-84 Vincenty direct formula.
        Calculates traverse closure error (misclosure) and precision ratio.
        """
        if not calls:
            return {"corners": [], "closure_error_feet": 0.0, "precision_ratio": "N/A", "calls": []}

        corners = [{
            "index": 0,
            "name": pob_monument,
            "lat": pob_lat,
            "lon": pob_lon,
            "is_pob": True,
            "target_call": None,
        }]

        curr_lat = pob_lat
        curr_lon = pob_lon
        total_traverse_len_m = 0.0

        for idx, call in enumerate(calls, 1):
            azimuth = call["azimuth"]
            dist_m = call["distance_meters"]
            total_traverse_len_m += dist_m

            next_lat, next_lon, _ = vincenty_direct(curr_lat, curr_lon, azimuth, dist_m)

            corners.append({
                "index": idx,
                "name": f"Corner {idx} ({call['monument']})",
                "lat": next_lat,
                "lon": next_lon,
                "is_pob": False,
                "incoming_call": call["bearing_str"],
                "incoming_distance_feet": round(call["distance_feet"], 2),
            })

            curr_lat = next_lat
            curr_lon = next_lon

        # Closure error: distance from last computed corner back to POB
        final_lat = curr_lat
        final_lon = curr_lon
        misclosure_m, misclosure_azimuth, _ = vincenty_inverse(final_lat, final_lon, pob_lat, pob_lon)
        misclosure_ft = misclosure_m * METERS_TO_FEET
        total_traverse_len_ft = total_traverse_len_m * METERS_TO_FEET

        if misclosure_ft > 0.001:
            ratio_val = int(total_traverse_len_ft / misclosure_ft)
            precision_ratio_str = f"1 : {ratio_val:,}"
        else:
            precision_ratio_str = "Exact / Closed"

        # Coordinates for GeoJSON polygon (closed loop)
        geojson_coords = [[c["lon"], c["lat"]] for c in corners]
        # Close loop back to POB
        if geojson_coords[0] != geojson_coords[-1]:
            geojson_coords.append(geojson_coords[0])

        return {
            "pob": {"lat": pob_lat, "lon": pob_lon, "monument": pob_monument},
            "corners": corners,
            "total_corners": len(corners) - 1,
            "total_perimeter_feet": round(total_traverse_len_ft, 2),
            "misclosure_feet": round(misclosure_ft, 4),
            "misclosure_azimuth": round(misclosure_azimuth, 2),
            "precision_ratio": precision_ratio_str,
            "geojson_coords": geojson_coords,
        }
