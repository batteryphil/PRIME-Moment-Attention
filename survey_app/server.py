#!/usr/bin/env python3
"""
Land Survey & GPS Corner Finder Web Server
==========================================
FastAPI backend providing:
- Metes-and-bounds deed parsing into WGS-84 survey corners
- Real-time GPS corner navigation vectors (range & relative bearing)
- ALTA/NSPS survey closure auditing via PRIME-Net / SymPy
- Pin marking, inspection logging, and GeoJSON / KML / CSV export
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from survey_app.cogo import (
    vincenty_inverse,
    relative_bearing,
    haversine_distance_and_bearing,
    polygon_geodesic_area_and_perimeter,
    METERS_TO_FEET,
)
from survey_app.metes_bounds_parser import MetesBoundsParser
from survey_app.primenet_verifier import PrimeNetSurveyVerifier

app = FastAPI(title="PRIME Land Survey & GPS Corner Navigation Engine")

# CORS middleware for mobile field access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = MetesBoundsParser()
verifier = PrimeNetSurveyVerifier()

PINS_STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "marked_pins.json"))


# --- Pydantic Request Models ---
class ParseDeedRequest(BaseModel):
    pob_lat: float
    pob_lon: float
    deed_text: str
    pob_monument: Optional[str] = "Point of Beginning (POB)"
    survey_class: Optional[str] = "urban"


class NavVectorRequest(BaseModel):
    user_lat: float
    user_lon: float
    target_lat: float
    target_lon: float
    device_heading: Optional[float] = 0.0


class MarkPinRequest(BaseModel):
    pin_name: str
    target_lat: float
    target_lon: float
    user_lat: float
    user_lon: float
    gps_accuracy_feet: float
    status: str  # "FOUND_ROD", "FLAGGED", "BURIED", "MISSING", "SET_STAKE"
    notes: Optional[str] = ""


# --- Helper: Cardinal Direction ---
def degrees_to_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]


# --- API Routes ---

@app.post("/api/survey/parse-deed")
def api_parse_deed(req: ParseDeedRequest):
    """Parses deed text and calculates corner coordinates with ALTA closure audit."""
    calls = parser.parse_text_calls(req.deed_text)
    if not calls:
        raise HTTPException(
            status_code=400,
            detail="No surveyor calls found. Check deed format (e.g. 'THENCE North 45° East, 150.0 feet to an iron pin;')."
        )

    poly = parser.compute_polygon_from_pob(
        pob_lat=req.pob_lat,
        pob_lon=req.pob_lon,
        calls=calls,
        pob_monument=req.pob_monument or "Point of Beginning (POB)"
    )

    # Calculate acreage and verify via PRIME-Net
    corner_coords = [(c["lat"], c["lon"]) for c in poly["corners"]]
    parcel_audit = verifier.verify_parcel_polygon(corner_coords, survey_class=req.survey_class or "urban")
    closure_audit = verifier.audit_survey_closure(
        misclosure_ft=poly["misclosure_feet"],
        total_perimeter_ft=poly["total_perimeter_feet"],
        survey_class=req.survey_class or "urban"
    )

    # Build GeoJSON FeatureCollection
    features = []
    
    # Boundary Line / Polygon Feature
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [poly["geojson_coords"]],
        },
        "properties": {
            "name": "Property Boundary",
            "area_acres": parcel_audit["area_acres"],
            "area_sq_feet": parcel_audit["area_sq_feet"],
            "perimeter_feet": poly["total_perimeter_feet"],
            "precision_ratio": poly["precision_ratio"],
            "status": closure_audit["status"],
        }
    })

    # Individual Corner Pin Markers
    for c in poly["corners"]:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [c["lon"], c["lat"]],
            },
            "properties": {
                "index": c["index"],
                "name": c["name"],
                "is_pob": c["is_pob"],
                "incoming_call": c.get("incoming_call"),
                "incoming_distance_feet": c.get("incoming_distance_feet"),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return {
        "success": True,
        "corners": poly["corners"],
        "calls": calls,
        "closure_audit": closure_audit,
        "parcel_audit": parcel_audit,
        "geojson": geojson,
    }


@app.post("/api/survey/nav-vector")
def api_nav_vector(req: NavVectorRequest):
    """
    Computes real-time navigation vector from user GPS to target pin:
    distance (feet/meters), true azimuth, relative turn angle for compass arrow, and proximity flag.
    """
    dist_m, azimuth_deg, _ = vincenty_inverse(
        req.user_lat, req.user_lon,
        req.target_lat, req.target_lon
    )
    dist_ft = dist_m * METERS_TO_FEET
    heading = req.device_heading or 0.0
    turn_angle = relative_bearing(azimuth_deg, heading)
    cardinal = degrees_to_cardinal(azimuth_deg)

    # Proximity classification
    if dist_ft <= 3.0:
        proximity = "PINPOINT"  # Within 3 feet (pin reachable / directly underfoot)
    elif dist_ft <= 15.0:
        proximity = "NEAR"      # Within 15 feet (search radius with pin finder / metal detector)
    elif dist_ft <= 50.0:
        proximity = "APPROACHING"
    else:
        proximity = "FAR"

    return {
        "distance_feet": round(dist_ft, 2),
        "distance_meters": round(dist_m, 2),
        "azimuth_deg": round(azimuth_deg, 1),
        "device_heading": round(heading, 1),
        "turn_angle_deg": round(turn_angle, 1),
        "cardinal_direction": cardinal,
        "proximity_status": proximity,
    }


@app.get("/api/survey/sample-properties")
def api_sample_properties():
    """Provides curated sample properties with real surveyor metes-and-bounds deed descriptions."""
    samples = [
        {
            "id": "austin_5acre",
            "title": "Hill Country Homestead (5.1 Acres)",
            "location_name": "Travis County, TX",
            "pob_lat": 30.2985,
            "pob_lon": -97.8125,
            "pob_monument": "1/2\" Iron Rod Found at West Fence Line",
            "survey_class": "rural",
            "deed_text": (
                "BEGINNING at a 1/2\" iron rod found in the East boundary line of Oak Grove Road;\n"
                "THENCE North 12° 30' 00\" East, a distance of 450.00 feet to an iron rod found;\n"
                "THENCE South 78° 15' 30\" East, a distance of 495.20 feet to a marked post by the creek;\n"
                "THENCE South 15° 45' 00\" West, a distance of 448.50 feet to a 5/8\" rebar set with cap;\n"
                "THENCE North 78° 25' 15\" West, a distance of 469.85 feet to the POINT OF BEGINNING."
            )
        },
        {
            "id": "suburban_lot",
            "title": "Suburban Quarter-Acre Lot (0.34 Acres)",
            "location_name": "Round Rock, TX",
            "pob_lat": 30.5083,
            "pob_lon": -97.6789,
            "pob_monument": "Brass Disk in Concrete at NW Lot Corner",
            "survey_class": "urban",
            "deed_text": (
                "BEGINNING at a brass disk in concrete marking the NW corner of Lot 14, Block B;\n"
                "THENCE North 90° 00' 00\" East, a distance of 100.00 feet to a 1/2\" iron rebar found;\n"
                "THENCE South 00° 00' 00\" East, a distance of 150.00 feet to an iron pipe found;\n"
                "THENCE South 90° 00' 00\" West, a distance of 100.00 feet to a capped rebar set;\n"
                "THENCE North 00° 00' 00\" West, a distance of 150.00 feet to the POINT OF BEGINNING."
            )
        },
        {
            "id": "ten_acre_ranch",
            "title": "10-Acre Rectangular Pasture",
            "location_name": "Bastrop County, TX",
            "pob_lat": 30.1105,
            "pob_lon": -97.3180,
            "pob_monument": "3-inch Iron Pipe at SW Corner",
            "survey_class": "rural",
            "deed_text": (
                "BEGINNING at a 3-inch iron pipe corner post;\n"
                "THENCE North 00° 15' 00\" East, a distance of 660.00 feet to an iron rod found;\n"
                "THENCE South 89° 45' 00\" East, a distance of 660.00 feet to a marked cedar post;\n"
                "THENCE South 00° 15' 00\" West, a distance of 660.00 feet to a 1/2\" rebar set;\n"
                "THENCE North 89° 45' 00\" West, a distance of 660.00 feet to the POINT OF BEGINNING."
            )
        }
    ]
    return {"samples": samples}


@app.post("/api/survey/mark-pin")
def api_mark_pin(req: MarkPinRequest):
    """Saves a verified surveyor pin record to persistent storage."""
    records = []
    if os.path.exists(PINS_STORAGE_PATH):
        try:
            with open(PINS_STORAGE_PATH, "r") as f:
                records = json.load(f)
        except Exception:
            records = []

    record = {
        "id": len(records) + 1,
        "pin_name": req.pin_name,
        "target_lat": req.target_lat,
        "target_lon": req.target_lon,
        "user_lat": req.user_lat,
        "user_lon": req.user_lon,
        "gps_accuracy_feet": req.gps_accuracy_feet,
        "status": req.status,
        "notes": req.notes,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    records.append(record)

    with open(PINS_STORAGE_PATH, "w") as f:
        json.dump(records, f, indent=2)

    return {"success": True, "saved_record": record, "total_pins_marked": len(records)}


@app.get("/api/survey/marked-pins")
def api_get_marked_pins():
    """Retrieves all saved pin markings."""
    if not os.path.exists(PINS_STORAGE_PATH):
        return {"pins": []}
    with open(PINS_STORAGE_PATH, "r") as f:
        records = json.load(f)
    return {"pins": records}


# Mount static files for PWA frontend
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("survey_app.server:app", host="0.0.0.0", port=8000, reload=True)
