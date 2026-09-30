/**
 * PRIME Survey: GPS Property Corner Navigation & COGO App
 * ========================================================
 * High-accuracy field GPS navigation, compass heading tracking,
 * and property boundary visualization powered by WGS-84 COGO.
 */

// Application State
let map = null;
let satelliteLayer = null;
let streetLayer = null;
let boundaryPolygonLayer = null;
let cornerMarkersLayer = null;
let userMarker = null;
let userAccuracyCircle = null;

let currentProperty = null;
let currentCorners = [];
let activeTargetPin = null;

// User GPS & Sensor State
let userLocation = { lat: 30.2982, lon: -97.8129, accuracy_ft: 15.0 };
let deviceHeading = 0.0;
let simulationMode = false;
let simulationStart = null;

// Audio Context for Proximity Radar Chirp
let audioCtx = null;
let lastChirpTime = 0;

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initSensors();
  initEventHandlers();
  loadSampleProperty("austin_5acre");
});

/**
 * 1. Initialize Leaflet Map with ESRI World Imagery
 */
function initMap() {
  map = L.map("survey-map", {
    center: [30.2985, -97.8125],
    zoom: 18,
    zoomControl: false,
  });

  L.control.zoom({ position: "bottomright" }).addTo(map);

  // ESRI World Imagery (High-res Satellite)
  satelliteLayer = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
      maxZoom: 20,
    }
  ).addTo(map);

  // OpenStreetMap / CartoDB Street layer
  streetLayer = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 20,
    }
  );

  boundaryPolygonLayer = L.layerGroup().addTo(map);
  cornerMarkersLayer = L.layerGroup().addTo(map);
}

/**
 * 2. GPS & Compass Sensor Integration
 */
function initSensors() {
  const statusBadge = document.getElementById("gps-status-badge");
  const statusText = document.getElementById("gps-status-text");

  if ("geolocation" in navigator) {
    navigator.geolocation.watchPosition(
      (pos) => {
        if (simulationMode) return;
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        const accFt = pos.coords.accuracy * 3.28084;

        userLocation = { lat, lon, accuracy_ft: accFt };
        updateUserMapMarker(lat, lon, accFt);

        statusBadge.classList.add("locked");
        statusText.textContent = `GPS Locked (±${Math.round(accFt)} ft)`;

        updateNavigationHUD();
      },
      (err) => {
        console.warn("GPS Warning:", err.message);
        if (!simulationMode) {
          statusText.textContent = "GPS Searching / Sim Mode";
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 0,
      }
    );
  }

  // Compass Heading with absolute orientation priority (Android Chrome & iOS Safari)
  const onOrientation = (e) => {
    if (e.webkitCompassHeading) {
      // iOS Safari (degrees relative to magnetic north, clockwise)
      deviceHeading = e.webkitCompassHeading;
    } else if (e.alpha !== null) {
      // Android / Standard W3C DeviceOrientationEvent
      deviceHeading = (360 - e.alpha) % 360;
    }
    updateNavigationHUD();
  };

  if ("ondeviceorientationabsolute" in window) {
    window.addEventListener("deviceorientationabsolute", onOrientation, true);
  } else if (window.DeviceOrientationEvent) {
    window.addEventListener("deviceorientation", onOrientation, true);
  }

  // iOS 13+ permission request on user tap
  document.addEventListener(
    "click",
    () => {
      if (
        typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function"
      ) {
        DeviceOrientationEvent.requestPermission()
          .then((res) => {
            if (res === "granted") {
              window.addEventListener("deviceorientation", onOrientation, true);
            }
          })
          .catch(() => {});
      }
    },
    { once: true }
  );
}


/**
 * 3. Render or Update User GPS Marker & Accuracy Circle
 */
function updateUserMapMarker(lat, lon, accuracyFt) {
  const accuracyMeters = accuracyFt / 3.28084;

  if (!userMarker) {
    const userIcon = L.divIcon({
      className: "user-gps-marker",
      html: '<div class="user-gps-dot"></div>',
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
    userMarker = L.marker([lat, lon], { icon: userIcon, zIndexOffset: 1000 }).addTo(map);
    userAccuracyCircle = L.circle([lat, lon], {
      radius: accuracyMeters,
      color: "#00d2ff",
      fillColor: "#00d2ff",
      fillOpacity: 0.15,
      weight: 1,
    }).addTo(map);
  } else {
    userMarker.setLatLng([lat, lon]);
    userAccuracyCircle.setLatLng([lat, lon]);
    userAccuracyCircle.setRadius(accuracyMeters);
  }
}

/**
 * 4. Load & Render Survey Property
 */
async function loadPropertyDeed(pobLat, pobLon, deedText, pobMonument, surveyClass) {
  try {
    const res = await fetch("/api/survey/parse-deed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pob_lat: pobLat,
        pob_lon: pobLon,
        deed_text: deedText,
        pob_monument: pobMonument,
        survey_class: surveyClass,
      }),
    });

    const data = await res.json();
    if (!data.success) {
      alert("Survey Parse Error: " + (data.detail || "Unable to parse deed"));
      return;
    }

    currentProperty = data;
    currentCorners = data.corners;

    renderPropertyOnMap(data);
    populatePinSelector(data.corners);
    updateSurveyStats(data);

    // Select first target corner
    if (data.corners.length > 1) {
      selectTargetPin(data.corners[1]);
    } else if (data.corners.length > 0) {
      selectTargetPin(data.corners[0]);
    }
  } catch (err) {
    console.error("Failed to parse deed:", err);
    alert("Connection error while parsing deed.");
  }
}

/**
 * 5. Render Boundary Lines & Numbered Corner Pins
 */
function renderPropertyOnMap(data) {
  boundaryPolygonLayer.clearLayers();
  cornerMarkersLayer.clearLayers();

  const corners = data.corners;
  const latLngs = corners.map((c) => [c.lat, c.lon]);

  // Closed Boundary Polygon
  const polygon = L.polygon(latLngs, {
    color: "#00d2ff",
    weight: 3,
    opacity: 0.9,
    fillColor: "#00d2ff",
    fillOpacity: 0.12,
    dashArray: "6, 4",
  }).addTo(boundaryPolygonLayer);

  map.fitBounds(polygon.getBounds(), { padding: [60, 60] });

  // Render Corner Pins
  corners.forEach((c) => {
    const isPob = c.is_pob;
    const pinHtml = isPob
      ? '<div class="pin-marker pob">⭐<span class="pin-tag">POB</span></div>'
      : `<div class="pin-marker corner"><span class="pin-num">${c.index}</span><span class="pin-tag">Corner ${c.index}</span></div>`;

    const customIcon = L.divIcon({
      className: "survey-pin-icon",
      html: pinHtml,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });

    const marker = L.marker([c.lat, c.lon], { icon: customIcon }).addTo(cornerMarkersLayer);

    marker.bindPopup(`
      <div style="font-family: 'Plus Jakarta Sans', sans-serif;">
        <h4 style="margin: 0 0 4px 0; color: #00d2ff;">${c.name}</h4>
        <p style="margin: 0; font-size: 12px; color: #555;">
          ${c.incoming_call ? `<b>Call:</b> ${c.incoming_call} (${c.incoming_distance_feet} ft)<br>` : ""}
          <b>GPS:</b> ${c.lat.toFixed(6)}°, ${c.lon.toFixed(6)}°
        </p>
        <button onclick="window.selectPinByIndex(${c.index})" style="margin-top: 8px; background: #00d2ff; color:#fff; border:none; padding:4px 8px; border-radius:4px; cursor:pointer; font-weight:700;">🎯 Navigate to Pin</button>
      </div>
    `);

    marker.on("click", () => selectTargetPin(c));
  });
}

window.selectPinByIndex = (idx) => {
  const target = currentCorners.find((c) => c.index === idx);
  if (target) selectTargetPin(target);
};

/**
 * 6. Pin Selector Dropdown
 */
function populatePinSelector(corners) {
  const select = document.getElementById("select-target-pin");
  select.innerHTML = "";

  corners.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.index;
    opt.textContent = `${c.name} ${c.incoming_distance_feet ? `(${c.incoming_distance_feet} ft leg)` : ""}`;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => {
    const idx = parseInt(e.target.value);
    const pin = currentCorners.find((c) => c.index === idx);
    if (pin) selectTargetPin(pin);
  });
}

function selectTargetPin(pin) {
  activeTargetPin = pin;
  document.getElementById("select-target-pin").value = pin.index;
  document.getElementById("target-index-badge").textContent = pin.is_pob ? "POB" : `CORNER ${pin.index}`;
  document.getElementById("target-coords-display").textContent = 
    `Target: ${pin.lat.toFixed(6)}° N, ${pin.lon.toFixed(6)}° W • ${pin.name}`;

  // Reset simulator if active
  if (simulationMode) {
    simulationStart = { ...userLocation };
    document.getElementById("sim-slider").value = 0;
  }

  updateNavigationHUD();
}

/**
 * 7. Client-Side Geodetic Navigation Vector (WGS-84 Vincenty Inverse)
 * Enables 0ms latency 60fps needle tracking and complete offline autonomy in the field.
 */
function calculateLocalNavVector(userLat, userLon, targetLat, targetLon, headingDeg) {
  const a = 6378137.0;
  const f = 1.0 / 298.257223563;
  const b = 6356752.314245;

  if (Math.abs(userLat - targetLat) < 1e-9 && Math.abs(userLon - targetLon) < 1e-9) {
    return { distM: 0.0, distFt: 0.0, azimuthDeg: 0.0, turnAngleDeg: 0.0, cardinal: "N" };
  }

  const phi1 = (userLat * Math.PI) / 180.0;
  const L1 = (userLon * Math.PI) / 180.0;
  const phi2 = (targetLat * Math.PI) / 180.0;
  const L2 = (targetLon * Math.PI) / 180.0;
  const deltaL = L2 - L1;

  const tanU1 = (1.0 - f) * Math.tan(phi1);
  const cosU1 = 1.0 / Math.sqrt(1.0 + tanU1 * tanU1);
  const sinU1 = tanU1 * cosU1;

  const tanU2 = (1.0 - f) * Math.tan(phi2);
  const cosU2 = 1.0 / Math.sqrt(1.0 + tanU2 * tanU2);
  const sinU2 = tanU2 * cosU2;

  let lam = deltaL;
  let lamPrev = 2.0 * Math.PI;
  let sinSigma = 0.0, cosSigma = 0.0, sigma = 0.0;
  let sinAlpha = 0.0, cos2Alpha = 0.0, cos2SigmaM = 0.0;

  for (let i = 0; i < 50; i++) {
    const sinLam = Math.sin(lam);
    const cosLam = Math.cos(lam);
    sinSigma = Math.sqrt(
      (cosU2 * sinLam) ** 2 +
      (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
    );
    if (sinSigma === 0) break;
    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam;
    sigma = Math.atan2(sinSigma, cosSigma);
    sinAlpha = (cosU1 * cosU2 * sinLam) / sinSigma;
    cos2Alpha = 1.0 - sinAlpha * sinAlpha;
    cos2SigmaM = cos2Alpha !== 0 ? cosSigma - (2.0 * sinU1 * sinU2) / cos2Alpha : 0.0;
    const C = (f / 16.0) * cos2Alpha * (4.0 + f * (4.0 - 3.0 * cos2Alpha));
    lamPrev = lam;
    lam = deltaL + (1.0 - C) * f * sinAlpha * (
      sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM))
    );
    if (Math.abs(lam - lamPrev) < 1e-12) break;
  }

  const u2 = cos2Alpha * (a * a - b * b) / (b * b);
  const A = 1.0 + (u2 / 16384.0) * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
  const B = (u2 / 1024.0) * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
  const deltaSigma = B * sinSigma * (
    cos2SigmaM + (B / 4.0) * (
      cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM) -
      (B / 6.0) * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) * (-3.0 + 4.0 * cos2SigmaM * cos2SigmaM)
    )
  );

  const distM = b * A * (sigma - deltaSigma);
  const distFt = distM * 3.280839895013123;

  const alpha1 = Math.atan2(
    cosU2 * Math.sin(lam),
    cosU1 * sinU2 - sinU1 * cosU2 * Math.cos(lam)
  );
  const azimuthDeg = ((alpha1 * 180.0) / Math.PI + 360.0) % 360.0;

  let turnAngle = (azimuthDeg - (headingDeg || 0.0)) % 360.0;
  if (turnAngle > 180.0) turnAngle -= 360.0;
  if (turnAngle < -180.0) turnAngle += 360.0;

  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const cardinal = dirs[Math.floor(((azimuthDeg + 11.25) % 360.0) / 22.5)];

  return { distM, distFt, azimuthDeg, turnAngleDeg: turnAngle, cardinal };
}

/**
 * 8. Live Real-Time Navigation HUD Updates
 */
function updateNavigationHUD() {
  if (!activeTargetPin) return;

  // Immediate synchronous calculation on client for zero lag
  const nav = calculateLocalNavVector(
    userLocation.lat,
    userLocation.lon,
    activeTargetPin.lat,
    activeTargetPin.lon,
    deviceHeading
  );

  const distFt = nav.distFt;
  const distM = nav.distM;

  document.getElementById("dist-feet-val").textContent = distFt < 10 ? distFt.toFixed(1) : Math.round(distFt);
  document.getElementById("dist-meters-val").textContent = `${distM.toFixed(1)} m`;
  document.getElementById("bearing-val").textContent = `${Math.round(nav.azimuthDeg)}°`;
  document.getElementById("cardinal-val").textContent = nav.cardinal;

  // Rotate compass arrow smoothly
  const navArrow = document.getElementById("nav-arrow");
  navArrow.style.transform = `rotate(${nav.turnAngleDeg}deg)`;

  // Proximity Status Badge & Proximity Audio Radar
  const badge = document.getElementById("proximity-status-badge");
  badge.className = "proximity-badge";

  if (distFt <= 3.0) {
    badge.classList.add("badge-pinpoint");
    badge.textContent = "📍 PINPOINT (< 3 FT) — DIRECTLY UNDERFOOT";
    triggerProximityChirp(1200, 0.15); // Fast high-pitch ping
  } else if (distFt <= 15.0) {
    badge.classList.add("badge-near");
    badge.textContent = "🎯 NEAR PIN (< 15 FT) — SWEEP METAL DETECTOR";
    triggerProximityChirp(800, 0.4);
  } else if (distFt <= 50.0) {
    badge.classList.add("badge-far");
    badge.textContent = "APPROACHING PIN (< 50 FT)";
  } else {
    badge.classList.add("badge-far");
    badge.textContent = "FOLLOW ARROW TO CORNER PIN";
  }
}


/**
 * 8. Audio Proximity Radar Chirp (Web Audio API)
 */
function triggerProximityChirp(freq, intervalSec) {
  const now = Date.now() / 1000.0;
  if (now - lastChirpTime < intervalSec) return;
  lastChirpTime = now;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + 0.08);

    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.08);

    // Haptic vibration on mobile
    if ("vibrate" in navigator) {
      navigator.vibrate(30);
    }
  } catch (e) {
    // Audio context may require initial user click on mobile
  }
}

/**
 * 9. Update Survey Stats & ALTA Legal Compliance
 */
function updateSurveyStats(data) {
  const parcel = data.parcel_audit;
  const closure = data.closure_audit;

  document.getElementById("stat-acres").textContent = `${parcel.area_acres.toFixed(2)} Acres`;
  document.getElementById("stat-sqft").textContent = `${Math.round(parcel.area_sq_feet).toLocaleString()} sq ft`;
  document.getElementById("stat-perimeter").textContent = `${Math.round(parcel.perimeter_feet).toLocaleString()} ft`;
  document.getElementById("stat-misclosure").textContent = `${closure.misclosure_feet.toFixed(3)} ft`;
  document.getElementById("stat-misclosure-in").textContent = `${closure.misclosure_inches} in`;
  document.getElementById("stat-precision").textContent = closure.precision_ratio;

  const legalBadge = document.getElementById("closure-legal-badge");
  if (closure.alta_nsps_compliant) {
    legalBadge.className = "badge-legal pass";
    legalBadge.textContent = "ALTA COMPLIANT";
  } else {
    legalBadge.className = "badge-legal warn";
    legalBadge.textContent = "MISCLOSURE WARNING";
  }
}

/**
 * 10. Load Sample Properties
 */
async function loadSampleProperty(sampleId = "austin_5acre") {
  try {
    const res = await fetch("/api/survey/sample-properties");
    const data = await res.json();
    const sample = data.samples.find((s) => s.id === sampleId) || data.samples[0];

    // Seed initial user location nearby for testing
    userLocation = {
      lat: sample.pob_lat - 0.0003,
      lon: sample.pob_lon - 0.0004,
      accuracy_ft: 12.0,
    };
    simulationStart = { ...userLocation };
    updateUserMapMarker(userLocation.lat, userLocation.lon, userLocation.accuracy_ft);

    // Populate modal inputs for reference
    document.getElementById("input-pob-lat").value = sample.pob_lat;
    document.getElementById("input-pob-lon").value = sample.pob_lon;
    document.getElementById("input-pob-monument").value = sample.pob_monument;
    document.getElementById("input-survey-class").value = sample.survey_class;
    document.getElementById("input-deed-text").value = sample.deed_text;

    await loadPropertyDeed(
      sample.pob_lat,
      sample.pob_lon,
      sample.deed_text,
      sample.pob_monument,
      sample.survey_class
    );
  } catch (err) {
    console.error("Failed to load sample property:", err);
  }
}

/**
 * 11. Event Handlers & Modal
 */
function initEventHandlers() {
  // Layer switching
  document.getElementById("btn-layer-sat").addEventListener("click", () => {
    map.removeLayer(streetLayer);
    map.addLayer(satelliteLayer);
    document.getElementById("btn-layer-sat").classList.add("active");
    document.getElementById("btn-layer-topo").classList.remove("active");
  });

  document.getElementById("btn-layer-topo").addEventListener("click", () => {
    map.removeLayer(satelliteLayer);
    map.addLayer(streetLayer);
    document.getElementById("btn-layer-topo").classList.add("active");
    document.getElementById("btn-layer-sat").classList.remove("active");
  });

  // Fit bounds button
  document.getElementById("btn-fit-bounds").addEventListener("click", () => {
    if (boundaryPolygonLayer) {
      const layers = boundaryPolygonLayer.getLayers();
      if (layers.length > 0) map.fitBounds(layers[0].getBounds(), { padding: [60, 60] });
    }
  });

  // GPS Simulation Toggle (Desktop testing)
  const simCard = document.getElementById("sim-controls-card");
  const btnSim = document.getElementById("btn-toggle-sim");
  btnSim.addEventListener("click", () => {
    simulationMode = !simulationMode;
    btnSim.classList.toggle("active", simulationMode);
    simCard.classList.toggle("hidden", !simulationMode);
    if (simulationMode) {
      simulationStart = { ...userLocation };
      document.getElementById("sim-slider").value = 0;
      document.getElementById("gps-status-badge").classList.add("locked");
      document.getElementById("gps-status-text").textContent = "GPS Simulator Active";
    }
  });

  // Simulator Slider
  document.getElementById("sim-slider").addEventListener("input", (e) => {
    if (!simulationMode || !activeTargetPin || !simulationStart) return;
    const progress = parseFloat(e.target.value) / 100.0;
    const simLat = simulationStart.lat + (activeTargetPin.lat - simulationStart.lat) * progress;
    const simLon = simulationStart.lon + (activeTargetPin.lon - simulationStart.lon) * progress;

    userLocation.lat = simLat;
    userLocation.lon = simLon;
    updateUserMapMarker(simLat, simLon, 5.0);
    updateNavigationHUD();
  });

  // Modal open/close
  const modal = document.getElementById("deed-modal");
  document.getElementById("btn-open-deed").addEventListener("click", () => modal.classList.remove("hidden"));
  document.getElementById("btn-close-modal").addEventListener("click", () => modal.classList.add("hidden"));

  // Sample load dropdown
  document.getElementById("btn-load-sample").addEventListener("click", () => {
    const choice = prompt("Select Sample Property:\n1. Hill Country 5-Acre Homestead\n2. Suburban Lot (Quarter-Acre)\n3. 10-Acre Pasture", "1");
    if (choice === "1") loadSampleProperty("austin_5acre");
    else if (choice === "2") loadSampleProperty("suburban_lot");
    else if (choice === "3") loadSampleProperty("ten_acre_ranch");
  });

  // Modal fill sample button
  document.getElementById("btn-load-sample-homestead").addEventListener("click", () => {
    loadSampleProperty("austin_5acre");
  });

  // Submit deed form
  document.getElementById("btn-submit-deed").addEventListener("click", async () => {
    const pobLat = parseFloat(document.getElementById("input-pob-lat").value);
    const pobLon = parseFloat(document.getElementById("input-pob-lon").value);
    const pobMon = document.getElementById("input-pob-monument").value;
    const surveyClass = document.getElementById("input-survey-class").value;
    const deedText = document.getElementById("input-deed-text").value;

    if (!deedText.trim()) {
      alert("Please enter surveyor deed text.");
      return;
    }

    modal.classList.add("hidden");
    await loadPropertyDeed(pobLat, pobLon, deedText, pobMon, surveyClass);
  });

  // Mark pin button actions
  const statusButtons = document.querySelectorAll(".btn-status");
  let selectedStatus = "FOUND_ROD";

  statusButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      statusButtons.forEach((b) => (b.style.borderColor = "var(--border)"));
      btn.style.borderColor = "var(--accent-cyan)";
      selectedStatus = btn.dataset.status;
    });
  });

  document.getElementById("btn-save-pin-log").addEventListener("click", async () => {
    if (!activeTargetPin) return;
    const notes = document.getElementById("pin-field-notes").value;

    try {
      const res = await fetch("/api/survey/mark-pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pin_name: activeTargetPin.name,
          target_lat: activeTargetPin.lat,
          target_lon: activeTargetPin.lon,
          user_lat: userLocation.lat,
          user_lon: userLocation.lon,
          gps_accuracy_feet: userLocation.accuracy_ft || 10.0,
          status: selectedStatus,
          notes: notes,
        }),
      });
      const data = await res.json();
      if (data.success) {
        alert(`✅ Pin Marked: "${activeTargetPin.name}" logged as ${selectedStatus}. Total marked: ${data.total_pins_marked}`);
        document.getElementById("pin-field-notes").value = "";
      }
    } catch (e) {
      alert("Failed to log pin inspection.");
    }
  });
}
