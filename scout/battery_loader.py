"""
PRIME-Scout: Real-World NASA Battery Telemetry Loader
Loads and preprocesses real empirical measurements from the NASA Ames
Prognostics Center of Excellence (PCoE) Li-ion Battery Dataset (Cell B0005).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from scout.config import BASE_DIR

BATTERY_DATA_DIR = BASE_DIR / "scout" / "data" / "battery"
B0005_CSV_PATH = BATTERY_DATA_DIR / "B0005.csv"

class NasaBatteryLoader:
    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or B0005_CSV_PATH
        self._df: Optional[pd.DataFrame] = None

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"NASA Battery dataset not found at {self.csv_path}")
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def get_capacity_fade_series(self) -> Dict[str, Any]:
        """
        Extracts real-world capacity fade trajectory across 168 discharge cycles.
        Returns:
        - cycles: list of cycle indices [1..168]
        - capacity_ah: measured capacity in Ah [1.856..1.288]
        - capacity_loss_ah: cumulative capacity loss Q_loss = Q_0 - Q
        - soh_ratio: State of Health ratio (Q / Q_0)
        - nominal_capacity_ah: Q_0
        """
        df = self._ensure_loaded()
        cycle_group = df.groupby("cycle")["capacity"].first().reset_index()
        cycles = cycle_group["cycle"].tolist()
        caps = [round(float(c), 4) for c in cycle_group["capacity"].tolist()]
        q0 = caps[0] if caps else 1.8565
        q_loss = [round(float(q0 - c), 4) for c in caps]
        soh = [round(float(c / q0), 4) for c in caps]

        return {
            "cycles": cycles,
            "capacity_ah": caps,
            "capacity_loss_ah": q_loss,
            "soh_ratio": soh,
            "nominal_capacity_ah": q0,
            "total_cycles": len(cycles)
        }

    def get_internal_resistance_series(self) -> Dict[str, Any]:
        """
        Estimates dynamic internal resistance R_0 = |Delta V / Delta I| per discharge cycle.
        Returns:
        - cycles: list of cycle indices
        - r0_ohms: internal resistance in Ohms
        - temp_c: ambient/initial temperature in Celsius
        """
        df = self._ensure_loaded()
        cycles = []
        r0_list = []
        temp_list = []

        for c_idx, group in df.groupby("cycle"):
            if len(group) < 5:
                continue
            v_init = group["voltage_measured"].iloc[0]
            v_loaded = group["voltage_measured"].iloc[min(3, len(group)-1)]
            i_init = group["current_measured"].iloc[0]
            i_loaded = group["current_measured"].iloc[min(3, len(group)-1)]
            temp = group["temperature_measured"].iloc[0]

            delta_i = abs(i_loaded - i_init)
            delta_v = abs(v_init - v_loaded)

            if delta_i > 0.1:
                r0 = delta_v / delta_i
                cycles.append(int(c_idx))
                r0_list.append(round(float(r0), 4))
                temp_list.append(round(float(temp), 2))

        return {
            "cycles": cycles,
            "r0_ohms": r0_list,
            "temperatures_c": temp_list,
            "mean_r0": round(float(np.mean(r0_list)), 4) if r0_list else 0.0
        }

    def get_thermal_rise_series(self) -> Dict[str, Any]:
        """
        Measures maximum cell temperature rise Delta T = T_max - T_ambient per cycle.
        """
        df = self._ensure_loaded()
        cycles = []
        delta_t_list = []

        for c_idx, group in df.groupby("cycle"):
            t_amb = group["ambient_temperature"].iloc[0]
            t_max = group["temperature_measured"].max()
            delta_t = t_max - t_amb
            cycles.append(int(c_idx))
            delta_t_list.append(round(float(delta_t), 2))

        return {
            "cycles": cycles,
            "delta_t_c": delta_t_list,
            "max_rise": round(float(max(delta_t_list)), 2) if delta_t_list else 0.0
        }

    def get_discharge_curve(self, cycle_num: int = 1) -> Dict[str, Any]:
        """
        Returns high-frequency discharge curve (V, I, T vs time) for a specific cycle.
        """
        df = self._ensure_loaded()
        sub = df[df["cycle"] == cycle_num].sort_values("time")
        if sub.empty:
            sub = df[df["cycle"] == 1].sort_values("time")

        return {
            "cycle": cycle_num,
            "time_sec": sub["time"].tolist(),
            "voltage": [round(float(v), 4) for v in sub["voltage_measured"].tolist()],
            "current": [round(float(i), 4) for i in sub["current_measured"].tolist()],
            "temperature": [round(float(t), 2) for t in sub["temperature_measured"].tolist()],
            "capacity": float(sub["capacity"].iloc[0])
        }

if __name__ == "__main__":
    loader = NasaBatteryLoader()
    fade = loader.get_capacity_fade_series()
    print(f"Loaded NASA B0005: {fade['total_cycles']} cycles. Initial Q={fade['capacity_ah'][0]} Ah, Final Q={fade['capacity_ah'][-1]} Ah")
    r0 = loader.get_internal_resistance_series()
    print(f"Internal resistance evaluated across {len(r0['cycles'])} cycles. Mean R0 = {r0['mean_r0']} Ohms")
    therm = loader.get_thermal_rise_series()
    print(f"Thermal generation evaluated across {len(therm['cycles'])} cycles. Max Delta T = {therm['max_rise']} C")
