"""
HELIOS-AI: A Model-Based Health & Wellness Advisor Agent
--------------------------------------------------------
PEAS:
- Performance (P): Maximize Wellness Score (0–100), reduce risk indices (Hydration/Fatigue/Stress),
  and increase adherence (tracked via daily plan checkboxes).
- Environment (E): Daily user metrics—sleep hours, resting HR, steps/active minutes, water intake (ml),
  self-reported stress (1–5), caffeine flag—plus historical records.
- Actuators (A): On-screen recommendations (ranked with rationale), alerts, daily plan checklist,
  score visualization, and saved records.
- Sensors (S): GUI inputs, optional CSV history import; GUI acts as perception channel.

Notes:
- Educational tool; not medical advice.
- Local-only persistence by default: /data CSV & JSON in script directory.

Run:
    python3 helios_ai.py
Python: 3.9+
"""

from __future__ import annotations
import json
import os
import math
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

import tkinter as tk
from tkinter import ttk, messagebox

import platform
# Suppress macOS system Tk warning; also set our fallback
if platform.system() == "Darwin":
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

# Force legacy tk widgets if ttk cannot render on this macOS
FORCE_LEGACY_TK = True  # set True for your current env; flip to False on a good Tk

# Matplotlib for trend chart embedding
try:
    import matplotlib
    matplotlib.use("Agg")  # draw off-screen, then render to Tk canvas as image
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

try:
    from io import BytesIO
    from PIL import Image, ImageTk  # pillow is commonly available; used to render chart
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "wellness_history.csv")
STATE_PATH = os.path.join(DATA_DIR, "model_state.json")


# ---------------------
# Utilities
# ---------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ewma_update(prev: float, x: float, alpha: float = 0.2) -> float:
    return alpha * x + (1 - alpha) * prev

def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default


# ---------------------
# Belief State (Model)
# ---------------------
@dataclass
class RollingBaselines:
    sleep_h: float = 7.0
    resting_hr: float = 70.0
    steps: float = 6000.0
    water_ml: float = 1800.0

@dataclass
class ModelState:
    baselines: RollingBaselines
    last_scores: List[float]  # trailing list of wellness scores for chart

    @staticmethod
    def default():
        return ModelState(baselines=RollingBaselines(), last_scores=[])

    def to_json(self):
        return {
            "baselines": asdict(self.baselines),
            "last_scores": self.last_scores[-30:],  # keep tail
        }

    @staticmethod
    def from_json(obj):
        bl = obj.get("baselines", {})
        return ModelState(
            baselines=RollingBaselines(
                sleep_h=bl.get("sleep_h", 7.0),
                resting_hr=bl.get("resting_hr", 70.0),
                steps=bl.get("steps", 6000.0),
                water_ml=bl.get("water_ml", 1800.0),
            ),
            last_scores=list(obj.get("last_scores", []))[:30],
        )


class BeliefEngine:
    """
    Maintains internal model: rolling baselines (EWMA), risk indices,
    and computes a composite Wellness Score.
    """
    def __init__(self, state: Optional[ModelState] = None):
        self.state = state or ModelState.default()
        self.log: List[str] = []

    def _log(self, msg: str):
        # Keep a short trace for GUI
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {msg}")
        if len(self.log) > 200:
            self.log.pop(0)

    def compute(self, sleep_h: float, resting_hr: float, steps: float, water_ml: float,
                stress_1to5: int, caffeine: bool) -> Dict[str, float]:
        bl = self.state.baselines

        # 1) Normalize against baselines (simple bounded scaling)
        # Sleep: ideal ~7-9h; penalty if <7 or >9 relative to baseline tilt
        sleep_norm = clamp((sleep_h / max(1e-6, bl.sleep_h)), 0.0, 1.5)
        # HR: lower is better relative to baseline; >baseline penalizes
        hr_norm = clamp(bl.resting_hr / max(1.0, resting_hr), 0.5, 1.5)
        # Steps: more than baseline is good up to ~1.5x
        steps_norm = clamp((steps / max(1.0, bl.steps)), 0.3, 1.6)
        # Water: target baseline water_ml; max benefit around 1.2x baseline
        water_norm = clamp((water_ml / max(1.0, bl.water_ml)), 0.3, 1.3)
        # Stress: lower is better; invert scale (1 best, 5 worst)
        stress_norm = clamp((6 - stress_1to5) / 5.0, 0.0, 1.0)

        # 2) Risk indices (simple interpretable rules)
        hydration_risk = clamp(1.0 - water_norm, 0.0, 1.0)
        fatigue_risk = clamp((1.0 - sleep_norm) + (1.0 - steps_norm) * 0.5 + (1.0 - hr_norm) * 0.5, 0.0, 1.5)
        stress_risk = clamp((1.0 - stress_norm) + (1.0 - steps_norm) * 0.2, 0.0, 1.5)
        if caffeine:
            hydration_risk = clamp(hydration_risk + 0.1, 0.0, 1.5)
            stress_risk = clamp(stress_risk + 0.1, 0.0, 1.5)

        # 3) Wellness score (weighted)
        # Start from 100 and subtract normalized penalties
        penalty = (
            (1.0 - sleep_norm) * 22 +
            (1.0 - hr_norm)   * 18 +
            (1.0 - steps_norm)* 24 +
            (1.0 - water_norm)* 16 +
            (1.0 - stress_norm)* 20
        )
        score = clamp(100.0 - max(0.0, penalty), 0.0, 100.0)

        self._log(f"Inputs: sleep={sleep_h:.1f}h, HR={resting_hr:.0f}, steps={steps:.0f}, water={water_ml:.0f}ml, stress={stress_1to5}, caffeine={caffeine}")
        self._log(f"Norms: sleep={sleep_norm:.2f}, hr={hr_norm:.2f}, steps={steps_norm:.2f}, water={water_norm:.2f}, stress={stress_norm:.2f}")
        self._log(f"Risks: hydration={hydration_risk:.2f}, fatigue={fatigue_risk:.2f}, stress={stress_risk:.2f}")
        self._log(f"Score: {score:.1f}")

        return {
            "score": score,
            "sleep_norm": sleep_norm,
            "hr_norm": hr_norm,
            "steps_norm": steps_norm,
            "water_norm": water_norm,
            "stress_norm": stress_norm,
            "hydration_risk": hydration_risk,
            "fatigue_risk": fatigue_risk,
            "stress_risk": stress_risk,
        }

    def plan_actions(self, metrics: Dict[str, float], inputs: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Returns list of (recommendation, rationale, expected_delta_score).
        Simple heuristics tied to risk indices and norms.
        """
        recs = []

        # Hydration
        if metrics["hydration_risk"] > 0.25:
            deficit_ratio = clamp(1.0 - metrics["water_norm"], 0.0, 1.0)
            add_ml = int(500 + 1500 * deficit_ratio)
            rationale = f"Hydration risk is elevated; water intake below baseline. Add ~{add_ml} ml today."
            recs.append((f"Drink +{add_ml} ml water", rationale, 4.5))

        # Fatigue: low sleep or low steps or higher HR vs baseline
        if metrics["fatigue_risk"] > 0.4:
            if inputs["sleep_h"] < 7:
                recs.append(("Aim for 7–8h sleep tonight", "Sleep below optimal range; fatigue risk detected.", 6.0))
            if inputs["steps"] < max(4000, self.state.baselines.steps * 0.8):
                recs.append(("Take a 20–25 min brisk walk", "Activity below baseline; movement can improve energy and stress.", 5.0))

        # Stress
        if metrics["stress_risk"] > 0.35 or inputs["stress_1to5"] >= 4:
            recs.append(("Do a 5–7 min breathing session", "Stress elevated; short breathing helps acute relief.", 4.0))
            recs.append(("Reduce screens 30 min before bed", "Sleep hygiene improves recovery & stress.", 3.0))

        # Cap to top-3 by expected impact
        recs.sort(key=lambda x: x[2], reverse=True)
        recs = recs[:3]

        if not recs:
            recs = [("Maintain routine", "Metrics near baseline—keep steady habits today.", 1.0)]

        # Deduplicate by text
        seen = set()
        final = []
        for r in recs:
            if r[0] not in seen:
                final.append(r)
                seen.add(r[0])
        return final

    def update_baselines(self, sleep_h: float, resting_hr: float, steps: float, water_ml: float, score: float):
        bl = self.state.baselines
        bl.sleep_h    = ewma_update(bl.sleep_h, sleep_h, 0.2)
        bl.resting_hr = ewma_update(bl.resting_hr, resting_hr, 0.2)
        bl.steps      = ewma_update(bl.steps, steps, 0.2)
        bl.water_ml   = ewma_update(bl.water_ml, water_ml, 0.2)

        self.state.last_scores.append(score)
        self.state.last_scores = self.state.last_scores[-30:]

        self._log(f"Baselines→ sleep={bl.sleep_h:.2f}h, HR={bl.resting_hr:.1f}, steps={bl.steps:.0f}, water={bl.water_ml:.0f}ml")

    def save_state(self):
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.state.to_json(), f, indent=2)

    def load_state(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.state = ModelState.from_json(obj)
            self._log("Model state loaded.")
        else:
            self._log("No existing model_state.json; using defaults.")


# ---------------------
# Persistence (CSV)
# ---------------------
CSV_HEADER = [
    "date", "sleep_h", "resting_hr", "steps", "water_ml", "stress_1to5", "caffeine",
    "score", "recs"
]

def append_csv(row: Dict[str, str]):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------
# GUI
# ---------------------
class SimpleTable:
    """
    Minimal replacement for ttk.Treeview used in Recommendations.
    Exposes methods: get_children, delete, insert, item, and a simple render.
    """
    def __init__(self, parent, columns, headings):
        self.frame = tk.Frame(parent)
        self.frame.pack(fill="both", expand=True)
        self.columns = columns
        self.headings = headings
        self.rows = []  # list of tuples
        # header
        header = tk.Frame(self.frame)
        header.pack(fill="x")
        for i, text in enumerate(headings):
            lab = tk.Label(header, text=text, font=("Helvetica", 10, "bold"), anchor="w")
            lab.grid(row=0, column=i, sticky="we", padx=4, pady=2)
            header.grid_columnconfigure(i, weight=1)
        # body
        self.body = tk.Frame(self.frame)
        self.body.pack(fill="both", expand=True)

    def _render(self):
        for w in self.body.winfo_children():
            w.destroy()
        for r, row in enumerate(self.rows):
            for c, val in enumerate(row):
                lab = tk.Label(self.body, text=str(val), anchor="w", wraplength=320 if c == 1 else 200, justify="left")
                lab.grid(row=r, column=c, sticky="we", padx=4, pady=2)
            self.body.grid_columnconfigure(0, weight=1)
            self.body.grid_columnconfigure(1, weight=2)
            self.body.grid_columnconfigure(2, weight=1)

    # Treeview-like API
    def get_children(self):
        return list(range(len(self.rows)))

    def delete(self, idx):
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        if 0 <= idx < len(self.rows):
            self.rows.pop(idx)
        self._render()

    def insert(self, _parent, _end, values):
        self.rows.append(tuple(values))
        self._render()
        return str(len(self.rows) - 1)

    def item(self, idx, option):
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        if option == "values" and 0 <= idx < len(self.rows):
            return self.rows[idx]
        return None


class HeliosGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HELIOS-AI — Model-Based Health & Wellness Advisor")

        self.engine = BeliefEngine()
        self.engine.load_state()

        # ----- widget helpers for legacy tk fallback -----
    def _W(self):
        # Return the widget module to use (tk in legacy mode, ttk otherwise)
        return tk if FORCE_LEGACY_TK else ttk

    def _label(self, parent, **kw):
        return (tk.Label if FORCE_LEGACY_TK else ttk.Label)(parent, **kw)

    def _frame(self, parent, **kw):
        return (tk.Frame if FORCE_LEGACY_TK else ttk.Frame)(parent, **kw)

    def _labelframe(self, parent, **kw):
        return (tk.LabelFrame if FORCE_LEGACY_TK else ttk.LabelFrame)(parent, **kw)

    def _entry(self, parent, **kw):
        return (tk.Entry if FORCE_LEGACY_TK else ttk.Entry)(parent, **kw)

    def _button(self, parent, **kw):
        return (tk.Button if FORCE_LEGACY_TK else ttk.Button)(parent, **kw)

    def _checkbutton(self, parent, **kw):
        return (tk.Checkbutton if FORCE_LEGACY_TK else ttk.Checkbutton)(parent, **kw)

    def _build_layout(self):
        self.root.geometry("1100x650")

        self.main = self._frame(self.root, padx=10, pady=10)
        self.main.pack(fill="both", expand=True)

        self.left = self._frame(self.main)
        self.center = self._frame(self.main)
        self.right = self._frame(self.main)

        self.left.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.center.grid(row=0, column=1, sticky="nsew", padx=(0,10))
        self.right.grid(row=0, column=2, sticky="nsew")

        for i in range(3):
            self.main.columnconfigure(i, weight=1 if i == 0 else 2)
        self.main.rowconfigure(0, weight=1)

        # Left: Inputs
        inputs = self._labelframe(self.left, text="Today's Inputs")
        inputs.pack(fill="x", pady=4)

        self.var_sleep = tk.StringVar(value="7.0")
        self.var_hr = tk.StringVar(value="70")
        self.var_steps = tk.StringVar(value="6000")
        self.var_water = tk.StringVar(value="1800")
        self.var_stress = tk.StringVar(value="3")
        self.var_caffeine = tk.BooleanVar(value=False)

        def add_row(parent, label, var, unit=""):
            frame = self._frame(parent)
            frame.pack(fill="x", pady=2)
            self._label(frame, text=label, width=16).pack(side="left")
            self._entry(frame, textvariable=var, width=12).pack(side="left")
            if unit:
                self._label(frame, text=unit).pack(side="left")

        add_row(inputs, "Sleep Hours", self.var_sleep, "h")
        add_row(inputs, "Resting HR", self.var_hr, "bpm")
        add_row(inputs, "Steps", self.var_steps, "count")
        add_row(inputs, "Water Intake", self.var_water, "ml")

        frame_stress = self._frame(inputs)
        frame_stress.pack(fill="x", pady=2)
        self._label(frame_stress, text="Stress (1–5)", width=16).pack(side="left")
        if FORCE_LEGACY_TK:
            # use OptionMenu in legacy mode
            opt = tk.OptionMenu(frame_stress, self.var_stress, "1","2","3","4","5")
            opt.config(width=8)
            opt.pack(side="left")
        else:
            ttk.Combobox(frame_stress, textvariable=self.var_stress, values=["1","2","3","4","5"], width=10, state="readonly").pack(side="left")

        frame_caf = self._frame(inputs)
        frame_caf.pack(fill="x", pady=2)
        self._checkbutton(frame_caf, text="Caffeine today", variable=self.var_caffeine).pack(side="left")

        actions = self._frame(self.left)
        actions.pack(fill="x", pady=8)
        self._button(actions, text="Compute Wellness", command=self.on_compute).pack(fill="x", pady=2)
        self._button(actions, text="Save Day", command=self.on_save).pack(fill="x", pady=2)
        self._button(actions, text="Reset Baselines", command=self.on_reset_baselines).pack(fill="x", pady=2)
        self._button(actions, text="Export CSV", command=self.on_export).pack(fill="x", pady=2)

        # Center: Score + Trend + Log
        top_center = self._frame(self.center)
        top_center.pack(fill="x")
        self.score_var = tk.StringVar(value="—")
        self._label(top_center, text="Wellness Score", font=("Helvetica", 16, "bold")).pack(side="left")
        self._label(top_center, textvariable=self.score_var, font=("Helvetica", 20)).pack(side="left", padx=10)

        chart_frame = self._labelframe(self.center, text="Last 7 Days Trend")
        chart_frame.pack(fill="both", expand=True, pady=6)
        self.chart_label = self._label(chart_frame)
        self.chart_label.pack(fill="both", expand=True)

        log_frame = self._labelframe(self.center, text="Agent Reasoning Trace")
        log_frame.pack(fill="both", expand=True, pady=6)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Right: Recommendations & Plan
        recs_frame = self._labelframe(self.right, text="Today's Recommendations")
        recs_frame.pack(fill="both", expand=True, pady=6)
        if FORCE_LEGACY_TK:
            self.recs_tree = SimpleTable(recs_frame, columns=("rec","why","impact"), headings=("Recommendation","Rationale","Expected ΔScore"))
        else:
            self.recs_tree = ttk.Treeview(recs_frame, columns=("rec", "why", "impact"), show="headings", height=6)
            self.recs_tree.heading("rec", text="Recommendation")
            self.recs_tree.heading("why", text="Rationale")
            self.recs_tree.heading("impact", text="Expected ΔScore")
            self.recs_tree.column("rec", width=180, anchor="w")
            self.recs_tree.column("why", width=320, anchor="w")
            self.recs_tree.column("impact", width=120, anchor="center")
            self.recs_tree.pack(fill="both", expand=True)

        plan_frame = self._labelframe(self.right, text="Daily Plan Checklist")
        plan_frame.pack(fill="x", pady=6)
        self.plan_vars: List[tk.BooleanVar] = []
        for i in range(3):
            var = tk.BooleanVar(value=False)
            self.plan_vars.append(var)
            self._checkbutton(plan_frame, text=f"Task {i+1}", variable=var).pack(anchor="w")

        alert = self._label(self.right, text="Disclaimer: Educational tool — not medical advice.", foreground="#a33")
        alert.pack(anchor="w", pady=4)

    def _build_layout(self):
        self.root.geometry("1100x650")

        self.main = ttk.Frame(self.root, padding=10)
        self.main.pack(fill="both", expand=True)

        self.left = ttk.Frame(self.main)
        self.center = ttk.Frame(self.main)
        self.right = ttk.Frame(self.main)

        self.left.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.center.grid(row=0, column=1, sticky="nsew", padx=(0,10))
        self.right.grid(row=0, column=2, sticky="nsew")

        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=2)
        self.main.columnconfigure(2, weight=2)
        self.main.rowconfigure(0, weight=1)

        # Left: Inputs
        inputs = ttk.LabelFrame(self.left, text="Today's Inputs")
        inputs.pack(fill="x", pady=4)

        self.var_sleep = tk.StringVar(value="7.0")
        self.var_hr = tk.StringVar(value="70")
        self.var_steps = tk.StringVar(value="6000")
        self.var_water = tk.StringVar(value="1800")
        self.var_stress = tk.StringVar(value="3")
        self.var_caffeine = tk.BooleanVar(value=False)

        def add_row(parent, label, var, unit=""):
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=16).pack(side="left")
            ttk.Entry(frame, textvariable=var, width=12).pack(side="left")
            if unit:
                ttk.Label(frame, text=unit).pack(side="left")

        add_row(inputs, "Sleep Hours", self.var_sleep, "h")
        add_row(inputs, "Resting HR", self.var_hr, "bpm")
        add_row(inputs, "Steps", self.var_steps, "count")
        add_row(inputs, "Water Intake", self.var_water, "ml")

        frame_stress = ttk.Frame(inputs)
        frame_stress.pack(fill="x", pady=2)
        ttk.Label(frame_stress, text="Stress (1–5)", width=16).pack(side="left")
        ttk.Combobox(frame_stress, textvariable=self.var_stress, values=["1","2","3","4","5"], width=10, state="readonly").pack(side="left")

        frame_caf = ttk.Frame(inputs)
        frame_caf.pack(fill="x", pady=2)
        ttk.Checkbutton(frame_caf, text="Caffeine today", variable=self.var_caffeine).pack(side="left")

        actions = ttk.Frame(self.left)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Compute Wellness", command=self.on_compute).pack(fill="x", pady=2)
        ttk.Button(actions, text="Save Day", command=self.on_save).pack(fill="x", pady=2)
        ttk.Button(actions, text="Reset Baselines", command=self.on_reset_baselines).pack(fill="x", pady=2)
        ttk.Button(actions, text="Export CSV", command=self.on_export).pack(fill="x", pady=2)

        # Center: Score + Trend + Log
        top_center = ttk.Frame(self.center)
        top_center.pack(fill="x")
        self.score_var = tk.StringVar(value="—")
        ttk.Label(top_center, text="Wellness Score", font=("Helvetica", 16, "bold")).pack(side="left")
        ttk.Label(top_center, textvariable=self.score_var, font=("Helvetica", 20)).pack(side="left", padx=10)

        chart_frame = ttk.LabelFrame(self.center, text="Last 7 Days Trend")
        chart_frame.pack(fill="both", expand=True, pady=6)
        self.chart_label = ttk.Label(chart_frame)
        self.chart_label.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(self.center, text="Agent Reasoning Trace")
        log_frame.pack(fill="both", expand=True, pady=6)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Right: Recommendations & Plan
        recs_frame = ttk.LabelFrame(self.right, text="Today's Recommendations")
        recs_frame.pack(fill="both", expand=True, pady=6)
        self.recs_tree = ttk.Treeview(recs_frame, columns=("rec", "why", "impact"), show="headings", height=6)
        self.recs_tree.heading("rec", text="Recommendation")
        self.recs_tree.heading("why", text="Rationale")
        self.recs_tree.heading("impact", text="Expected ΔScore")
        self.recs_tree.column("rec", width=180, anchor="w")
        self.recs_tree.column("why", width=320, anchor="w")
        self.recs_tree.column("impact", width=120, anchor="center")
        self.recs_tree.pack(fill="both", expand=True)

        plan_frame = ttk.LabelFrame(self.right, text="Daily Plan Checklist")
        plan_frame.pack(fill="x", pady=6)
        self.plan_vars: List[tk.BooleanVar] = []
        for i in range(3):
            var = tk.BooleanVar(value=False)
            self.plan_vars.append(var)
            ttk.Checkbutton(plan_frame, text=f"Task {i+1}", variable=var).pack(anchor="w")

        alert = ttk.Label(self.right, text="Disclaimer: Educational tool — not medical advice.", foreground="#a33")
        alert.pack(anchor="w", pady=4)

    def _log_to_ui(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for line in self.engine.log[-200:]:
            self.log_text.insert("end", line + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _render_chart(self):
        # Render last 7 scores to an image and display in label (only if libs available)
        if not (matplotlib and PIL_AVAILABLE):
            self.chart_label.configure(text="Install matplotlib & pillow to view chart")
            return
        scores = self.engine.state.last_scores[-7:]
        plt.figure(figsize=(5.8, 2.2), dpi=100)
        if scores:
            plt.plot(range(1, len(scores)+1), scores, marker="o")
            plt.ylim(0, 100)
            plt.ylabel("Score")
            plt.xlabel("Day")
            plt.title("Wellness Score Trend")
            plt.grid(True, linewidth=0.5)
        else:
            plt.text(0.5, 0.5, "No history yet", ha="center", va="center")
            plt.axis("off")
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        if PIL_AVAILABLE:
            img = Image.open(buf)
            self.chart_img = ImageTk.PhotoImage(img)
            self.chart_label.configure(image=self.chart_img)
        else:
            self.chart_label.configure(text="Install pillow to view chart")
        buf.close()

    # --- Controls ---
    def on_compute(self):
        in_sleep = safe_float(self.var_sleep.get(), 7.0)
        in_hr = safe_int(self.var_hr.get(), 70)
        in_steps = safe_int(self.var_steps.get(), 6000)
        in_water = safe_int(self.var_water.get(), 1800)
        in_stress = clamp(safe_int(self.var_stress.get(), 3), 1, 5)
        in_caf = bool(self.var_caffeine.get())

        metrics = self.engine.compute(
            sleep_h=in_sleep,
            resting_hr=in_hr,
            steps=in_steps,
            water_ml=in_water,
            stress_1to5=in_stress,
            caffeine=in_caf
        )
        self.score_var.set(f"{metrics['score']:.1f}")

        # Plan actions
        inputs = {
            "sleep_h": in_sleep,
            "resting_hr": in_hr,
            "steps": in_steps,
            "water_ml": in_water,
            "stress_1to5": in_stress,
            "caffeine": in_caf,
        }
        recs = self.engine.plan_actions(metrics, inputs)

        # Update recs tree
        for item in self.recs_tree.get_children():
            self.recs_tree.delete(item)
        for rec, why, impact in recs:
            self.recs_tree.insert("", "end", values=(rec, why, f"+{impact:.1f}"))

        self._log_to_ui()

    def on_save(self):
        # Save today's record to CSV and update baselines + chart
        try:
            score_text = self.score_var.get()
            if score_text == "—":
                messagebox.showinfo("Info", "Compute the wellness score first.")
                return

            in_sleep = safe_float(self.var_sleep.get(), 7.0)
            in_hr = safe_int(self.var_hr.get(), 70)
            in_steps = safe_int(self.var_steps.get(), 6000)
            in_water = safe_int(self.var_water.get(), 1800)
            in_stress = clamp(safe_int(self.var_stress.get(), 3), 1, 5)
            in_caf = bool(self.var_caffeine.get())

            score = float(score_text)

            # Gather recs in UI
            rec_texts = []
            for item in self.recs_tree.get_children():
                vals = self.recs_tree.item(item, "values")
                if vals:
                    rec_texts.append(vals[0])
            recs_joined = " | ".join(rec_texts)

            append_csv({
                "date": date.today().isoformat(),
                "sleep_h": f"{in_sleep:.1f}",
                "resting_hr": str(in_hr),
                "steps": str(in_steps),
                "water_ml": str(in_water),
                "stress_1to5": str(in_stress),
                "caffeine": str(int(in_caf)),
                "score": f"{score:.1f}",
                "recs": recs_joined
            })

            self.engine.update_baselines(in_sleep, in_hr, in_steps, in_water, score)
            self.engine.save_state()
            self._render_chart()
            self._log_to_ui()
            messagebox.showinfo("Saved", "Day saved. Baselines updated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save day: {e}")

    def on_reset_baselines(self):
        self.engine.state = ModelState.default()
        self.engine.save_state()
        self._render_chart()
        self._log_to_ui()
        messagebox.showinfo("Reset", "Baselines reset to defaults.")

    def on_export(self):
        if os.path.exists(CSV_PATH):
            messagebox.showinfo("Export", f"CSV saved at:\n{CSV_PATH}")
        else:
            messagebox.showinfo("Export", "No CSV exists yet. Save at least one day first.")


# ---------------------
# Launcher
# ---------------------
if __name__ == "__main__":
    root = tk.Tk()
    # bring the window to the front (helps on macOS)
    try:
        root.lift(); root.attributes('-topmost', True); root.after(300, lambda: root.attributes('-topmost', False))
    except Exception:
        pass

    if not FORCE_LEGACY_TK:
        try:
            style = ttk.Style(root)
            for name in ('aqua', 'default', 'alt', 'clam'):
                if name in style.theme_names():
                    style.theme_use(name)
                    break
            style.configure('.', background='white')
        except Exception:
            pass

    try:
        app = HeliosGUI(root)
    except Exception as e:
        import traceback
        traceback.print_exc()
        (tk.Label if FORCE_LEGACY_TK else ttk.Label)(root, text=f"HELIOS-AI failed to initialize: {e}", fg="#a33" if FORCE_LEGACY_TK else None, foreground=None if FORCE_LEGACY_TK else "#a33").pack(padx=20, pady=20)

    root.update_idletasks()
    root.deiconify()
    root.mainloop()
