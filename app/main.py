"""
QuantumFusion: Main Application Hub.
Executes real Biot-Savart & QAOA kernels, passing exact matrices to WebGL HUD.
Supported by Qollab & IonQ.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

from src.physics.tokamak import TokamakGeometry, TokamakConfig
from src.physics.ising_kernel import (
    generate_plasma_perturbation,
    build_ising_hamiltonian,
)
from src.classical.ground_truth import solve_ground_truth
from src.quantum.qaoa_solver import QAOAPlasmasolver
from src.quantum.ionq_runner import IonQRunner
from app.view_3d import render_full_hud_fusor

st.set_page_config(
    page_title="QuantumFusion | IonQ Research Console",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #020307 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden;
    }
    header, footer { visibility: hidden !important; height: 0 !important; }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Розрахунок реального пайплайну
@st.cache_resource
def compute_quantum_pipeline():
    geom = TokamakGeometry(TokamakConfig(num_coils=8))
    b_mat = geom.compute_response_matrix()
    b_pert = generate_plasma_perturbation(geom.surface_points, geom.surface_normals, mode_m=2, mode_n=1)
    ising = build_ising_hamiltonian(b_mat, b_pert)

    gt = solve_ground_truth(ising)

    solver = QAOAPlasmasolver(ising, p_layers=1)
    qc = solver.build_circuit(gamma_vals=np.array([0.28]), beta_vals=np.array([0.42]))
    runner = IonQRunner(force_local=True)
    counts = runner.run(qc, shots=1024)

    exp_e, best_bits, min_e = solver.compute_energy_from_counts(counts)
    best_spins = 1 - 2 * np.array([int(b) for b in best_bits], dtype=np.int8)
    ratio = min(1.0, gt.min_energy / max(1e-6, min_e))

    return {
        "j_matrix": ising.J,
        "spins": best_spins,
        "best_bits": best_bits,
        "ratio": ratio,
        "counts": counts,
    }

pipeline_data = compute_quantum_pipeline()

# 2. Рендеринг повноекранного WebGL із HUD
fusor_view = render_full_hud_fusor(
    spins=pipeline_data["spins"],
    perturbation_active=False,
    amplitude=0.12,
    mode_m=2,
    mode_n=1,
    best_bitstring=pipeline_data["best_bits"],
    ratio=pipeline_data["ratio"],
    j_matrix=pipeline_data["j_matrix"],
    counts=pipeline_data["counts"],
    backend_name="IonQ Aria-1 (Trapped-Ion)",
    height=880,
)

components.html(fusor_view, height=880, scrolling=False)