"""
Ising Hamiltonian mapping for magnetic confinement optimization.
Supported by Qollab & IonQ.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IsingModel:
    J: np.ndarray      # Symmetrized coupling matrix (N, N) with zero diagonal
    h: np.ndarray      # Linear bias vector / perturbation projection (N,)
    offset: float      # Energy scalar offset


def generate_plasma_perturbation(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    mode_m: int = 2,
    mode_n: int = 1,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generates an MHD boundary perturbation (m/n tearing/kink mode).
    Returns normalized normal component: B_pert_n of shape (K,).
    """
    phi = np.arctan2(surface_points[:, 1], surface_points[:, 0])
    z = surface_points[:, 2]
    theta = np.arcsin(np.clip(z / 0.6, -1.0, 1.0))

    # Phase-dependent spatial mode
    perturbation = amplitude * np.cos(mode_m * theta - mode_n * phi)
    return perturbation


def build_ising_hamiltonian(
    B_response_matrix: np.ndarray,
    B_perturb_normal: np.ndarray,
) -> IsingModel:
    """
    Constructs the exact Ising Hamiltonian:
    E(s) = || B_response @ s + B_perturb ||^2
    """
    K, N = B_response_matrix.shape

    # Balanced physical scaling between response and perturbation
    norm_factor = np.linalg.norm(B_response_matrix) / np.sqrt(K * N)
    B_scaled = B_response_matrix / norm_factor
    b_scaled = B_perturb_normal / (np.std(B_perturb_normal) + 1e-6) * 1.8

    Q = B_scaled.T @ B_scaled
    L = 2.0 * (B_scaled.T @ b_scaled)
    const_perturb = float(np.sum(b_scaled**2))
    diag_sum = float(np.trace(Q))

    total_offset = const_perturb + diag_sum

    J = np.zeros((N, N), dtype=np.float64)
    for m in range(N):
        for l in range(m + 1, N):
            coupling = 2.0 * Q[m, l]
            J[m, l] = coupling
            J[l, m] = coupling

    h = L.copy()
    return IsingModel(J=J, h=h, offset=total_offset)


def evaluate_ising_energy(model: IsingModel, spins: np.ndarray) -> float:
    quadratic = 0.5 * float(spins.T @ model.J @ spins)
    linear = float(np.dot(model.h, spins))
    return quadratic + linear + model.offset