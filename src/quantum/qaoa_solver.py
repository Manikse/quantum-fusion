"""
Parameterized Quantum Approximate Optimization Algorithm (QAOA) for Tokamak Confinement.
Supported by Qollab & IonQ.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.physics.ising_kernel import IsingModel, evaluate_ising_energy


@dataclass
class QAOAResult:
    optimal_gamma: np.ndarray
    optimal_beta: np.ndarray
    best_bitstring: str
    best_spins: np.ndarray
    expected_energy: float
    min_measured_energy: float
    counts: Dict[str, int]
    circuit_depth: int
    num_two_qubit_gates: int


class QAOAPlasmasolver:
    def __init__(self, ising_model: IsingModel, p_layers: int = 1):
        """
        :param ising_model: Ising Hamiltonian parameters (J, h, offset)
        :param p_layers: Number of QAOA alternating operator layers
        """
        self.model = ising_model
        self.p = p_layers
        self.num_qubits = ising_model.J.shape[0]

    def build_circuit(
        self, gamma_vals: Optional[np.ndarray] = None, beta_vals: Optional[np.ndarray] = None
    ) -> QuantumCircuit:
        """
        Constructs the parameterized QAOA quantum circuit.
        If numeric parameters are supplied, binds them directly.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # 1. Uniform superposition initialization: |+>^N
        qc.h(range(self.num_qubits))

        if gamma_vals is None or beta_vals is None:
            gamma = ParameterVector("gamma", self.p)
            beta = ParameterVector("beta", self.p)
        else:
            gamma = gamma_vals
            beta = beta_vals

        N = self.num_qubits
        J = self.model.J
        h = self.model.h

        # 2. Alternating Cost and Mixer layers
        for layer in range(self.p):
            g = gamma[layer]
            b = beta[layer]

            # --- Problem Hamiltonian U(H_C, gamma) ---
            # Pairwise interaction terms: exp(-i * gamma * J_ml * Z_m * Z_l)
            for m in range(N):
                for l in range(m + 1, N):
                    weight = J[m, l]
                    if abs(weight) > 1e-6:
                        # RZZ gate implements exp(-i * (theta/2) * Z x Z)
                        qc.rzz(2.0 * g * weight, m, l)

            # Linear terms: exp(-i * gamma * h_m * Z_m)
            for m in range(N):
                field = h[m]
                if abs(field) > 1e-6:
                    qc.rz(2.0 * g * field, m)

            # --- Mixer Hamiltonian U(H_M, beta) ---
            # Transverse field: exp(-i * beta * X_m)
            for m in range(N):
                qc.rx(2.0 * b, m)

        # Measurement layer
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def compute_energy_from_counts(self, counts: Dict[str, int]) -> Tuple[float, str, float]:
        """
        Calculates expectation energy and finds the best sample from measured bitstrings.
        """
        total_shots = sum(counts.values())
        expected_energy = 0.0
        min_energy = float("inf")
        best_bitstring = ""

        for bitstring, count in counts.items():
            # Standard Qiskit bit ordering is reversed (q_{N-1}...q_0)
            bits = np.array([int(bit) for bit in bitstring[::-1]], dtype=np.int8)
            spins = 1 - 2 * bits
            energy = evaluate_ising_energy(self.model, spins)

            expected_energy += (count / total_shots) * energy
            if energy < min_energy:
                min_energy = energy
                best_bitstring = bitstring[::-1]

        return expected_energy, best_bitstring, min_energy

    def analyze_circuit_complexity(self) -> Dict[str, Any]:
        """
        Extracts metrics demonstrating IonQ hardware-fit (depth, 2-qubit gates).
        """
        qc = self.build_circuit()
        ops = qc.count_ops()
        two_qubit_count = ops.get("rzz", 0) + ops.get("cx", 0)

        return {
            "qubits": self.num_qubits,
            "depth": qc.depth(),
            "two_qubit_gates": two_qubit_count,
            "total_gates": sum(ops.values()),
        }