"""
Cross-architecture transpilation benchmark: IonQ All-to-All vs Constrained Topology.
Supported by Qollab & IonQ.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from src.physics.tokamak import TokamakGeometry, TokamakConfig
from src.physics.ising_kernel import generate_plasma_perturbation, build_ising_hamiltonian
from src.quantum.qaoa_solver import QAOAPlasmasolver


@dataclass
class ArchitectureBenchmarkResult:
    architecture: str
    num_qubits: int
    circuit_depth: int
    two_qubit_gates: int
    total_gates: int
    swap_gates: int


def run_architecture_benchmark(num_qubits: int = 8) -> List[ArchitectureBenchmarkResult]:
    """
    Compares circuit compilation on:
    1. IonQ All-to-All topology (fully connected graph).
    2. Linear / Nearest-Neighbor chain topology (superconducting-like constraint).
    3. Heavy-Hex / Grid-like sparse topology.
    """
    # 1. Physics model setup
    config = TokamakConfig(num_coils=num_qubits)
    geom = TokamakGeometry(config)
    B_mat = geom.compute_response_matrix()
    b_pert = generate_plasma_perturbation(geom.surface_points, geom.surface_normals)
    ising = build_ising_hamiltonian(B_mat, b_pert)

    # 2. Build canonical QAOA circuit
    solver = QAOAPlasmasolver(ising, p_layers=1)
    qc = solver.build_circuit(gamma_vals=np.array([0.2]), beta_vals=np.array([0.4]))

    results = []

    # --- Scenario A: IonQ (Native All-to-All) ---
    all_to_all_map = CouplingMap.from_full(num_qubits)
    qc_ionq = transpile(
        qc,
        coupling_map=all_to_all_map,
        basis_gates=["rx", "ry", "rz", "rxx", "cx"],
        optimization_level=2,
    )
    ops_ionq = qc_ionq.count_ops()
    results.append(
        ArchitectureBenchmarkResult(
            architecture="IonQ (Trapped-Ion All-to-All)",
            num_qubits=num_qubits,
            circuit_depth=qc_ionq.depth(),
            two_qubit_gates=ops_ionq.get("rxx", 0) + ops_ionq.get("cx", 0),
            total_gates=sum(ops_ionq.values()),
            swap_gates=ops_ionq.get("swap", 0),
        )
    )

    # --- Scenario B: Linear Chain Topology ---
    linear_map = CouplingMap.from_line(num_qubits)
    qc_linear = transpile(
        qc,
        coupling_map=linear_map,
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=2,
        seed_transpiler=42,
    )
    ops_linear = qc_linear.count_ops()
    results.append(
        ArchitectureBenchmarkResult(
            architecture="Linear Chain (Nearest-Neighbor)",
            num_qubits=num_qubits,
            circuit_depth=qc_linear.depth(),
            two_qubit_gates=ops_linear.get("cx", 0),
            total_gates=sum(ops_linear.values()),
            swap_gates=ops_linear.get("swap", 0),
        )
    )

    # --- Scenario C: 2D Grid Topology ---
    # Approximate 2x4 grid for 8 qubits
    grid_rows = 2
    grid_cols = num_qubits // 2
    grid_map = CouplingMap.from_grid(grid_rows, grid_cols)
    qc_grid = transpile(
        qc,
        coupling_map=grid_map,
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=2,
        seed_transpiler=42,
    )
    ops_grid = qc_grid.count_ops()
    results.append(
        ArchitectureBenchmarkResult(
            architecture="2D Grid Topology (Planar Mesh)",
            num_qubits=num_qubits,
            circuit_depth=qc_grid.depth(),
            two_qubit_gates=ops_grid.get("cx", 0),
            total_gates=sum(ops_grid.values()),
            swap_gates=ops_grid.get("swap", 0),
        )
    )

    return results


def print_benchmark_table(results: List[ArchitectureBenchmarkResult]):
    header = f"{'Architecture':<35} | {'Qubits':<6} | {'Depth':<6} | {'2Q Gates':<8} | {'Total Gates':<11} | {'SWAPs':<6}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(" HARDWARE-FIT TRANSPILATION BENCHMARK (Plasma Coil Trimming QAOA)")
    print(separator)
    print(header)
    print(separator)

    for r in results:
        print(
            f"{r.architecture:<35} | {r.num_qubits:<6} | {r.circuit_depth:<6} | {r.two_qubit_gates:<8} | {r.total_gates:<11} | {r.swap_gates:<6}"
        )
    print(separator + "\n")


if __name__ == "__main__":
    benchmark_results = run_architecture_benchmark(num_qubits=8)
    print_benchmark_table(benchmark_results)