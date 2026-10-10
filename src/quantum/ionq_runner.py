"""
IonQ backend execution pipeline with local fallback and mock routing.
Supported by Qollab & IonQ.
"""

import os
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider

load_dotenv()


class IonQRunner:
    def __init__(self, api_key: Optional[str] = None, force_local: bool = False):
        """
        :param api_key: Explicit IonQ API token, defaults to env IONQ_API_KEY
        :param force_local: If True, uses local Qiskit StatevectorSampler without remote calls
        """
        self.api_key = api_key or os.getenv("IONQ_API_KEY")
        self.force_local = force_local
        self.backend = None
        self.mode = "local"
        self._setup_backend()

    def _setup_backend(self):
        if not self.force_local and self.api_key:
            try:
                provider = IonQProvider(token=self.api_key)
                self.backend = provider.get_backend("ionq_simulator")
                # Profile options: "ideal", "aria-1", "harmony"
                self.backend.set_options(noise_model="ideal")
                self.mode = "ionq_simulator"
                print(f"[OK] IonQ remote simulator connected successfully: {self.backend.name()}")
                return
            except Exception as e:
                print(f"[WARN] Failed to connect to IonQ cloud: {e}. Falling back to local simulator.")

        from qiskit.primitives import StatevectorSampler
        self.backend = StatevectorSampler()
        self.mode = "local_sampler"
        print("[INFO] Running in local high-precision simulation mode.")

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """
        Executes a quantum circuit and returns raw measurement distribution.
        """
        if self.mode == "ionq_simulator":
            job = self.backend.run(circuit, shots=shots)
            result = job.result()
            return result.get_counts(circuit)

        # Local Qiskit 1.x StatevectorSampler execution
        pub = (circuit,)
        job = self.backend.run([pub], shots=shots)
        result = job.result()[0]
        data_bin = result.data
        creg_name = circuit.cregs[0].name if circuit.cregs else "c"
        counts = getattr(data_bin, creg_name).get_counts()
        return counts


if __name__ == "__main__":
    from src.physics.tokamak import TokamakGeometry
    from src.physics.ising_kernel import generate_plasma_perturbation, build_ising_hamiltonian
    from src.quantum.qaoa_solver import QAOAPlasmasolver

    # 1. Physics setup
    geom = TokamakGeometry()
    B_mat = geom.compute_response_matrix()
    b_pert = generate_plasma_perturbation(geom.surface_points, geom.surface_normals)
    ising = build_ising_hamiltonian(B_mat, b_pert)

    # 2. QAOA setup
    solver = QAOAPlasmasolver(ising, p_layers=1)
    complexity = solver.analyze_circuit_complexity()
    print("[OK] Circuit Complexity Analysis:", complexity)

    # 3. Test execution (gamma=0.2, beta=0.4)
    qc = solver.build_circuit(gamma_vals=np.array([0.2]), beta_vals=np.array([0.4]))
    runner = IonQRunner(force_local=True)
    counts = runner.run(qc, shots=500)

    exp_e, best_bits, min_e = solver.compute_energy_from_counts(counts)
    print(f"[OK] Sampled {len(counts)} unique states.")
    print(f"Top candidate bitstring: {best_bits}")
    print(f"Minimum measured field error energy: {min_e:.4f}")
    print(f"Expected ensemble energy: {exp_e:.4f}")