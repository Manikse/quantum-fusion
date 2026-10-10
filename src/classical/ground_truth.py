"""
Exact Classical Ground Truth solver for Ising magnetic confinement.
Supported by Qollab & IonQ.
"""

from dataclasses import dataclass
import time
from typing import Tuple, List
import numpy as np

from src.physics.ising_kernel import IsingModel, evaluate_ising_energy


@dataclass
class GroundTruthResult:
    best_spins: np.ndarray       # Оптимальні напрямки струмів (-1 або +1)
    best_bitstring: str          # Бітовий рядок ('0' для +1, '1' для -1)
    min_energy: float            # Мінімальна залишкова енергія витоку поля
    all_energies: np.ndarray     # Енергії всіх 2^N станів (для аналізу спектра)
    execution_time_ms: float     # Час класичного розрахунку


def solve_ground_truth(model: IsingModel) -> GroundTruthResult:
    """
    Векторизований точний перебір усіх 2^N комбінацій спінів.
    Знаходить глобальний мінімум енергії Гамільтоніана.
    """
    start_time = time.perf_counter()
    N = model.J.shape[0]
    total_states = 1 << N  # 2^N

    # Генеруємо матрицю всіх можливих бітових конфігурацій (2^N, N)
    indices = np.arange(total_states, dtype=np.uint32)[:, None]
    bit_shifts = np.arange(N - 1, -1, -1, dtype=np.uint32)
    # Біти: 0 або 1
    bits = (indices >> bit_shifts) & 1

    # Перетворюємо біти в спіни: біт 0 -> spin +1, біт 1 -> spin -1
    # Це стандартний конвеншн Qiskit: |0> відповідає Z = +1, |1> відповідає Z = -1
    spins = 1.0 - 2.0 * bits.astype(np.float64)  # (2^N, N)

    # Векторизований розрахунок енергії для всіх 2^N станів:
    # E = 0.5 * sum_i,j (s_i * J_ij * s_j) + sum_i (h_i * s_i) + offset
    # (spins @ J) * spins дає построковий розрахунок форми (2^N, N)
    quadratic_terms = 0.5 * np.sum((spins @ model.J) * spins, axis=1)
    linear_terms = spins @ model.h
    energies = quadratic_terms + linear_terms + model.offset

    # Знаходимо стан з абсолютно мінімальною енергією
    min_idx = int(np.argmin(energies))
    min_energy = float(energies[min_idx])
    best_spins = spins[min_idx].astype(np.int8)
    best_bits = bits[min_idx]
    best_bitstring = "".join(str(b) for b in best_bits)

    calc_time = (time.perf_counter() - start_time) * 1000.0

    return GroundTruthResult(
        best_spins=best_spins,
        best_bitstring=best_bitstring,
        min_energy=min_energy,
        all_energies=energies,
        execution_time_ms=calc_time,
    )


if __name__ == "__main__":
    from src.physics.tokamak import TokamakGeometry
    from src.physics.ising_kernel import (
        generate_plasma_perturbation,
        build_ising_hamiltonian,
    )

    geom = TokamakGeometry()
    B_mat = geom.compute_response_matrix()
    b_pert = generate_plasma_perturbation(geom.surface_points, geom.surface_normals)
    ising = build_ising_hamiltonian(B_mat, b_pert)

    res = solve_ground_truth(ising)

    print("[OK] Класичний точний розв'язок знайдено:")
    print(f"Час обчислення: {res.execution_time_ms:.3f} ms")
    print(f"Оптимальний бітовий рядок (bitstring): {res.best_bitstring}")
    print(f"Оптимальні струми (spins): {res.best_spins}")
    print(f"Мінімальна залишкова енергія поля: {res.min_energy:.6f}")
    print(f"Максимальна енергія (найгірший стан): {np.max(res.all_energies):.6f}")