"""
Tokamak geometry and Biot-Savart magnetic field kernel.
Supported by Qollab & IonQ.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class TokamakConfig:
    major_radius: float = 2.0      # R: відстань від осі тора до центру трубки (м)
    minor_radius: float = 0.6      # r: радіус плазмового шнура (м)
    num_coils: int = 8             # N: кількість коригувальних котушок (кубітів)
    coil_radius: float = 0.35      # радіус кожної окремої котушки (м)
    coil_standoff: float = 0.85    # радіальне зміщення котушок від центру трубки (м)
    surface_points_theta: int = 12 # дискретизація полоїдального кута
    surface_points_phi: int = 16   # дискретизація тороїдального кута
    coil_discretization: int = 32  # кількість сегментів на кожну котушку


class TokamakGeometry:
    def __init__(self, config: TokamakConfig = TokamakConfig()):
        self.cfg = config
        self.surface_points, self.surface_normals = self._build_plasma_surface()
        self.coils_segments = self._build_correction_coils()

    def _build_plasma_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерує K контрольних точок на межі плазми (LCFS)
        та одиничні вектори зовнішньої нормалі n.
        """
        theta = np.linspace(0, 2 * np.pi, self.cfg.surface_points_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, self.cfg.surface_points_phi, endpoint=False)
        th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")

        th_flat = th_grid.flatten()
        ph_flat = ph_grid.flatten()

        R, r = self.cfg.major_radius, self.cfg.minor_radius

        # Координати поверхні тора
        x = (R + r * np.cos(th_flat)) * np.cos(ph_flat)
        y = (R + r * np.cos(th_flat)) * np.sin(ph_flat)
        z = r * np.sin(th_flat)
        points = np.stack([x, y, z], axis=-1)  # Форма: (K, 3)

        # Зовнішня нормаль для кругового тора
        nx = np.cos(th_flat) * np.cos(ph_flat)
        ny = np.cos(th_flat) * np.sin(ph_flat)
        nz = np.sin(th_flat)
        normals = np.stack([nx, ny, nz], axis=-1)  # Форма: (K, 3)

        return points, normals

    def _build_correction_coils(self) -> np.ndarray:
        """
        Генерує контури N коригувальних котушок навколо тора.
        Повертає масив форми: (N_coils, segments, 3).
        """
        coils = []
        angles = np.linspace(0, 2 * np.pi, self.cfg.num_coils, endpoint=False)
        t_seg = np.linspace(0, 2 * np.pi, self.cfg.coil_discretization, endpoint=False)

        R_c = self.cfg.major_radius + self.cfg.coil_standoff
        r_c = self.cfg.coil_radius

        for phi_c in angles:
            # Центр котушки на зовнішньому периметрі тора
            center = np.array([R_c * np.cos(phi_c), R_c * np.sin(phi_c), 0.0])

            # Площина котушки перпендикулярна радіальному вектору
            tangent = np.array([-np.sin(phi_c), np.cos(phi_c), 0.0])
            normal_z = np.array([0.0, 0.0, 1.0])

            # Дискретні точки контуру котушки
            loop_points = (
                center[None, :]
                + r_c * np.cos(t_seg)[:, None] * tangent[None, :]
                + r_c * np.sin(t_seg)[:, None] * normal_z[None, :]
            )
            coils.append(loop_points)

        return np.array(coils)  # (num_coils, segments, 3)

    def compute_response_matrix(self) -> np.ndarray:
        """
        Обчислює матрицю B_n форми (K, num_coils) через чисельне інтегрування
        закону Біо-Савара. Кожен елемент B_n[k, m] — це проекція магнітного поля
        від котушки m на нормаль n у контрольній точці k (при одиничному струмі I=1 А).
        """
        K = self.surface_points.shape[0]
        N = self.cfg.num_coils
        B_norm_matrix = np.zeros((K, N), dtype=np.float64)

        # mu_0 / (4 * pi) = 1e-7 Тл*м/А (для масштабування беремо нормалізоване значення 1.0)
        c = 1.0

        for m in range(N):
            loop = self.coils_segments[m]  # (S, 3)
            # Вектори відрізків dl_j = r_{j+1} - r_j
            dl = np.roll(loop, -1, axis=0) - loop  # (S, 3)
            midpoints = 0.5 * (loop + np.roll(loop, -1, axis=0))  # (S, 3)

            # Вектори від сегментів котушки до кожної контрольної точки:
            # r_vec: (K, S, 3)
            r_vec = self.surface_points[:, None, :] - midpoints[None, :, :]
            dist = np.linalg.norm(r_vec, axis=-1)  # (K, S)
            dist_cubed = np.maximum(dist, 1e-4) ** 3  # захист від ділення на 0

            # Векторний добуток dl x r_vec для кожного сегмента
            # (K, S, 3)
            dl_cross_r = np.cross(dl[None, :, :], r_vec)
            # Поле Біо-Савара B_field у точці k: сума по сегментах S
            B_field = c * np.sum(dl_cross_r / dist_cubed[:, :, None], axis=1)  # (K, 3)

            # Проекція на нормаль: B_n = B · n
            B_norm_matrix[:, m] = np.sum(B_field * self.surface_normals, axis=-1)

        return B_norm_matrix


if __name__ == "__main__":
    geom = TokamakGeometry()
    B_matrix = geom.compute_response_matrix()
    print("[OK] Геометрія успішно ініціалізована.")
    print(f"Контрольних точок на торі: {B_matrix.shape[0]}")
    print(f"Кількість котушок (кубітів): {B_matrix.shape[1]}")
    print(f"Середнє значення поля: {np.mean(np.abs(B_matrix)):.4e}")