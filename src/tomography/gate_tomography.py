import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Configuration
# ============================================================

# data shape:
#   data[ip0, ip1, im0, im1, iout]
#   shape = (4, 4, 3, 3, 4)

QPT_INPUT = {
    0: ["I"],
    1: ["X"],
    2: ["Y/2"],
    3: ["-X/2"],
}

GATE_PREPARATION = {
    "Q1": ["X/2"],
    "Q2": ["X/2"],
}

QST_SEQS = {
    0: ["I"],
    1: ["X/2"],
    2: ["Y/2"],
}

OUTCOME_ORDER = ("00", "01", "10", "11")

# ["A", "B"] means apply A then B
SEQUENCE_LEFT_TO_RIGHT = True

PROJECT_DENSITY_MATRIX = False
PROJECT_CPTP_CHANNEL = False


# ============================================================
# Basic matrices and gates
# ============================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([1.0, 0.0], dtype=complex)
ket1 = np.array([0.0, 1.0], dtype=complex)

rho0_1q = np.outer(ket0, ket0.conj())
P0_1q = np.outer(ket0, ket0.conj())
P1_1q = np.outer(ket1, ket1.conj())


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def rx(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X


def ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y


def rz(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ],
        dtype=complex,
    )


GATES = {
    "I": I2,
    "X": rx(np.pi),
    "X/2": rx(np.pi / 2),
    "-X/2": rx(-np.pi / 2),
    "Y/2": ry(np.pi / 2),
    "-Y/2": ry(-np.pi / 2),
}


def compose_sequence(seq: list[str]) -> np.ndarray:
    order = seq if SEQUENCE_LEFT_TO_RIGHT else list(reversed(seq))
    U = I2.copy()
    for gate_name in order:
        U = GATES[gate_name] @ U
    return U


# ============================================================
# Preparation states and measurement projectors
# ============================================================

def prep_sequence_for_qubit(qubit_name: str, idx: int) -> list[str]:
    return QPT_INPUT[idx] + GATE_PREPARATION[qubit_name]

def prep_state_1q_from_index(idx: int) -> np.ndarray:
    U = compose_sequence(QPT_INPUT[idx])
    return U @ rho0_1q @ U.conj().T

def prep_state_2q(ip0: int, ip1: int) -> np.ndarray:
    return kron(prep_state_1q_from_index(ip0), prep_state_1q_from_index(ip1))

def meas_projectors_1q_from_index(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Measurement is:
        apply qst pulse U, then do computational-basis readout.

    Effective POVM:
        M0 = U' |0><0| U
        M1 = U' |1><1| U
    """
    U = compose_sequence(QST_SEQS[idx])
    M0 = U.conj().T @ P0_1q @ U
    M1 = U.conj().T @ P1_1q @ U
    return M0, M1

def meas_projectors_2q(im0: int, im1: int) -> dict[str, np.ndarray]:
    M0a, M1a = meas_projectors_1q_from_index(im0)
    M0b, M1b = meas_projectors_1q_from_index(im1)
    return {
        "00": kron(M0a, M0b),
        "01": kron(M0a, M1b),
        "10": kron(M1a, M0b),
        "11": kron(M1a, M1b),
    }


# ============================================================
# Pauli basis
# ============================================================

PAULI_1Q = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_basis_2q() -> tuple[list[str], list[np.ndarray]]:
    labels = []
    mats = []
    for a in ["I", "X", "Y", "Z"]:
        for b in ["I", "X", "Y", "Z"]:
            labels.append(a + b)
            mats.append(kron(PAULI_1Q[a], PAULI_1Q[b]))
    return labels, mats


PTM_LABELS, PAULI_BASIS_2Q = pauli_basis_2q()


# ============================================================
# Linear algebra helpers
# ============================================================

def vec(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat, dtype=complex).reshape(-1, order="F")


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    return np.asarray(v, dtype=complex).reshape((d, d), order="F")


def unitary_to_superop(U: np.ndarray) -> np.ndarray:
    return np.kron(U, U.conj())


def apply_channel(S: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return unvec(S @ vec(rho), d=4)


def superop_to_choi(S: np.ndarray, d: int = 4) -> np.ndarray:
    return S.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)


def choi_to_superop(J: np.ndarray, d: int = 4) -> np.ndarray:
    return J.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)


def choi_ptrace_output(J: np.ndarray, d: int = 4) -> np.ndarray:
    JJ = J.reshape(d, d, d, d)
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        out += JJ[i, :, i, :]
    return out


# ============================================================
# Physical projection
# ============================================================

def project_to_psd(mat: np.ndarray) -> np.ndarray:
    mat = (mat + mat.conj().T) / 2
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(vals) @ vecs.conj().T


def project_density_matrix(rho: np.ndarray) -> np.ndarray:
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0.0, None)
    if vals.sum() <= 0:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    rho = vecs @ np.diag(vals) @ vecs.conj().T
    rho /= np.trace(rho)
    return rho


def project_to_tp_choi(J: np.ndarray, d: int = 4) -> np.ndarray:
    delta = choi_ptrace_output(J, d=d) - np.eye(d)
    return J - np.kron(np.eye(d) / d, delta)


def project_to_cptp(S: np.ndarray, d: int = 4, n_iter: int = 30) -> np.ndarray:
    J = superop_to_choi(S, d=d)
    J = (J + J.conj().T) / 2
    for _ in range(n_iter):
        J = project_to_psd(J)
        J = project_to_tp_choi(J, d=d)
        J = (J + J.conj().T) / 2
    return choi_to_superop(J, d=d)


# ============================================================
# Target processes
# ============================================================

def cz_unitary() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(complex)

# ============================================================
# Data normalization
# ============================================================

def normalize_last_axis(data: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    data = np.asarray(data)
    if data.shape != (4, 4, 3, 3, 4):
        raise ValueError(f"data.shape must be (4,4,3,3,4), got {data.shape}")

    sums = data.sum(axis=-1)

    if np.allclose(sums, 1.0, atol=atol):
        return data.copy()

    if np.any(sums <= 0):
        raise ValueError("Found settings with non-positive total counts.")

    return data / sums[..., None]


# ============================================================
# State tomography for one fixed input
# ============================================================

def reconstruct_output_state(block_prob: np.ndarray) -> np.ndarray:
    """
    block_prob shape = (3, 3, 4)
    """
    A_rows = []
    p_list = []

    for im0 in range(3):
        for im1 in range(3):
            proj_dict = meas_projectors_2q(im0, im1)
            for iout, outcome in enumerate(OUTCOME_ORDER):
                M = proj_dict[outcome]
                p = block_prob[im0, im1, iout]
                row = [np.real(np.trace(M @ Pk)) / 4.0 for Pk in PAULI_BASIS_2Q]
                A_rows.append(row)
                p_list.append(p)

    A = np.array(A_rows)
    p = np.array(p_list)

    r, *_ = np.linalg.lstsq(A, p, rcond=None)

    rho = np.zeros((4, 4), dtype=complex)
    for coeff, Pk in zip(r, PAULI_BASIS_2Q):
        rho += coeff * Pk
    rho *= 0.25

    if PROJECT_DENSITY_MATRIX:
        rho = project_density_matrix(rho)

    return rho


def reconstruct_all_output_states(data_prob: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    output_states = {}
    for ip0 in range(4):
        for ip1 in range(4):
            output_states[(ip0, ip1)] = reconstruct_output_state(data_prob[ip0, ip1])
    return output_states


# ============================================================
# Process tomography
# ============================================================

def fit_superop(output_states: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    inputs = [(i, j) for i in range(4) for j in range(4)]

    Xmat = np.column_stack([vec(prep_state_2q(i, j)) for (i, j) in inputs])
    Ymat = np.column_stack([vec(output_states[(i, j)]) for (i, j) in inputs])

    S = Ymat @ np.linalg.pinv(Xmat)

    if PROJECT_CPTP_CHANNEL:
        S = project_to_cptp(S, d=4, n_iter=30)

    return S


def superop_to_ptm(S: np.ndarray) -> np.ndarray:
    R = np.zeros((16, 16), dtype=float)
    for j, Pj in enumerate(PAULI_BASIS_2Q):
        E_Pj = apply_channel(S, Pj)
        for i, Pi in enumerate(PAULI_BASIS_2Q):
            R[i, j] = np.real(np.trace(Pi @ E_Pj)) / 4.0
    return R


def superop_to_chi_pauli(S: np.ndarray) -> np.ndarray:
    d = 4
    J = superop_to_choi(S, d=d)
    basis = [P / np.sqrt(d) for P in PAULI_BASIS_2Q]
    B = np.column_stack([vec(E) for E in basis])
    chi = B.conj().T @ J @ B
    chi = (chi + chi.conj().T) / 2
    return chi / d


# ============================================================
# Fidelities
# ============================================================

def process_fidelity_to_target(S_est: np.ndarray, U_target: np.ndarray) -> float:
    d = U_target.shape[0]
    J_est = superop_to_choi(S_est, d=d)
    J_tar = superop_to_choi(unitary_to_superop(U_target), d=d)
    return float(np.real(np.trace(J_est @ J_tar)) / (d * d))


def average_gate_fidelity_to_target(S_est: np.ndarray, U_target: np.ndarray) -> float:
    d = U_target.shape[0]
    f_pro = process_fidelity_to_target(S_est, U_target)
    return float((d * f_pro + 1.0) / (d + 1.0))


def optimize_post_local_z_fidelity(
    S_est: np.ndarray,
    target_builder,
    n_grid: int = 181,
) -> dict[str, float]:
    phis = np.linspace(-np.pi, np.pi, n_grid)

    best = {
        "phi1": 0.0,
        "phi2": 0.0,
        "process_fidelity": -1.0,
        "average_gate_fidelity": -1.0,
    }

    U_base = target_builder()

    for phi1 in phis:
        for phi2 in phis:
            U_loc = kron(rz(phi1), rz(phi2))
            U_target = U_loc @ U_base
            f_pro = process_fidelity_to_target(S_est, U_target)
            if f_pro > best["process_fidelity"]:
                best["phi1"] = float(phi1)
                best["phi2"] = float(phi2)
                best["process_fidelity"] = float(f_pro)
                best["average_gate_fidelity"] = float(
                    average_gate_fidelity_to_target(S_est, U_target)
                )

    return best


# ============================================================
# Residual diagnostics
# ============================================================

def predict_probs_for_setting(
    S: np.ndarray,
    ip0: int,
    ip1: int,
    im0: int,
    im1: int,
) -> np.ndarray:
    rho_in = prep_state_2q(ip0, ip1)
    rho_out = apply_channel(S, rho_in)
    proj_dict = meas_projectors_2q(im0, im1)

    p = np.array(
        [
            np.real(np.trace(proj_dict["00"] @ rho_out)),
            np.real(np.trace(proj_dict["01"] @ rho_out)),
            np.real(np.trace(proj_dict["10"] @ rho_out)),
            np.real(np.trace(proj_dict["11"] @ rho_out)),
        ],
        dtype=float,
    )

    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        return np.ones(4) / 4
    return p / p.sum()


def global_rms_residual(data_prob: np.ndarray, S: np.ndarray) -> float:
    rms_list = []
    for ip0 in range(4):
        for ip1 in range(4):
            for im0 in range(3):
                for im1 in range(3):
                    p_obs = data_prob[ip0, ip1, im0, im1, :]
                    p_fit = predict_probs_for_setting(S, ip0, ip1, im0, im1)
                    rms_list.append(np.sqrt(np.mean((p_obs - p_fit) ** 2)))
    return float(np.real(np.sqrt(np.mean(np.array(rms_list) ** 2))))


# ============================================================
# Result object
# ============================================================

@dataclass
class CZTomographyResult:
    superop: np.ndarray
    choi: np.ndarray
    chi: np.ndarray
    ptm: np.ndarray
    ptm_labels: list[str]
    output_states: dict[tuple[int, int], np.ndarray]
    process_fidelity_bare_cz: float
    average_gate_fidelity_bare_cz: float
    optimized_local_z_bare_cz: dict[str, float]
    rms_residual: float


# ============================================================
# Main API
# ============================================================

def run_cz_tomography(data: np.ndarray) -> CZTomographyResult:
    data_prob = normalize_last_axis(data)
    output_states = reconstruct_all_output_states(data_prob)
    S = fit_superop(output_states)

    choi = superop_to_choi(S, d=4)
    chi = superop_to_chi_pauli(S)
    ptm = superop_to_ptm(S)

    f_pro_cz = process_fidelity_to_target(S, cz_unitary())
    f_avg_cz = average_gate_fidelity_to_target(S, cz_unitary())
    f_opt_cz = optimize_post_local_z_fidelity(S, cz_unitary, n_grid=181)

    rms = global_rms_residual(data_prob, S)

    return CZTomographyResult(
        superop=S,
        choi=choi,
        chi=chi,
        ptm=ptm,
        ptm_labels=PTM_LABELS,
        output_states=output_states,
        process_fidelity_bare_cz=f_pro_cz,
        average_gate_fidelity_bare_cz=f_avg_cz,
        optimized_local_z_bare_cz=f_opt_cz,
        rms_residual=rms,
    )


# ============================================================
# Diagnostics
# ============================================================

def effective_measured_observable(idx: int) -> np.ndarray:
    U = compose_sequence(QST_SEQS[idx])
    return U.conj().T @ Z @ U


def print_effective_axes() -> None:
    print("Effective measured observables from qst only:")
    for idx in [0, 1, 2]:
        print(f"qst index {idx}, seq = {QST_SEQS[idx]}")
        print(np.round(effective_measured_observable(idx), 4))
        print()


def print_result(res: CZTomographyResult) -> None:
    np.set_printoptions(precision=5, suppress=True)

    print("PTM basis order:")
    print(res.ptm_labels)
    print()

    print("Fidelity to bare CZ:")
    print(f"  process fidelity      = {res.process_fidelity_bare_cz:.8f}")
    print(f"  average gate fidelity = {res.average_gate_fidelity_bare_cz:.8f}")
    print()

    print("Optimized local-Z fidelity to bare CZ:")
    print(f"  phi1                  = {res.optimized_local_z_bare_cz['phi1']:.8f}")
    print(f"  phi2                  = {res.optimized_local_z_bare_cz['phi2']:.8f}")
    print(f"  process fidelity      = {res.optimized_local_z_bare_cz['process_fidelity']:.8f}")
    print(f"  average gate fidelity = {res.optimized_local_z_bare_cz['average_gate_fidelity']:.8f}")
    print()

    print(f"Global RMS residual     = {res.rms_residual:.8e}")
    print()

    print("Example output state for input (2, 2) [Y/2, Y/2]:")
    print(res.output_states[(2, 2)])
    print()


# ============================================================
# Plotting
# ============================================================

def plot_ptm_diagonal_bar(res: CZTomographyResult) -> None:
    diag = np.diag(res.ptm)
    plt.figure(figsize=(10, 4))
    plt.bar(res.ptm_labels, diag)
    plt.xlabel("Pauli basis")
    plt.ylabel("PTM diagonal element")
    plt.title("CZ tomography: PTM diagonal")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_chi_3d(res: CZTomographyResult) -> None:
    chi = res.chi
    vals = np.abs(chi)
    phases = np.angle(chi)  # in [-pi, pi]
    title = "Chi matrix (absolute value, color = phase)"

    n = vals.shape[0]
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = xx.ravel()
    y = yy.ravel()

    flat_vals = vals.ravel()
    flat_phases = phases.ravel()

    z0 = np.where(flat_vals >= 0, 0.0, flat_vals)
    dz = np.abs(flat_vals)

    dx = 0.6 * np.ones_like(x, dtype=float)
    dy = 0.6 * np.ones_like(y, dtype=float)

    phase_cmap = LinearSegmentedColormap.from_list(
        "phase_cmap",
        [
            (0.00, "red"),
            (0.25, "lime"),
            (0.50, "blue"),
            (0.75, "cyan"),
            (1.00, "red"),
        ]
    )
    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = phase_cmap
    bar_colors = cmap(norm(flat_phases))

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(x, y, z0, dx, dy, dz, color=bar_colors, shade=True)

    ax.set_title(title)

    ax.set_xticks(np.arange(n) + 0.3)
    ax.set_yticks(np.arange(n) + 0.3)
    ax.set_xticklabels(res.ptm_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(res.ptm_labels, fontsize=8)

    # colorbar for phase
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
    cbar.set_label("Phase")
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    plt.tight_layout()
    plt.show()
