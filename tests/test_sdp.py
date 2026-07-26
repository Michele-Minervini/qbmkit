"""SDP solving through the QBM core: exact solutions, KKT, strong duality, and an
independent reference implementation."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import logsumexp

import qbm
from qbm.losses import SDPDual, sdp_hamiltonian


# --- an independent reference dual, coded from scratch (no qbm machinery) ---
def _ref_gibbs(M, beta):
    w, V = np.linalg.eigh(M)
    p = np.exp(beta * (w - w[-1]))
    p /= p.sum()
    return (V * p) @ V.conj().T


def _ref_dual(y, C, A, b, beta):
    M = C - sum(yi * a for yi, a in zip(y, A)) if A else C
    w = np.linalg.eigvalsh(M)
    return logsumexp(beta * w) / beta + float(np.dot(b, y))


def _ref_solve(C, A, b, beta):
    y0 = np.zeros(len(A) if A else 1)
    res = minimize(
        _ref_dual, y0, args=(C, A, b, beta), method="BFGS", options={"gtol": 1e-10, "maxiter": 500}
    )
    M = C - sum(yi * a for yi, a in zip(res.x, A)) if A else C
    return res.x, _ref_gibbs(M, beta), res.fun


def _random_hermitian(dim, rng):
    Z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (Z + Z.conj().T) / 2


# ---------------------------------------------------------------------------
def test_unconstrained_solution_is_exact_gibbs_of_C():
    # With no constraints the optimum is exactly exp(beta C)/Tr exp(beta C).
    rng = np.random.default_rng(0)
    C = _random_hermitian(4, rng)
    beta = 3.0
    res = qbm.solve_sdp(C, beta=beta, steps=5)  # y stays at 0; no optimisation needed
    expected = expm(beta * C)
    expected = expected / np.trace(expected)
    assert np.allclose(res.X, expected, atol=1e-9)


def test_large_beta_recovers_max_eigenvalue():
    # As beta -> infinity the entropy term vanishes and <C,X> -> lambda_max(C).
    C = np.diag([3.0, 1.0, 0.0, -1.0]).astype(complex)
    res = qbm.solve_sdp(C, beta=200.0, steps=5)
    assert abs(res.objective - 3.0) < 1e-3


def test_dual_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    C = _random_hermitian(4, rng)
    A = [_random_hermitian(4, rng) for _ in range(2)]
    b = np.array([0.1, -0.2])
    beta = 2.0
    ham = sdp_hamiltonian(C, A, beta=beta)
    loss = SDPDual(b=b, beta=beta)
    y = rng.normal(scale=0.3, size=2)
    backend = qbm.DenseBackend()
    analytic = loss.grad(backend.thermal_state(ham, y))
    eps = 1e-6
    fd = np.zeros_like(y)
    for j in range(len(y)):
        yp = y.copy()
        yp[j] += eps
        ym = y.copy()
        ym[j] -= eps
        fd[j] = (
            loss.value(backend.thermal_state(ham, yp)) - loss.value(backend.thermal_state(ham, ym))
        ) / (2 * eps)
    assert np.allclose(analytic, fd, atol=1e-6)


def test_constrained_sdp_matches_independent_reference():
    rng = np.random.default_rng(2)
    dim, beta = 4, 4.0
    C = _random_hermitian(dim, rng)
    A = [_random_hermitian(dim, rng) for _ in range(2)]
    # choose b that is attainable: the constraint values of an actual density matrix
    X0 = _ref_gibbs(_random_hermitian(dim, rng), 1.0)
    b = np.array([float(np.real(np.trace(a @ X0))) for a in A])

    res = qbm.solve_sdp(C, A, b, beta=beta, steps=300)
    y_ref, X_ref, g_ref = _ref_solve(C, A, b, beta)

    assert np.allclose(res.X, X_ref, atol=1e-6)
    assert np.isclose(res.dual_value, g_ref, atol=1e-8)
    assert res.constraint_violation < 1e-6


def test_strong_duality_and_kkt():
    # at the optimum: dual value == <C,X> + S(X)/beta, and constraints hold.
    rng = np.random.default_rng(3)
    dim, beta = 4, 5.0
    C = _random_hermitian(dim, rng)
    A = [_random_hermitian(dim, rng)]
    X0 = _ref_gibbs(_random_hermitian(dim, rng), 1.0)
    b = np.array([float(np.real(np.trace(A[0] @ X0)))])

    res = qbm.solve_sdp(C, A, b, beta=beta, steps=300)
    primal = res.objective + res.entropy / beta
    assert np.isclose(primal, res.dual_value, atol=1e-6)  # strong duality
    assert res.constraint_violation < 1e-6  # primal feasibility


def test_solution_is_a_valid_density_matrix():
    rng = np.random.default_rng(4)
    C = _random_hermitian(4, rng)
    A = [_random_hermitian(4, rng)]
    res = qbm.solve_sdp(C, A, np.array([0.05]), beta=3.0, steps=200)
    X = res.X
    assert np.allclose(X, X.conj().T, atol=1e-10)
    assert np.isclose(np.trace(X).real, 1.0, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(X)) > -1e-10


def test_offset_does_not_break_gradients():
    # ParamHamiltonian.offset must leave dG/dtheta_j (and hence every gradient) intact.
    rng = np.random.default_rng(5)
    gens = qbm.local_pauli_generators(3)
    off = _random_hermitian(8, rng)
    ham = qbm.ParamHamiltonian(gens, offset=off)
    theta = rng.normal(scale=0.3, size=ham.n_params)
    O = qbm.hamiltonians.tfim(3, g=1.1)
    backend = qbm.DenseBackend()
    analytic = backend.thermal_state(ham, theta).observable_gradient(O)
    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += eps
        tm = theta.copy()
        tm[j] -= eps
        fd[j] = (
            backend.thermal_state(ham, tp).expect(O) - backend.thermal_state(ham, tm).expect(O)
        ) / (2 * eps)
    assert np.allclose(analytic, fd, atol=1e-6)


@pytest.mark.parametrize("backend", ["dense", "statevector"])
def test_sdp_backend_agnostic(backend):
    C = np.diag([2.0, 0.5, -1.0, -1.5]).astype(complex)
    res = qbm.solve_sdp(C, beta=20.0, steps=5, backend=backend)
    assert abs(res.objective - 2.0) < 0.05
