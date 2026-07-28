"""alpha-z information matrices (arXiv:2510.02218, Wilde) and arbitrary metric kernels.

The alpha-z family is a two-parameter generalisation of the quantum Fisher
information that contains every metric this library shipped before as a special case.
"""

import numpy as np
import pytest

import qbm
from qbm.metrics import AlphaZ, CustomMetric, PetzRenyi, SandwichedRenyi, alpha_z_is_monotone
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _state(seed=0, n=3, scale=0.5):
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = np.random.default_rng(seed).normal(scale=scale, size=ham.n_params)
    return ham, theta, qbm.DenseBackend().thermal_state(ham, theta)


def _psd(M, tol=1e-9):
    return np.min(np.linalg.eigvalsh(M)) > -tol


# ---------------------------------------------------------------------------
# the named metrics are special cases of the family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, alpha, z",
    [
        ("kubo_mori", 1.0, 1.0),  # alpha -> 1, any z
        ("fisher_bures", 0.5, 0.5),  # sandwiched Renyi 1/2
        ("wigner_yanase", 0.5, 1.0),  # Petz-Renyi 1/2
    ],
)
def test_named_metrics_are_alpha_z_special_cases(name, alpha, z):
    _, _, st = _state(seed=1)
    assert np.allclose(st.metric(name), AlphaZ(alpha, z).matrix(st), atol=1e-12)


def test_sandwiched_alpha_2_is_the_geometric_mean_kernel():
    _, _, st = _state(seed=2)
    geometric = CustomMetric(lambda x, y: 1.0 / np.sqrt(x * y), "geometric")
    assert np.allclose(SandwichedRenyi(2.0).matrix(st), geometric.matrix(st), atol=1e-10)


def test_alpha_to_one_converges_to_kubo_mori_for_any_z():
    _, _, st = _state(seed=3)
    km = st.metric("kubo_mori")
    prev = np.inf
    for a in (0.9, 0.99, 0.999):
        err = np.max(np.abs(AlphaZ(a, 3.0, check_monotone=False).matrix(st) - km))
        assert err < prev  # monotone convergence to the Kubo-Mori limit
        prev = err
    assert prev < 1e-3


# ---------------------------------------------------------------------------
# the definitive check: against the divergence the matrix is defined from
# ---------------------------------------------------------------------------
def _matrix_power(A, r):
    w, V = np.linalg.eigh(A)
    return (V * (np.clip(w.real, 1e-300, None) ** r)) @ V.conj().T


def _alpha_z_divergence(rho, sigma, a, z):
    """``D_{alpha,z}(rho||sigma)`` straight from its definition (Eq. 5.1)."""
    S = _matrix_power(sigma, (1 - a) / (2 * z))
    R = _matrix_power(rho, a / z)
    return float(np.real(np.log(np.trace(_matrix_power(S @ R @ S, z))) / (a - 1)))


@pytest.mark.parametrize("alpha, z", [(0.5, 0.5), (0.5, 1.0), (0.3, 0.8), (2.0, 1.0), (0.7, 2.0)])
def test_matches_hessian_of_the_alpha_z_divergence(alpha, z):
    # Eq. (5.4): I_{a,z} = (1/a) * Hessian_eps D_{a,z}(rho(theta) || rho(theta+eps)).
    # Comparing the Theorem-10 spectral formula against a finite-difference Hessian of
    # the divergence independently validates the whole implementation.
    ham, theta, st = _state(seed=4, n=3, scale=0.4)
    backend = qbm.DenseBackend()
    rho0 = st.density_matrix()
    J = ham.n_params
    eps = 2e-4

    def f(e):
        return _alpha_z_divergence(
            rho0, backend.thermal_state(ham, theta + e).density_matrix(), alpha, z
        )

    numeric = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            ei = np.zeros(J)
            ei[i] = eps
            ej = np.zeros(J)
            ej[j] = eps
            numeric[i, j] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * eps**2)
    numeric /= alpha

    analytic = AlphaZ(alpha, z, check_monotone=False).matrix(st)
    assert np.max(np.abs(analytic - numeric)) / np.max(np.abs(numeric)) < 1e-5


# ---------------------------------------------------------------------------
# structural properties from the paper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha, z", [(0.5, 0.5), (0.5, 1.0), (0.3, 0.8), (2.0, 1.0), (2.0, 2.0)])
def test_information_matrices_are_symmetric_and_psd(alpha, z):
    _, _, st = _state(seed=5)
    g = AlphaZ(alpha, z, check_monotone=False).matrix(st)
    assert np.all(np.isfinite(g))
    assert np.allclose(g, g.T, atol=1e-9)
    assert _psd(g)


def test_sandwiched_renyi_is_increasing_in_alpha():
    # Theorem 30: monotone increasing in the Loewner order on alpha in (0, inf).
    _, _, st = _state(seed=6)
    alphas = [0.6, 0.8, 1.0, 1.5, 2.0]
    mats = [AlphaZ(a, a, check_monotone=False).matrix(st) for a in alphas]
    for lo, hi in zip(mats, mats[1:]):
        assert _psd(hi - lo)


def test_petz_renyi_ordering_around_one_half():
    # Theorem 28: decreasing on (0, 1/2], increasing on [1/2, inf).
    _, _, st = _state(seed=7)
    lower = [PetzRenyi(a).matrix(st) for a in (0.1, 0.3, 0.5)]
    upper = [AlphaZ(a, 1.0, check_monotone=False).matrix(st) for a in (0.5, 0.7, 1.0, 1.5, 2.0)]
    for a, b in zip(lower, lower[1:]):
        assert _psd(a - b)  # decreasing
    for a, b in zip(upper, upper[1:]):
        assert _psd(b - a)  # increasing


def test_diagonal_weight_is_one_over_population():
    # zeta(x, x) = 1/x for every (alpha, z) -- the classical Fisher-information limit.
    from qbm.metrics import alpha_z_weight

    p = np.array([0.25, 0.25])
    for a, z in [(0.3, 0.7), (2.0, 1.5), (0.5, 3.0)]:
        W = alpha_z_weight(p, a, z)
        assert np.allclose(W, 4.0, atol=1e-10)


# ---------------------------------------------------------------------------
# data-processing region, custom kernels, and optimizer integration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alpha, z, expected",
    [
        (0.5, 1.0, True),  # 0<a<1, z >= max(a, 1-a)
        (0.5, 0.5, True),  # z = 1/2 = max(0.5, 0.5)
        (0.3, 0.4, False),  # z < 1 - alpha
        (2.0, 1.5, True),  # a>1: a-1 <= z <= a <= 2z
        (2.0, 0.5, False),  # z < a - 1
        (3.0, 1.0, False),  # a > 2z
    ],
)
def test_data_processing_region(alpha, z, expected):
    assert alpha_z_is_monotone(alpha, z) is expected


def test_warns_outside_the_data_processing_region():
    with pytest.warns(RuntimeWarning, match="data-processing"):
        AlphaZ(0.3, 0.4)
    # ...and stays silent inside it
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AlphaZ(0.5, 1.0)


def test_custom_metric_accepts_an_arbitrary_kernel():
    _, _, st = _state(seed=8)
    # a user-invented kernel: harmonic mean
    harmonic = CustomMetric(lambda x, y: (x + y) / (2 * x * y), "harmonic")
    g = harmonic.matrix(st)
    assert g.shape == (st.ham.n_params,) * 2
    assert np.allclose(g, g.T, atol=1e-9)
    assert _psd(g)


def test_natural_gradient_works_with_a_parameterized_metric():
    H = qbm.hamiltonians.tfim(3, g=1.2)
    model = qbm.FullyVisibleQBM(n=3)
    model.theta = np.random.default_rng(0).normal(scale=0.05, size=model.n_params)
    qbm.fit(
        model,
        qbm.losses.Energy(H),
        qbm.optim.NaturalGradient(metric=SandwichedRenyi(0.5), lr=0.2, reg=1e-3),
        steps=200,
    )
    assert model.energy(H) - qbm.oracles.ground_energy(H) < 1e-2


def test_alpha_z_available_on_every_backend():
    ham, theta, _ = _state(seed=9)
    metric = AlphaZ(0.6, 1.0)
    dense = metric.matrix(qbm.DenseBackend().thermal_state(ham, theta))
    sv = metric.matrix(qbm.StatevectorBackend().thermal_state(ham, theta))
    assert np.allclose(dense, sv, atol=1e-9)
