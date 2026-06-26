"""GPU test for ChaiBackend.truncated_refine (structure-conditioned diffusion).

Validates HalluDesign §2.1: starting from a perturbed prior, a short refine
(ref_time_steps=50 of 200) should stay close to the input topology (aligned
RMSD < 5 Å) and produce a valid iptm in [0, 1].

The raw (unaligned) RMSD can be large because Chai's diffusion sampler applies
a random rotation+translation augmentation at each step. We use Kabsch
superimposition to align the refined structure onto the base structure before
computing RMSD, which correctly measures topological preservation independent of
the arbitrary coordinate frame produced by the sampler.

Run with:
    pytest tests/gpu/test_truncated_refine_gpu.py -m gpu -v -s
"""
import numpy as np
import pytest

from tests.gpu.conftest import require_chai, require_cuda


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD after optimal superimposition of P onto Q (Kabsch algorithm).

    P, Q: (n, 3) arrays of 3-D coordinates.  Returns the minimal RMSD after
    centering and rotating P to best match Q.
    """
    n = min(len(P), len(Q))
    P, Q = P[:n].copy(), Q[:n].copy()

    # Centre both
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    P -= p_mean
    Q -= q_mean

    # Covariance matrix and SVD
    H = P.T @ Q
    U, _S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_rot = P @ R.T
    diff = P_rot - Q
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


@pytest.mark.gpu
def test_truncated_refine_low_sigma_preserves_topology(tmp_path):
    require_cuda()
    require_chai()
    from xenodesign.backends.chai_backend import ChaiBackend

    entities = [
        {"type": "protein", "name": "target", "sequence": "GSHMKVLITGGAGFIGSHLVDRL", "chirality": "L"},
        {"type": "protein", "name": "binder", "sequence": "ACDEFGHIK", "chirality": "D"},
    ]
    backend = ChaiBackend(device="cuda:0", seed=0)

    # Full prediction to get a base structure
    base = backend.predict(entities, tmp_path / "p0", num_diffn_timesteps=200)

    # Truncated refinement from that structure (only last 50 of 200 steps)
    refined = backend.truncated_refine(
        structure={"entities": entities, "coords": base.coords},
        ref_time_steps=50,
        out_dir=tmp_path / "r1",
    )

    # RMSD after Kabsch superimposition: a short refine should stay near input
    # topology. Raw RMSD is meaningless because chai's random augmentation puts
    # each run in an arbitrary frame; Kabsch alignment makes the comparison fair.
    rmsd = _kabsch_rmsd(refined.coords, base.coords)
    print(f"Kabsch-aligned RMSD base vs refined: {rmsd:.3f} Å")
    print(f"Refined iptm: {refined.iptm:.4f}")
    assert rmsd < 5.0, f"RMSD {rmsd:.3f} Å >= 5.0 Å — refine drifted too far from input topology"
    assert 0.0 <= refined.iptm <= 1.0, f"iptm {refined.iptm} out of range [0, 1]"
