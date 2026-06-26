"""Network test: PepMLM target-conditioned seed generation (downloads weights).

Run locally:  pytest tests/gpu/test_pepmlm_network.py -m network -v

Validates the BEST-EFFORT PepMLMSeedGenerator._real_generate (simplified single-pass
masked-fill). For the exact PepMLM protocol see https://github.com/programmablebio/pepmlm.
"""
import pytest

from tests.gpu.conftest import require_transformers


@pytest.mark.network
def test_pepmlm_generates_valid_peptide():
    require_transformers()
    from xenodesign.seed import AA_ALPHABET, PepMLMSeedGenerator

    target = "GSHMKVLITGGAGFIGSHLVDRLMERGHEVVVLDNL"
    gen = PepMLMSeedGenerator(reverse=True)  # real PepMLM (no injected fn)
    peptide = gen.generate(target_seq=target, length=12)

    assert 1 <= len(peptide) <= 12
    assert set(peptide) <= set(AA_ALPHABET)


@pytest.mark.network
def test_build_alpha_seed_with_real_pepmlm():
    """End-to-end: the per-case dispatcher drives the REAL PepMLM seed (no fake) for the
    alpha policy (len 21, retro-inverso). Uses a public target (not the gitignored FASTA)."""
    require_transformers()
    from xenodesign.seed import AA_ALPHABET, PepMLMSeedGenerator, SeedResult
    from xenodesign.benchmark.cases import get_case
    from xenodesign.benchmark.seeding import build_seed_for_case

    target = "GSHMKVLITGGAGFIGSHLVDRLMERGHEVVVLDNL"  # public; not the alpha FASTA
    res = build_seed_for_case(
        get_case("alpha"),
        generator=PepMLMSeedGenerator(reverse=True),  # real model, no injected fn
        target_seq=target,
    )
    assert isinstance(res, SeedResult)
    assert res.conditioned is True and res.reverse_applied is True
    assert res.length == 21
    assert 1 <= len(res.one_letter) <= 21
    assert set(res.one_letter) <= set(AA_ALPHABET)
