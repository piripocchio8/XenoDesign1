import pytest
from xenodesign.scorer import design_score, select_topk


def test_design_score_rewards_confidence_penalizes_chirality():
    good = design_score(iptm=0.9, interface_plddt=90.0, chirality_violation_frac=0.0)
    bad = design_score(iptm=0.9, interface_plddt=90.0, chirality_violation_frac=0.5)
    assert good > bad


def test_design_score_monotonic_in_iptm():
    lo = design_score(iptm=0.5, interface_plddt=80.0, chirality_violation_frac=0.0)
    hi = design_score(iptm=0.8, interface_plddt=80.0, chirality_violation_frac=0.0)
    assert hi > lo


def test_select_topk_returns_highest_fraction():
    items = [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7)]
    top = select_topk(items, key=lambda x: x[1], frac=0.5)
    assert [name for name, _ in top] == ["b", "d"]


def test_select_topk_always_returns_at_least_one():
    items = [("a", 0.1), ("b", 0.9)]
    top = select_topk(items, key=lambda x: x[1], frac=0.01)
    assert len(top) == 1 and top[0][0] == "b"


# Optional SS / contact terms in design_score — default-OFF (weight 0.0)

def test_design_score_unchanged_when_new_terms_omitted():
    base = 0.5 * 0.9 + 0.5 * (90.0 / 100.0) - 1.0 * 0.0
    assert design_score(iptm=0.9, interface_plddt=90.0,
                        chirality_violation_frac=0.0) == pytest.approx(base)


def test_design_score_new_terms_default_weight_zero_is_noop():
    without = design_score(iptm=0.7, interface_plddt=80.0, chirality_violation_frac=0.1)
    with_metrics_no_weight = design_score(
        iptm=0.7, interface_plddt=80.0, chirality_violation_frac=0.1,
        helix_frac=0.9, num_intra_contacts=50, num_inter_contacts=12,
    )
    assert with_metrics_no_weight == pytest.approx(without)


def test_design_score_helix_weight_rewards_helix():
    lo = design_score(iptm=0.7, interface_plddt=80.0, chirality_violation_frac=0.0,
                      helix_frac=0.1, helix_weight=0.5)
    hi = design_score(iptm=0.7, interface_plddt=80.0, chirality_violation_frac=0.0,
                      helix_frac=0.9, helix_weight=0.5)
    assert hi > lo
    assert (hi - lo) == pytest.approx(0.5 * (0.9 - 0.1))


def test_design_score_contact_terms_saturate_at_target():
    half = design_score(iptm=0.5, interface_plddt=50.0, chirality_violation_frac=0.0,
                        num_inter_contacts=5, target_inter_contacts=10, inter_contact_weight=0.4)
    full = design_score(iptm=0.5, interface_plddt=50.0, chirality_violation_frac=0.0,
                        num_inter_contacts=10, target_inter_contacts=10, inter_contact_weight=0.4)
    over = design_score(iptm=0.5, interface_plddt=50.0, chirality_violation_frac=0.0,
                        num_inter_contacts=99, target_inter_contacts=10, inter_contact_weight=0.4)
    assert full > half
    assert full == pytest.approx(over)
    assert (full - half) == pytest.approx(0.4 * (1.0 - 0.5))


def test_design_score_intra_contact_term():
    lo = design_score(iptm=0.5, interface_plddt=50.0, chirality_violation_frac=0.0,
                      num_intra_contacts=10, target_intra_contacts=40, intra_contact_weight=0.3)
    hi = design_score(iptm=0.5, interface_plddt=50.0, chirality_violation_frac=0.0,
                      num_intra_contacts=40, target_intra_contacts=40, intra_contact_weight=0.3)
    assert hi > lo


# Per-case SS-bias term (#21): helix-reward (alpha) vs anti-alpha penalty
from xenodesign.scorer import SSBiasConfig, ss_bias_score


def test_ss_bias_default_neutral():
    cfg = SSBiasConfig()
    assert ss_bias_score(0.0, cfg) == 0.0
    assert ss_bias_score(1.0, cfg) == 0.0


def test_ss_bias_helix_reward_alpha_case():
    cfg = SSBiasConfig(target_helix_frac=1.0, weight=0.5)
    assert ss_bias_score(0.9, cfg) > ss_bias_score(0.2, cfg)


def test_ss_bias_anti_alpha_penalty_case():
    cfg = SSBiasConfig(target_helix_frac=0.0, weight=0.5)
    assert ss_bias_score(0.2, cfg) > ss_bias_score(0.9, cfg)
    assert ss_bias_score(1.0, cfg) < ss_bias_score(0.0, cfg)


def test_ss_bias_score_is_a_proximity_reward():
    cfg = SSBiasConfig(target_helix_frac=0.0, weight=1.0)
    assert ss_bias_score(0.0, cfg) == pytest.approx(1.0)
    assert ss_bias_score(1.0, cfg) == pytest.approx(0.0)
    cfg2 = SSBiasConfig(target_helix_frac=1.0, weight=1.0)
    assert ss_bias_score(1.0, cfg2) == pytest.approx(1.0)
    assert ss_bias_score(0.0, cfg2) == pytest.approx(0.0)


def test_ss_bias_usable_as_selection_key():
    cfg = SSBiasConfig(target_helix_frac=0.0, weight=1.0)   # anti-alpha
    candidates = [("helical", 0.9), ("mixed", 0.5), ("loopy", 0.1)]
    best = max(candidates, key=lambda c: ss_bias_score(c[1], cfg))
    assert best[0] == "loopy"


# --- P1a: MultiCandidate real key_fn — sequence_quality_key (the de-gaming re-rank lever) ---
from xenodesign.scorer import sequence_complexity, sequence_quality_key


def test_sequence_complexity_bounds():
    assert sequence_complexity("") == 0.0
    assert sequence_complexity("AAAAAAAA") == 0.0          # homopolymer -> 0
    diverse = "ACDEFGHIKLMNPQRSTVWY"                       # 20 distinct -> max entropy
    assert sequence_complexity(diverse) == pytest.approx(1.0, abs=1e-9)
    assert sequence_complexity("ACDEFGAA") < sequence_complexity(diverse)


def test_sequence_quality_key_prefers_diverse_over_polyala():
    diverse = "SLLNRTFARKGIEELIEEKLV"      # a real-shaped 21-mer
    polyala = "AAAAAAAAAAAAAAAAAAAAA"
    assert sequence_quality_key(diverse) > sequence_quality_key(polyala)
    assert sequence_quality_key(polyala) < 0.0             # penalties push it negative
    assert sequence_quality_key(diverse) >= 0.0            # clean diverse -> its entropy


def test_sequence_quality_key_penalizes_homopolymer_run_and_empty():
    base = "SLLNRTFARKDEIWQYHCMV"          # 20 distinct, no run, no Ala/Gly flood
    runny = "SLLNRTFRKDEIWAAAAHCMV"        # a >=4 Ala run injected
    assert sequence_quality_key(runny) < sequence_quality_key(base)
    assert sequence_quality_key("") == -1.0


def test_sequence_quality_key_as_multicandidate_key():
    from xenodesign.inverse_folding import MultiCandidate

    pool = ["AAAAAAAAAA", "SLLNRTFARK", "GGGGGGGGGG"]

    def backend(db, cc, ce, fm, t, n):
        return list(pool[:n])

    mc = MultiCandidate(backend, num_seqs=3, key_fn=sequence_quality_key)
    assert mc(None, [], [], [False], 0.1, 1) == ["SLLNRTFARK"]
