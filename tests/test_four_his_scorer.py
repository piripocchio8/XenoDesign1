"""S3a.4 (#5): four_his_tetrahedron_score rewards a clean 4-coordinate ~109.5-deg site and penalises
2-His / open / wide-angle geometry. Pure function over zn_coordination_geometry's dict."""
from __future__ import annotations

from xenodesign.classes.cyclic import four_his_tetrahedron_score


def test_perfect_tetrahedron_scores_high():
    geom = {"n_coordinating": 4, "mean_zn_n_distance": 2.05, "max_zn_n_distance": 2.1,
            "mean_n_zn_n_angle": 109.47, "ideal_tetrahedral": 109.47}
    assert four_his_tetrahedron_score(geom) > 0.9


def test_two_his_scores_low():
    geom = {"n_coordinating": 2, "mean_zn_n_distance": 2.05, "max_zn_n_distance": 2.1,
            "mean_n_zn_n_angle": 95.0, "ideal_tetrahedral": 109.47}
    assert four_his_tetrahedron_score(geom) < 0.6


def test_empty_site_scores_zero():
    geom = {"n_coordinating": 0, "mean_zn_n_distance": None, "max_zn_n_distance": None,
            "mean_n_zn_n_angle": None, "ideal_tetrahedral": 109.47}
    assert four_his_tetrahedron_score(geom) == 0.0
