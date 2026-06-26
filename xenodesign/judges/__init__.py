"""Adversarial pLM-judge panel for chirality-aware, naturalness-aware design selection.

Architecture
-----------
The design loop plays an *adversarial game* against a frozen protein language model
(ESM-2).  Each iteration proposes a new sequence; the panel of referees scores it and
can veto it:

  Chirality referee  — veto if D-chirality violation fraction > 0.1; score = 1 − viol.
  pLM naturalness    — masked-marginal ESM-2 pseudo-log-likelihood (the adversary).
                       The design "wins" by producing sequences the LM finds plausible.
  Binding referee    — ipTM (+ optional interface pLDDT).
  Mirror consistency — optional; mirror-discrepancy penalty.

``JudgePanel.select(history)`` picks the non-vetoed step maximising the weighted
composite.  Chirality drift that LigandMPNN gradually introduces is killed by the
chirality veto; the naturalness term pushes designs toward foldable, in-manifold
sequences.
"""
from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
from xenodesign.judges.panel import JudgePanel, RefereeScore, PanelResult

__all__ = ["ESMPseudoLogLikelihood", "JudgePanel", "RefereeScore", "PanelResult"]
