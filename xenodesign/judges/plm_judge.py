"""Adversarial pLM judge: ESM-2 pseudo-log-likelihood for design naturalness.

The adversarial framing
-----------------------
ESM-2 is a protein language model trained on hundreds of millions of natural L-protein
sequences.  It implicitly encodes the full evolutionary prior over sequence space: what
residue combinations are biologically plausible, structurally stable, and evolutionarily
sensible.  We fix ESM-2 as a *discriminator / adversary* — the design loop must produce
sequences that fool it into assigning high pseudo-log-likelihood (PLL).

High PLL = "ESM-2 thinks this sequence is natural" = the sequence is in-manifold.
Low PLL  = out-of-distribution garbage.  LigandMPNN's L-bias corrupts chirality AND
tends to produce sequences that are inconsistent with the target interface context;
the PLL term catches both.

Pseudo-log-likelihood (masked-marginal)
---------------------------------------
For sequence s = (s_1, ..., s_L):

    PLL(s) = (1/L) * Σ_i log p(s_i | s_{-i}; θ)

where p(s_i | s_{-i}; θ) is ESM-2's probability for residue i given all other residues
masked.  This is the standard masked-marginal PLL (Salazar et al. 2022; Meier et al. 2021).
Note: ESM-2 is sequence-only — it is chirality-agnostic.  The PLL measures *sequence*
plausibility as an L-protein; we apply it to the *one-letter L equivalent* of the
designed D-sequence.  Correct division of labour: the chirality referee handles stereochemistry
separately.

Usage
-----
    judge = ESMPseudoLogLikelihood()          # loads ESM-2 lazily on first call
    pll   = judge("ACDEFGHIKLMNPQRSTVWY")      # float, nats per residue

Injection for testing
---------------------
    def mock_pll(seq): return -1.0
    judge = ESMPseudoLogLikelihood(model_fn=mock_pll)
    pll   = judge("ACDE")                      # returns -1.0 (mock)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Reasonably sized ESM-2 (650M params).  Large enough to have meaningful evolutionary
# knowledge; small enough to fit in ~3 GB GPU RAM alongside other tasks.
_DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

# Standard 20 amino-acid alphabet (one-letter).
_AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


class ESMPseudoLogLikelihood:
    """Adversarial discriminator: masked-marginal PLL via ESM-2.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.  Default: ``facebook/esm2_t33_650M_UR50D``.
    device:
        Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).  If ``None``, auto-selects
        CUDA when available, else CPU.
    model_fn:
        Optional callable ``(sequence: str) -> float``.  When provided, bypasses the
        real ESM model entirely (used for unit testing without torch/HF).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        model_fn: Optional[Callable[[str], float]] = None,
    ):
        self._model_id = model_id
        self._device_str = device
        self._model_fn = model_fn  # injected mock; overrides real ESM
        # Lazy-loaded ESM state (None until first call to the real model).
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """Lazy-load ESM-2 + tokenizer from HuggingFace (or local cache).

        Called on first real PLL computation.  Torch + transformers are NOT imported at
        module level so the module is importable without those packages.
        """
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if self._device_str is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)

        logger.info("Loading ESM-2 from %s onto %s …", self._model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(self._model_id)
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("ESM-2 loaded.")

    def _compute_pll(self, sequence: str) -> float:
        """Masked-marginal pseudo-log-likelihood, averaged over positions.

        For each position i, mask it, run the model, read log-prob of the true residue.
        PLL = mean over all positions.  Units: nats per residue.
        """
        import torch
        import torch.nn.functional as F

        seq = sequence.upper()
        if not seq:
            raise ValueError("sequence must be non-empty")

        # Encode once to get token ids; then mask each position in turn.
        enc = self._tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self._device)   # (1, L+2) — BOS/EOS tokens
        attention_mask = enc["attention_mask"].to(self._device)

        # Token ids for the sequence residues (strip BOS/EOS).
        seq_token_ids = input_ids[0, 1:-1]   # (L,)
        L = len(seq_token_ids)

        mask_token_id = self._tokenizer.mask_token_id

        log_probs_sum = 0.0
        with torch.no_grad():
            for i in range(L):
                masked_ids = input_ids.clone()
                masked_ids[0, i + 1] = mask_token_id   # +1 for BOS offset
                logits = self._model(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                ).logits  # (1, L+2, vocab)
                # Log-softmax over vocab at position i+1.
                log_prob_i = F.log_softmax(logits[0, i + 1], dim=-1)
                true_token_id = seq_token_ids[i].item()
                log_probs_sum += log_prob_i[true_token_id].item()

        return log_probs_sum / L

    def __call__(self, sequence: str) -> float:
        """Return the masked-marginal PLL for ``sequence``.

        Strips non-standard residues from the sequence before scoring (replaces
        with Ala as a neutral proxy) to handle glycine-substituted sequences from
        the loop.

        Parameters
        ----------
        sequence:
            One-letter L amino-acid sequence.

        Returns
        -------
        float
            Average log-probability per residue (nats).  Higher = more natural.
        """
        # Sanitise: uppercase first, then replace non-standard characters with Ala.
        sanitised = "".join(aa.upper() if aa.upper() in _AA_ALPHABET else "A" for aa in sequence)
        n_substituted = sum(1 for aa in sequence if aa.upper() not in _AA_ALPHABET)
        if n_substituted:
            logger.warning(
                "ESMPseudoLogLikelihood: %d/%d residues are non-standard and were mapped to Ala "
                "before PLL scoring. If you passed a D-CCD string (e.g. '(DAL)') instead of the "
                "one-letter L equivalent, PLL will be meaningless.",
                n_substituted, len(sequence),
            )
        if not sanitised:
            return 0.0

        # Injected mock: bypass real model.
        if self._model_fn is not None:
            return self._model_fn(sanitised)

        # Lazy-load.
        if self._model is None:
            self._load_model()

        return self._compute_pll(sanitised)
