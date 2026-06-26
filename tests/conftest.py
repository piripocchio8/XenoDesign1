import numpy as np

# Standard ideal L-alanine backbone+CB frame (Engh & Huber-style ideal residue).
IDEAL_L_ALA = {
    "N": np.array([-0.525, 1.363, 0.000]),
    "CA": np.array([0.000, 0.000, 0.000]),
    "C": np.array([1.526, 0.000, 0.000]),
    "CB": np.array([-0.529, -0.774, -1.205]),
}
