"""ABC mixed-chirality designer package.

Pure-Python / numpy ABC engine + adapters whose fitness oracle is a calibrated
low-diffusion-step Chai cycle plus the parity-aware / intramolecular objective.
Heavy imports (torch / chai / gemmi / LigandMPNN) stay deferred to call time so
every module here imports CPU-clean.
"""
