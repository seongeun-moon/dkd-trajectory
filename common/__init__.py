"""Code shared by the pipeline stages.

``preprocessing/``, ``het_trans/``, ``baselines/`` and ``analysis/`` are run
independently, each from its own directory, so they add the repository root to
``sys.path`` before importing from here.
"""
