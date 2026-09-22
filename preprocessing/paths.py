"""Restricted-data locations for the preprocessing pipeline.

The EHR extract is restricted and is not distributed with this code. Point
these at your own copy, either by editing this file or through the environment
variables.

Each step imports only the root it reads from:

    01-04  RAW_ROOT    the hospital CSV/XLSX export tree
    05     DATA_ROOT   the preprocessed reduced24_*.pkl tensors
"""

import os

#: Hospital CSV/XLSX export tree.
RAW_ROOT = os.environ.get('CKD_RAW_ROOT', './data/raw')

#: Preprocessed tensors (reduced24_*.pkl, key_info.pkl).
DATA_ROOT = os.environ.get('CKD_DATA_ROOT', './data/preprocessed')

#: Training run directories.
RESULT_ROOT = os.environ.get('CKD_RESULT_ROOT', './result')

#: Writable scratch directory for anything this pipeline emits.
OUT_ROOT = os.environ.get('CKD_WORK_DIR', './work')
