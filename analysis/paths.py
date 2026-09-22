"""Restricted-data and result locations for the analysis scripts.

The EHR extract, the derived caches and the trained checkpoints are restricted
and are not distributed with this code. Every location comes from the command
line or from an environment variable.

Two interfaces, on purpose:

* the module-level constants (``DATA_ROOT`` and friends) for the scripts that
  read a path and fail on their own if it is wrong;
* the resolver functions (``data_root()`` and friends), which validate and fail
  with an actionable message, for the scripts that take ``--data-root`` style
  overrides.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- plain constants ---------------------------------------------------------

#: Hospital CSV/XLSX export tree.
RAW_ROOT = os.environ.get('CKD_RAW_ROOT', './data/raw')

#: Preprocessed tensors (reduced24_*.pkl, key_info.pkl).
DATA_ROOT = os.environ.get('CKD_DATA_ROOT', './data/preprocessed')

#: Training run directories, one per session.
RESULT_ROOT = os.environ.get('CKD_RESULT_ROOT', './result')

#: Writable directory for anything these scripts emit.
OUT_ROOT = os.environ.get('CKD_WORK_DIR', './work')

# --- validating resolvers ----------------------------------------------------

_HELP = {
    'CKD_DATA_ROOT': 'the directory holding the preprocessed reduced24_*.pkl '
                     'and key_info.pkl files',
    'CKD_CKPT_ROOT': 'the directory holding the archived transformer runs '
                     '(<run_name>/best_valid_loss_model.pt)',
    'CKD_RAW_RESULTS': 'the directory holding the archived per-model per-seed '
                       'prediction CSVs',
}


class DataNotConfigured(SystemExit):
    """Raised when a restricted-data location has not been supplied."""


def _resolve(env_var, override=None):
    raw = override or os.environ.get(env_var)
    if not raw:
        raise DataNotConfigured(
            f"\n{env_var} is not set.\n"
            f"  Point it at {_HELP[env_var]}, e.g.\n"
            f"    export {env_var}=/path/to/data\n")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise DataNotConfigured(f"\n{env_var} points at a missing directory: {path}\n")
    return path


def data_root(override=None):
    """Directory of the preprocessed dataset."""
    return _resolve('CKD_DATA_ROOT', override)


def ckpt_root(override=None):
    """Directory of the archived transformer checkpoints."""
    return _resolve('CKD_CKPT_ROOT', override)


def raw_results_root(override=None):
    """Directory of the archived per-model per-seed prediction CSVs."""
    return _resolve('CKD_RAW_RESULTS', override)


def work_dir(override=None):
    """Writable output directory, created on demand."""
    raw = override or os.environ.get('CKD_WORK_DIR') or (REPO_ROOT / 'work')
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_data_arguments(parser, *, data=False, ckpt=False, raw_results=False):
    """Attach the standard restricted-data path overrides to an ArgumentParser."""
    if data:
        parser.add_argument(
            '--data-root', default=None,
            help='Preprocessed dataset directory (default: $CKD_DATA_ROOT).')
    if ckpt:
        parser.add_argument(
            '--ckpt-root', default=None,
            help='Archived transformer checkpoint directory '
                 '(default: $CKD_CKPT_ROOT).')
    if raw_results:
        parser.add_argument(
            '--raw-results', default=None,
            help='Archived prediction-CSV directory (default: $CKD_RAW_RESULTS).')
    parser.add_argument(
        '--out-dir', default=None,
        help='Output directory for derived artefacts '
             '(default: $CKD_WORK_DIR or ./work).')
    return parser
