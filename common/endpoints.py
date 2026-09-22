"""The two binary endpoints, defined once.

Both are derived from the ``targets`` array of the preprocessed extract, whose
last columns are ``[..., baseline_eGFR, ..., horizon_eGFR, CKD_stage]``.

``ckd-stage``    -- the CKD stage at the horizon is above the baseline band.
``egfr-decline`` -- horizon eGFR has fallen to <= 70% of baseline.

The two are not interchangeable: they label different slices positive and give
different models, so every driver requires the endpoint to be named.
"""

import numpy as np


def multi2binary(labels):
    return np.where(labels > 0, 1, 0)


def stage2progress(target, baseline):
    return np.where(target/baseline <= 0.7, 1, 0)


#: Binary endpoints --binary can collapse the multi-class stage label to.
TARGETS = {
    'ckd-stage': lambda d: multi2binary(d['targets'][:, -1]),
    'egfr-decline': lambda d: stage2progress(d['targets'][:, -2],
                                             d['targets'][:, -4]),
}


def resolve_target_type(args):
    """Return the binary endpoint to train against.

    There is deliberately no default: the two endpoints look superficially
    alike but produce different models, and picking the wrong one silently is
    exactly the mistake this guard exists to prevent.
    """
    if getattr(args, 'target_type', None):
        return args.target_type

    config = getattr(args, 'config', None)
    if config:
        import yaml
        with open(config) as fh:
            cfg = yaml.safe_load(fh) or {}
        value = (cfg.get('exp_setting') or {}).get('target_type')
        if value in TARGETS:
            return value
        raise SystemExit(
            f"{config} has exp_setting.target_type={value!r}; expected one "
            f"of {sorted(TARGETS)}.")

    raise SystemExit(
        '--binary needs the endpoint named. Pass --target_type '
        '{ckd-stage|egfr-decline}.\n'
        '  ckd-stage    CKD stage at the horizon is above baseline\n'
        '  egfr-decline horizon eGFR <= 70% of baseline')


def binarise(data, target_type):
    """Collapse the multi-class stage label to the requested binary endpoint."""
    data['targets'][:, -1] = TARGETS[target_type](data)
