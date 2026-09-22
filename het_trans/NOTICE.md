# Third-party code

Two modules under `model/` are adapted from an external implementation rather
than written by the authors, and are redistributed here unchanged so the model
can be rebuilt:

| File | Origin | License |
|---|---|---|
| `model/performer_attention.py` | [`lucidrains/performer-pytorch`](https://github.com/lucidrains/performer-pytorch) — FAVOR+ attention | MIT |
| `model/reversible.py` | [`lucidrains/performer-pytorch`](https://github.com/lucidrains/performer-pytorch) — reversible residual blocks | MIT |

Both retain their upstream MIT license and must not be modified. The Performer
method itself is:

> Choromanski K, Likhosherstov V, Dohan D, et al. *Rethinking Attention with
> Performers.* ICLR 2021. arXiv:2009.14794

`performer_attention.py` optionally imports `local_attention`,
`axial_positional_embedding` (both MIT, by the same author) and `apex`; these
are ordinary dependencies, not vendored.

Everything else under `het_trans/` — the heterogeneous-input embedding, the
masked-token and eGFR-regression auxiliary heads, the training driver and the
data utilities — is the authors' own work.
