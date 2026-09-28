## Reverse Label Exoplanets

Inverse surrogate for exoplanet atmospheres: predict the 9 atmospheric parameters
(`Kzz, Rp, Tint, C/O, [N/H], [O/H], [S/H], logg, f`) from the 195-bin spectrum.
This inverts the pipeline of [Exoplanets_MF_GPs](https://github.com/camillandreozzi/Exoplanets_MF_GPs).

The model is a GPBoost boosted-tree mean + GP, with mean and covariance estimated jointly,
in single-fidelity (`sf`, HF only) and multi-fidelity (`mf`, AR(1) HF/LF) variants.
It is fit first independently per target, then with coregionalization across the 9 targets.

### Data (`data/`)
| File | Shape | Content |
|---|---|---|
| `XHF_reverse.csv` | 97 × (id + 195) | HF spectra |
| `XLF_reverse.csv` / `XLF10k_reverse.csv` | 97 / 10000 × 195 | LF spectra |
| `YHF_reverse.csv` / `YLF10k_reverse.csv` | 97 / 10000 × 9 | parameters |
| `Observed_Spectra.csv` | 195 × 4 | observed spectrum to invert |

### Layout
```
src/          data loading, preprocessing, metrics, CV splits, model.py, model_coreg.py
eda/          exploratory analysis of targets and spectra
modelling/    independent/, coregionalization/, compare/, plot/
inference/    apply fitted models to the observed spectrum
euler/        cluster job scripts
tests/        unit tests
results/      outputs (gitignored)
```
