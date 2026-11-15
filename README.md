# BCI Master Studio: Spatial Analysis + Classification

Our team's module for the course BCI app: **spatial analysis** (where in the head is the information?) and **classification** (can we read the intention, honestly?).
Both pages pair **concept demos** (small interactive 2-D models that show the idea) with **real EEG**, linked plots, explanations with self-check questions, and full code/parameter export.

```bash
pip install -r requirements.txt && streamlit run app.py
```

The app opens at http://localhost:8501 on the Spatial analysis page. A synthetic motor-imagery recording loads instantly; no download is needed.

> **Scope.** Data loading, filtering, ICA, epoching, time-frequency, features and paradigms are built by the other teams.
> Here they are replaced by one deliberately minimal **Setup** page (dataset, band-pass, notch, epoch window), so our two pages always have data to work on.

---

## 1 · User guide

The sidebar has three pages:

| page | what it is |
|---|---|
| **Setup** (optional) | Pick a sample dataset or upload a file; set high-pass, low-pass, notch and the epoch window; quick-look spectrum. |
| **Spatial analysis** (part 1 of 2) | Concepts and real-data spatial filtering. |
| **Classification** (part 2 of 2) | Concepts, then build, evaluate, understand and compare classifiers. |

On every page:
- **Apply** is at the top right, with a status chip ("Default settings, not applied", "Unapplied changes", "In pipeline").
- **Undo / Reset** and the **random seed** are in the sidebar.
- The **dataset menu** (top right) switches data on any page.

### Spatial analysis
The page has two modes: **Concepts** and **Your data**.

**Concepts.** Each demo is live and comes with a "what to notice" note.

| demo | idea it teaches |
|---|---|
| Why spatial filters? | Volume conduction: two independent sources become correlated electrode signals as their spread grows. |
| Spin a filter | A spatial filter is a direction. Rotate it and watch each class's power; click the curve to set the angle; "Jump to CSP" snaps to the eigenvector. |
| Whiten, then rotate | How CSP is computed: whitening turns the pooled covariance into a circle, then a rotation aligns the axes with the class differences. |
| Filters vs patterns | Why filters put weight on noise channels and patterns show where the signal is (Haufe 2014). |
| Riemann in 2×2 | Covariance matrices as shapes; the Euclidean average "swells", the Riemannian (geodesic) average keeps the size. |

**Your data.** Choose a method (CSP · Riemann · xDAWN · PCA · Whitening · CAR · Laplacian) and adjust the control bar (band, time window, components, covariance shrinkage, % of trials, class pair). The CSP views:

| view | what it shows |
|---|---|
| Power maps | Band power per class and the difference map. Click an electrode to see every trial's power there. |
| Patterns | Eigen-ladder: click a component to see its filter vs pattern topomaps, class distributions, and a plain-language reading in the inspector. |
| Pull apart | Animated morph from electrode space into CSP space, with the Fisher ratio. |
| 3-D head | The pattern painted on a rotating head. |
| Beat CSP | Build your own spatial filter by clicking electrodes and compare it with CSP. |
| Stability | The pattern refit on each CV training fold. |
| Truth | Synthetic data only: the hidden true sources next to the recovered patterns. |

The other methods have their own views:
- **Riemann:** distance plane (click a trial), class means, tangent map.
- **xDAWN:** components ranked by SNR, enhanced ERP.
- **PCA / Whitening:** scree plot, covariance → identity.
- **CAR / Laplacian:** before/after topographies, volume-conduction histogram.

All fits on this page use all trials, which is fine for *looking* and is labelled "Exploratory". **Apply spatial recipe** passes the method and settings to Classification, which **refits them inside every training fold**.

### Classification
A stepper runs **Concepts · Build · Understand · Compete · Leakage guard · Lab log**.

**Concepts**

| demo | idea it teaches |
|---|---|
| Draw the line | LDA projects each trial onto a score. Drag the threshold and watch the decision line, confusion matrix and ROC point move; the AUC stays the same. |
| Chance is not 50% | The binomial accuracy distribution of a guessing classifier, the α-threshold, 1,000 simulated guessers, and "is my accuracy significant?". |
| Too simple, too flexible | kNN on toy data: train vs held-out accuracy as model complexity changes; a decision-region map. |
| Leakage in one picture | Pure noise plus feature selection on all trials looks like a brain signal; selection inside folds stays at chance. |
| What cross-validation does | Step through the folds: each trial is tested once by a model that never saw it. |

**The real-data stages**
- **Build:** recipe cards, Pipeline / Data / Protocol cards, a pipeline strip showing where each block is fitted, a fold map, and the run bar with your bet.
  - Recipes: CSP+LDA, FBCSP, tangent space + LR, MDM, xDAWN, band power + SVM, or a custom feature set (choose the families).
  - Protocols: k-fold, repeated, leave-one-session/subject-out, cross-session; optional nested CV and permutations.
- **Understand:** six views.
  - **Scoreboard:** accuracy, balanced accuracy, κ, F1, AUC and ITR, each as mean ± sd, with a significance badge and your bet.
  - **Chance:** null distribution and binomial threshold; per-fold dots (click one) with ROC and confusion matrix.
  - **Errors:** click a confusion-matrix cell to list its trials, then a trial to see its signal and where its fold's model placed it; the trial barcode shows drift.
  - **Over time:** time-resolved accuracy, learning curve, ITR frontier.
  - **Replay:** a live-BCI cursor driven by out-of-fold probabilities.
  - **Playground** (decision boundaries) and **Inside the model**.
- **Compete:** ≥ 4 pipelines race on identical folds; leaderboard with error bars; Friedman test and Holm-corrected Wilcoxon / Nadeau-Bengio tests.
- **Leakage guard:** what is fitted where for *your* pipeline, plus the real "cheat experiment" on shuffled labels.
- **Lab log:** every run with its parameters, score and bet; restore any configuration.

### Downloads
Both pages end with a **Downloads** panel, plus a code/parameter export for the page:
- **Data:** raw FIF / EDF / CSV, epochs FIF / NPZ.
- **Results:** features CSV / NPY, model (joblib), predictions, fold metrics, metrics JSON (with the null distribution), benchmark, lab log.
- **Figures** (HTML).
- **parameters.json:** re-import it to restore the pipeline.
- **Standalone script and notebook:** they reproduce the app's numbers for the same seed (tested).
- **report.html** and a **ZIP** of everything.

---

## 2 · Developer guide

```
app.py                navigation (Setup · Spatial · Classification), CSS, sidebar, pager
pages/00_setup.py     minimal stand-in for the other teams' modules
pages/01_spatial.py   Spatial analysis (Concepts | Your data)
pages/02_classify.py  Classification (Concepts · Build · Understand · Compete · Leakage guard · Lab log)
concepts_spatial.py   UI of the five spatial concept demos     ┐  one render_*() per demo,
concepts_classify.py  UI of the five classification demos      ┘  each an st.fragment
toys.py               the toy models behind the demos (pure numpy / sklearn, unit-tested)
core.py               data (synthetic + loaders + upload), filter, ICA, epochs, build(params)
spatial.py            CSP (transparent implementation), CAR, Laplacian, PCA/whitening, Riemann helpers
features.py           feature transformers + FilterBankCSP
classify.py           pipelines, CV protocols, evaluate, permutations, chance, benchmark stats, leakage demo
export.py             standalone script / notebook / report / ZIP;  downloads.py  the Downloads panel
viz.py                palette, Plotly template, clickable topomaps, 3-D head
ui.py                 state (apply / undo / reset), header, explain panel, plot wrapper, fragments, error guard
assets/style.css      design system;  static/fonts  IBM Plex (OFL)
tests/                core, spatial, classify, export reproducibility, toys, page smoke tests
```

**Rules**
1. `core.py`, `spatial.py`, `features.py` and `classify.py` import no Streamlit or Plotly: they are embedded verbatim into the generated script. Imports between them end with `# @local-import`.
2. The whole pipeline is one JSON-able dict (`core.default_params()`); `core.build(params, cache)` turns it into objects.
3. Anything that learns from data lives inside an sklearn `Pipeline` from `classify.make_pipeline`, cloned and fitted per training fold.
4. Interactive areas are `@ui.guarded_fragment("name")` functions, so a click reruns only that area.
   - Linked plots use `ui.plot(fig, key, select=True)` + `ui.on_click(event, key)` + `ui.rerun_local()`.
   - Clickable curves need markers: Plotly only selects points.
5. Toy-model maths goes in `toys.py` with a unit test; the UI goes in `concepts_*.py`.

**Add a concept demo:** write the computation in `toys.py` (+ a test in `tests/test_toys.py`), add a `render_x()` fragment in `concepts_spatial.py` or `concepts_classify.py`, and add it to `DEMOS` / `RENDER`. The smoke test loops over `DEMOS` automatically.

**Add a page/module:** create `pages/NN_name.py` using `ui.page_objects()`, `slot = ui.header_row(...)`, `with ui.guard(...)`, `ui.apply_control(slot, ...)`, `ui.explain(...)`, `ui.export_drawer(...)`; register it in `PAGES` in `app.py`; add it to `tests/test_pages_smoke.py`.

**Tests:**
```bash
python -m pytest -q tests
```
They include:
- CSP recovers the hidden ground-truth patterns;
- the toy filter optimum equals the CSP eigenvector;
- whitening gives an identity covariance;
- the Riemannian mean does not swell;
- the binomial null matches the chance levels;
- shuffled labels give chance;
- leakage is detected;
- nested CV never sees test trials;
- the generated script reproduces the app's metrics;
- every page, concept demo and view opens without error.

---

## 3 · Datasets and references
- **Synthetic** motor imagery / P300 (built in, with known ground-truth sources).
- **PhysioNet EEG Motor Movement/Imagery:** Schalk et al. 2004, *IEEE TBME* 51(6); Goldberger et al. 2000.
- **BCI Competition IV 2a** via MOABB: Tangermann et al. 2012, *Front. Neurosci.* 6:55; Jayaram & Barachant 2018, *J. Neural Eng.* 15(6).
- **MNE sample:** Gramfort et al. 2013, *Front. Neurosci.* 7:267.
- **Methods:**
  - Ramoser et al. 2000 (CSP); Ang et al. 2008 (FBCSP);
  - Haufe et al. 2014 (filters vs patterns); Rivet et al. 2009 (xDAWN);
  - Barachant et al. 2012 (Riemannian BCI); Combrisson & Jerbi 2015 (chance levels);
  - Nadeau & Bengio 2003 (corrected t-test); Wolpaw et al. 2002 (ITR); Varoquaux et al. 2017 (CV pitfalls).

## 4 · Assumptions and limitations
- **Scope:**
  - The other teams' modules are reduced to the minimal Setup page.
  - ICA stays in the code (`core.py`) but is off by default and has no page here.
- **Concept demos** use small simulated 2-D data on purpose; they illustrate ideas, not real-data performance.
- **Exploratory plots:** all spatial-analysis plots use all trials and are labelled exploratory. Every number in Classification comes from out-of-fold predictions.
- **Preprocessing:** filtering (and ICA, if enabled) runs on the continuous recording before CV. It never uses labels.
- **Statistics:** fold-based tests are approximate (Nadeau-Bengio correction, Holm adjustment).
- **Deployment model:** it is fitted on all trials for export and interpretation only, and is never reported as performance.
- **Not included:**
  - EEGNet (optional in the brief).
  - The light theme is the only theme.
  - Large datasets download only on request, into `~/mne_data`.

## 5 · AI-use note
This module was developed with substantial help from an AI coding assistant (Anthropic's Claude), working from the team's specification. The team set the requirements, scope and design direction; the assistant wrote most of the code, tests and documentation. All metrics are computed live; none are hardcoded. Correctness is checked by the automated tests listed above. Students should verify claims against the cited literature.

## 6 · Demo
See [docs/demo_script.md](docs/demo_script.md) (10 minutes, 4 speakers) and [docs/modules.md](docs/modules.md) (plain-language explanation).
