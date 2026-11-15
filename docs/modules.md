# Our module in plain language

## Spatial analysis: where in the head is the information?
Every electrode on the scalp hears many brain sources at once, because electrical activity spreads through the skull ("volume conduction"). Two electrodes far apart can therefore record almost the same signal.

A **spatial filter** adds and subtracts electrodes to build a "virtual electrode" that listens to one source and ignores the rest.
- **CSP** (Common Spatial Patterns) finds the filters whose output power differs most between two imagined movements. In 2-D you can literally rotate the filter by hand and watch the two classes' power change; CSP finds the best angle.
- **Patterns** tell you where on the head the information comes from. **Filters** are only the recipe for extracting it, which is why we always draw patterns on the head.
- **Riemannian methods** skip filters and compare whole covariance matrices, using a geometry made for them (no "swelling" when averaging).
- **xDAWN**, **PCA**, **whitening**, **CAR** and the **Laplacian** are other spatial methods you can try on the same data.

## Classification: can we read the intention?
A classifier turns each trial into a score and compares it with a threshold. Moving the threshold trades hits for false alarms; the ROC curve shows all possible trade-offs at once.

We only trust results on trials the classifier has **never seen**. **Cross-validation** tests every trial exactly once with a model trained on the others. Everything that learns from data (CSP, scaling, feature selection, tuning) is redone inside each training fold. Otherwise information leaks from the test trials, and even pure noise looks like a brain signal.

Accuracy is meaningless without **chance level**. With 40 test trials, a coin-flipping classifier still reaches 62.5% one time in twenty. The app always shows the binomial threshold and a permutation test next to the score. It also compares pipelines with corrected statistics, so a 2% difference is not mistaken for progress.
