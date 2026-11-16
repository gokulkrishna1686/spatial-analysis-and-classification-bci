# 10-minute demo script: 4 speakers

Speakers: **Gokul Krishna Balaji** (concept demos), **Thirunavukarasu** (spatial analysis on real EEG), **Sanjay K** (classification), **Surya V** (results, honesty, reproducibility). The slide deck's footers name the same speaker for each slide.

**Before the demo:** run `streamlit run app.py` and open http://localhost:8501. Press **Reset** in the sidebar and keep seed 42. Set Permutations to 20 in Classification → Build if you want a faster run.

| time | speaker | screen | what to say / do |
|---|---|---|---|
| 0:00–0:40 | **Gokul** | Spatial · Concepts · *Why spatial filters?* | "Our module is spatial analysis and classification. Each page starts with concept demos, then real EEG." Drag the spread slider to the right: the two sources stay independent, but the C3–C4 correlation climbs to about 0.9. "Electrodes record mixtures. That is why we need spatial filters." |
| 0:40–1:50 | **Gokul** | *Spin a filter* | "A spatial filter is just a direction." Drag θ: the two class powers rise and fall. Click the λ curve near its peak, then press **Jump to CSP**: the diamond marks the optimum. Raise the shared noise: "the best direction learns to subtract what both electrodes share." |
| 1:50–2:30 | **Gokul** | *Whiten, then rotate* | Slide 0 → 1 → 2: "whitening makes the pooled data a circle; a rotation lines the axes up with the class difference. That is CSP." |
| 2:30–3:10 | **Gokul** | *Filters vs patterns*, *Riemann in 2×2* | "The best filter weights a channel that has no signal at all, so we interpret patterns." Riemann: "the ordinary average of two thin shapes is fat (determinant 6.8 vs 1); the Riemannian one is not." |
| 3:10–4:20 | **Thirunavukarasu** | Spatial · **Your data** · Power maps → Patterns | Power maps: "less power over the opposite motor cortex. Click C3: single trials overlap a lot." Patterns: click component 1 on the eigen-ladder; read the inspector sentence aloud ("peaks at C3, drops when imagining the right hand"). |
| 4:20–5:00 | **Thirunavukarasu** | Pull apart → Truth → Apply | Press ▶ (the Fisher ratio rises). Truth: toggle Reveal, the match is about 0.99. Press **Apply spatial recipe**: "Classification will refit this inside every training fold." |
| 5:00–6:00 | **Sanjay** | Classification · Concepts | *Draw the line*: move the threshold and watch the confusion matrix and the ROC dot. *Chance is not 50%*: 40 trials, accuracy 62.5% is **not significant**; raise trials to 200 and the threshold drops. |
| 6:00–6:40 | **Sanjay** | *Too simple, too flexible*, *Leakage in one picture* | k = 1: 100% on training, much lower on held-out. Leakage: noise with 1,000 features gives about 75% when leaky and 50% when honest. |
| 6:40–7:40 | **Sanjay** | Build → Run | CSP + LDA, 5-fold. Show the pipeline strip ("blue tags = fitted on the training fold only") and the fold map. Ask the audience for a bet, then **Run**. Scoreboard: about 80%, *Significant*, p = 0.01; compare with the bet. |
| 7:40–8:40 | **Surya** | Understand → Errors, Replay | Click an off-diagonal confusion cell, then a trial: "this is where its fold's model put it." Over time → Run time analysis → Replay: the cursor flies towards the decoded side. |
| 8:40–9:30 | **Surya** | Compete → Leakage guard | Start the race with 5 pipelines; the leaderboard and corrected p-values: "small differences are not significant." Leakage guard: run the cheat experiment on real data. |
| 9:30–10:00 | **Surya** | Downloads | Open Downloads → Prepare → ZIP. "The script embeds our code and parameters and reproduces these exact numbers; our tests check it." Close. |
