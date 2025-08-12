# Rocket League Match Winner Predictor

## Project Overview
This project builds and evaluates a **machine learning model** to predict the winner of Rocket League matches using **historical team performance data**.  

Given **per-team match statistics** (e.g., shots, saves, boosts, positioning) and match metadata (e.g., event, date, region), the model learns patterns from **past matches** to estimate the probability of each team winning a future matchup.

---

## Data Sources
1. **`matches_by_teams.csv`**  
   - Contains per-team stats for each match.  
   - Example stats: shots, goals, saves, assists, boost usage, positioning time, demolitions.  
   - Includes a `winner` column (1 if that team won the match, 0 otherwise).

2. **`main.csv`**  
   - Contains match metadata and timing information.  
   - Includes event name, region, tier, stage, LAN/qualifier flags, and most importantly a **date column** (`game_date` or `match_date`).

---

## Methodology

### 1. **No Data Leakage**
We ensure the model never “sees the future” by:
- Computing **rolling averages** of each stat using only matches **before** the one being predicted.
- Using `shift(1)` before the rolling mean so the current match's stats are never included in the features.

### 2. **Rolling Feature Engineering**
For each team:
- Take the mean of the last **N=5 matches** for each numeric stat.
- This represents the team’s **recent form** heading into the match.

### 3. **Match-Level Feature Diffs**
Convert the problem into a single row per match by:
- Sorting the two teams alphabetically.
- Computing `feature_diff = (Team A’s rolling stats) − (Team B’s rolling stats)`.
- Setting the label: `1` if Team A won, `0` if Team B won.

**Why diffs?**  
This gives the model a **directional measure of relative strength** instead of two separate feature sets, simplifying the problem.

### 4. **Time-Based Train/Test Split**
Split the dataset chronologically:
- **Train**: earliest 80% of matches  
- **Test**: most recent 20% of matches  
This mimics real-world forecasting where we predict future matches from past ones.

### 5. **Models**
Train and compare:
- **Logistic Regression** (with feature scaling) - interpretable baseline.
- **Random Forest Classifier** - captures nonlinear relationships without manual feature interaction.

### 6. **Evaluation**
Metrics:
- **Accuracy** - % of correct predictions.
- **AUC** - model’s ability to rank winners higher than losers.
- **Log Loss** - penalizes over-confident wrong predictions.

Visualizations:
- ROC Curves for both models.
- Confusion Matrix for the better model.

---

## Saving & Reusing the Model
After training:
- Save the model and its feature schema to `rl_match_predictor.pkl` using `pickle`.
- Save test predictions to `test_set_predictions.csv`.

This allows for:
- **Reloading the trained model** without retraining.
- Making predictions for any two teams as of a given date.

---

## Inference Helper
I provided a `predict_match()` function that:
- Takes two team names and an optional `as_of` date.
- Looks up the teams’ most recent rolling stats before that date.
- Returns a **human-friendly prediction**:

Example:
```python
predict_match("G2 Esports", "NRG")
# Output: G2 Esports is predicted to win with 68.3% confidence.
```

---

## Future Improvements
- **Add Elo ratings**  
  Implement rating systems to dynamically track team strength and incorporate them as features.

- **Include categorical context**  
  Encode features like event region, LAN vs. online, and stage to capture matchup conditions.

- **Experiment with rolling window sizes**  
  Test different match history lengths (e.g., 3, 10 matches) or use **exponential decay** weighting to emphasize recent games.

- **Model stacking or blending**  
  Combine multiple models (Logistic Regression, Random Forest, Gradient Boosting) for better accuracy.

- **Deploy as an application**  
  Build a simple web interface or API to make predictions accessible without running the notebook.

- **Add automated retraining**  
  Set up a pipeline to retrain the model as new matches are added to the dataset.
  
---

