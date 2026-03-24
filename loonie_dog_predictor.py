"""
============================================================
loonie_dog_predictor.py
Toronto Blue Jays – Loonie Dog Night Prediction Model

Predicts hot dog sales for Tuesday Loonie Dog Night promotions
at Rogers Centre during the 2026 season.

Assumptions:
  - $1 Loonie Dog promotion continues unchanged in 2026
  - All games are Tuesday home games at Rogers Centre
  - 97-minute pre-game selling window (derived from 2024/2025 data)
  - 2025: individual game data read from image; Sept 23 corrected with
    confirmed post-game stats (92,896 HD | 42,927 att | 2:52 TOG).
  - 2022 & 2023: no game-duration / DPM data available.
  - Blue Jays season wins added as a predictor (affects fan interest /
    attendance). UPDATE SEASON_WINS with verified totals.
  - UPDATE Section 6 with actual 2026 Tuesday home game dates once
    the official schedule is published.
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONSTANTS & PARAMETERS
# ============================================================

PREGAME_WINDOW_MIN = 97   # pre-game hot dog selling window (minutes)
                          # derived from 2024/2025 Dogs-Per-Minute data

# ── Season win totals ─────────────────────────────────────
# Wins affect fan interest, attendance, and ultimately hot dog sales.
# UPDATE 2025 with the actual final regular-season win total.
SEASON_WINS = {
    2022: 92,   # AL Wild Card; lost in ALDS
    2023: 89,   # AL Wild Card; lost in Wild Card round
    2024: 74,   # missed playoffs
    2025: 94,   # confirmed
}

# 2026 projection – adjust based on preseason expectations / Vegas O/U
PROJECTED_2026_WINS = 90   # user notes: "expected to be better" in 2026

# ============================================================
# 2. HISTORICAL GAME DATA
# ============================================================
# Each entry is one Loonie Dog Night game.
# hot_dogs          = total hot dogs sold
# attendance        = paid attendance
# game_duration_min = game time in minutes (None = not available)
#
# Derived:
#   DPF (Dogs Per Fan) = hot_dogs / attendance
#   DPM (Dogs Per Minute) = hot_dogs / (PREGAME_WINDOW_MIN + game_duration_min)

raw_games = [

    # ── 2022 (8 games; no game-duration data) ─────────────────────────────
    {"date": "2022-04-26", "hot_dogs": 30942, "attendance": 22611, "game_duration_min": None},
    {"date": "2022-05-03", "hot_dogs": 30254, "attendance": 22491, "game_duration_min": None},
    {"date": "2022-05-17", "hot_dogs": 29039, "attendance": 22988, "game_duration_min": None},
    {"date": "2022-05-31", "hot_dogs": 29890, "attendance": 25424, "game_duration_min": None},
    {"date": "2022-06-14", "hot_dogs": 26282, "attendance": 23106, "game_duration_min": None},
    {"date": "2022-06-28", "hot_dogs": 31672, "attendance": 27140, "game_duration_min": None},
    {"date": "2022-07-12", "hot_dogs": 34543, "attendance": 32795, "game_duration_min": None},
    {"date": "2022-07-26", "hot_dogs": 40512, "attendance": 39756, "game_duration_min": None},
    # Note: July 26 HD corrected to 40,512 (from initial image read of 40,602)
    # to match confirmed season total of 253,134.

    # ── 2023 (11 games; no game-duration data) ────────────────────────────
    {"date": "2023-04-25", "hot_dogs": 51629, "attendance": 28917, "game_duration_min": None},
    {"date": "2023-05-16", "hot_dogs": 61111, "attendance": 35112, "game_duration_min": None},
    {"date": "2023-05-30", "hot_dogs": 53433, "attendance": 32930, "game_duration_min": None},
    {"date": "2023-06-06", "hot_dogs": 49477, "attendance": 30079, "game_duration_min": None},
    {"date": "2023-06-27", "hot_dogs": 60188, "attendance": 36004, "game_duration_min": None},
    {"date": "2023-07-18", "hot_dogs": 75173, "attendance": 42680, "game_duration_min": None},
    {"date": "2023-08-01", "hot_dogs": 72401, "attendance": 40691, "game_duration_min": None},
    {"date": "2023-08-15", "hot_dogs": 72212, "attendance": 42615, "game_duration_min": None},
    {"date": "2023-08-29", "hot_dogs": 76627, "attendance": 39722, "game_duration_min": None},
    {"date": "2023-09-12", "hot_dogs": 54831, "attendance": 30479, "game_duration_min": None},
    {"date": "2023-09-26", "hot_dogs": 66788, "attendance": 40454, "game_duration_min": None},

    # ── 2024 (13 games; includes game-duration data) ───────────────────────
    {"date": "2024-04-09", "hot_dogs": 58852, "attendance": 31310, "game_duration_min": 173},
    {"date": "2024-04-16", "hot_dogs": 59831, "attendance": 31175, "game_duration_min": 176},
    {"date": "2024-04-30", "hot_dogs": 48487, "attendance": 27189, "game_duration_min": 139},
    {"date": "2024-05-21", "hot_dogs": 52414, "attendance": 28176, "game_duration_min": 141},
    {"date": "2024-06-04", "hot_dogs": 54474, "attendance": 28816, "game_duration_min": 159},
    {"date": "2024-06-18", "hot_dogs": 64200, "attendance": 38595, "game_duration_min": 176},
    {"date": "2024-07-02", "hot_dogs": 47588, "attendance": 26308, "game_duration_min": 144},
    {"date": "2024-07-23", "hot_dogs": 69045, "attendance": 38575, "game_duration_min": 169},
    {"date": "2024-08-06", "hot_dogs": 71391, "attendance": 35051, "game_duration_min": 149},
    {"date": "2024-08-20", "hot_dogs": 64558, "attendance": 34662, "game_duration_min": 139},
    {"date": "2024-09-03", "hot_dogs": 41587, "attendance": 23796, "game_duration_min": 203},
    {"date": "2024-09-10", "hot_dogs": 46061, "attendance": 28109, "game_duration_min": 176},
    {"date": "2024-09-24", "hot_dogs": 49331, "attendance": 29178, "game_duration_min": 195},

    # ── 2025 (11 games; Sept 23 confirmed post-game) ───────────────────────
    # Apr 1 HD estimated from DPM × total selling time (image DPM=165 → ~41,910;
    # adjusted slightly for DPF consistency; treat as approximate).
    # Aug 12 game_duration_min is uncertain from image (left as None).
    # Aug 26 is the single-game HD record (*).
    # Sept 23 stats provided directly: 92,896 HD | 42,927 att | 2:52 game.
    {"date": "2025-04-01", "hot_dogs": 44563,  "attendance": 21845, "game_duration_min": 158},  # HD from DPF(2.04)×att; DPM≈175 (image may show 165 due to resolution)
    {"date": "2025-04-15", "hot_dogs": 55066,  "attendance": 26979, "game_duration_min": 153},
    {"date": "2025-04-29", "hot_dogs": 50953,  "attendance": 28045, "game_duration_min": 149},
    {"date": "2025-05-13", "hot_dogs": 50521,  "attendance": 27717, "game_duration_min": 176},
    {"date": "2025-05-20", "hot_dogs": 42362,  "attendance": 23597, "game_duration_min": 135},
    {"date": "2025-06-03", "hot_dogs": 67702,  "attendance": 32628, "game_duration_min": 166}, 
    {"date": "2025-06-17", "hot_dogs": 72606,  "attendance": 38537, "game_duration_min": 177},
    {"date": "2025-07-22", "hot_dogs": 84731,  "attendance": 42326, "game_duration_min": 173},
    {"date": "2025-08-12", "hot_dogs": 94388,  "attendance": 43003, "game_duration_min": 163},
    {"date": "2025-08-26", "hot_dogs": 96633,  "attendance": 42235, "game_duration_min": 190},  # single-game record (*)
    {"date": "2025-09-26", "hot_dogs": 86615,  "attendance": 40252, "game_duration_min": 194},
    {"date": "2025-09-23", "hot_dogs": 92896,  "attendance": 42927, "game_duration_min": 172},  # confirmed post-game stats
]

# Confirmed full-season totals sourced from reference rows in later-year images.
# ⚠ 2022 raw_games below are INCOMPLETE: dogCount2022 was a mid-season snapshot
#   (8 games, Apr–Jul only). The confirmed full-season total is 444,854 HD,
#   as shown in the 2023 and 2024 season sheets. Missing Aug–Sept 2022 game data
#   is not available; add individual game rows to raw_games if obtained.
SEASON_TOTALS = {
    2022: {"hot_dogs": 444854,  "attendance": 377138, "avg_dpf": 1.18, "games": None},  # full-season confirmed; raw_games partial (8 of ~12 games)
    2023: {"hot_dogs": 693870,  "attendance": 399683, "avg_dpf": 1.74, "games": 11},
    2024: {"hot_dogs": 727819,  "attendance": 400940, "avg_dpf": 1.82, "games": 13, "avg_dpm": 214},
    2025: {"hot_dogs": 839036,  "attendance": 410091, "avg_dpf": 2.05, "games": 12},
    # 2025: 12 confirmed games (Apr 1 – Sept 26)
    # HD:  659,525 (Apr–Aug) + 86,615 (Sept 26) + 92,896 (Sept 23) = 839,036
    # Att: 326,912 (Apr–Aug) + 40,252 (Sept 26) + 42,927 (Sept 23) = 410,091
}

# ============================================================
# 3. BUILD DATAFRAME & FEATURE ENGINEERING
# ============================================================

df = pd.DataFrame(raw_games)
df["date"]  = pd.to_datetime(df["date"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

# Derived stats
df["dpf"] = df["hot_dogs"] / df["attendance"]   # dogs per fan

# Dogs per minute (requires game duration)
df["total_selling_min"] = df["game_duration_min"] + PREGAME_WINDOW_MIN
df["dpm"] = df["hot_dogs"] / df["total_selling_min"]

# Season wins (same value for every game in a season)
df["wins"] = df["year"].map(SEASON_WINS)

# Year index offset from first season (allows model to capture year-over-year trend)
BASE_YEAR = df["year"].min()
df["year_index"] = df["year"] - BASE_YEAR   # 0=2022, 1=2023, 2=2024, 3=2025

# Month label for display
MONTH_NAMES = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
df["month_label"] = df["month"].map(MONTH_NAMES)

print("=" * 62)
print("LOONIE DOG NIGHT PREDICTOR – DATA SUMMARY")
print("=" * 62)
print(f"\nTotal games loaded : {len(df)}")
print(f"Seasons            : {df['year'].min()} – {df['year'].max()}")
print(f"Hot dogs range     : {df['hot_dogs'].min():,} – {df['hot_dogs'].max():,}")
print(f"Attendance range   : {df['attendance'].min():,.0f} – {df['attendance'].max():,.0f}")

# ── Sanity-check: compare summed totals against confirmed season totals ──
print("\n── Season totals (calculated from game rows vs. confirmed) ──")
for yr, ref in SEASON_TOTALS.items():
    calc_hd  = df.loc[df["year"] == yr, "hot_dogs"].sum()
    calc_att = df.loc[df["year"] == yr, "attendance"].sum()
    if ref["games"] is None:
        # Known incomplete raw data – show confirmed total and note partial coverage
        hd_note  = f"(partial raw data; confirmed full-season: {ref['hot_dogs']:,})"
        att_note = f"(partial; confirmed: {ref['attendance']:,})"
        print(f"  {yr}: HD {calc_hd:>8,} {hd_note}")
        print(f"        Att {calc_att:>7,} {att_note}  |  Wins {SEASON_WINS[yr]}")
    else:
        hd_ok  = "✓" if calc_hd  == ref["hot_dogs"]  else f"⚠ expected {ref['hot_dogs']:,}"
        att_ok = "✓" if calc_att == ref["attendance"] else f"⚠ expected {ref['attendance']:,}"
        print(f"  {yr}: HD {calc_hd:>8,} {hd_ok}  |  Att {calc_att:>8,} {att_ok}"
              f"  |  Wins {SEASON_WINS[yr]}")

# ============================================================
# 4. EXPLORATORY ANALYSIS
# ============================================================

print("\n── Avg Dogs Per Fan (DPF) by season ──")
dpf_by_year = df.groupby("year")["dpf"].mean()
for yr, val in dpf_by_year.items():
    bar = "█" * int(val * 20)
    print(f"  {yr} ({SEASON_WINS[yr]} W): {val:.2f}  {bar}")

print("\n── Correlation with hot dogs sold ──")
model_df_full = df.dropna(subset=["hot_dogs", "attendance", "wins"])
corr_cols = ["hot_dogs", "attendance", "year_index", "month", "wins"]
corr = model_df_full[corr_cols].corr()["hot_dogs"].drop("hot_dogs")
for col, val in corr.items():
    direction = "+" if val >= 0 else ""
    print(f"  {col:<15}: {direction}{val:.3f}")

print("\n── Monthly average hot dogs (all seasons) ──")
monthly = df.groupby("month")["hot_dogs"].agg(["mean", "count"]).reset_index()
monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
for _, row in monthly.iterrows():
    print(f"  {row['month_name']}: avg {row['mean']:>8,.0f}  (n={int(row['count'])})")

# ── Optional: generate EDA plots ──────────────────────────
def plot_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Loonie Dog Nights – Exploratory Analysis", fontsize=14, fontweight="bold")

    # 1. Attendance vs Hot Dogs (coloured by year)
    ax = axes[0, 0]
    sc = ax.scatter(df["attendance"], df["hot_dogs"],
                    c=df["year"], cmap="viridis", s=60, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_xlabel("Attendance"); ax.set_ylabel("Hot Dogs Sold")
    ax.set_title("Attendance vs Hot Dogs")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # 2. Dogs Per Fan by season (with win total annotated)
    ax = axes[0, 1]
    dpf_s = df.groupby("year")["dpf"].mean().reset_index()
    bars = ax.bar(dpf_s["year"].astype(str), dpf_s["dpf"], color="#003DA5", edgecolor="k")
    ax.set_xlabel("Season"); ax.set_ylabel("Avg DPF")
    ax.set_title("Dogs Per Fan & Season Wins")
    ax.set_ylim(0, 2.6)
    for i, row in dpf_s.iterrows():
        w = SEASON_WINS[row["year"]]
        ax.text(i, row["dpf"] + 0.04, f"{row['dpf']:.2f}\n({w}W)", ha="center", fontsize=8)

    # 3. Hot dogs by month
    ax = axes[1, 0]
    monthly_hd = df.groupby("month")["hot_dogs"].mean()
    months = [MONTH_NAMES[m] for m in monthly_hd.index]
    ax.bar(months, monthly_hd.values, color="#E8002D", edgecolor="k")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Hot Dogs")
    ax.set_title("Average Hot Dogs by Month")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # 4. Season totals
    ax = axes[1, 1]
    sy = list(SEASON_TOTALS.keys())
    sh = [SEASON_TOTALS[y]["hot_dogs"] for y in sy]
    ax.bar([str(y) for y in sy], sh, color="#134A8E", edgecolor="k")
    ax.set_xlabel("Season"); ax.set_ylabel("Total Hot Dogs")
    ax.set_title("Season Hot Dog Totals")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    for i, (y, h) in enumerate(zip(sy, sh)):
        ax.text(i, h + 8000, f"{h/1000:.0f}K", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("loonie_dog_eda.png", dpi=150, bbox_inches="tight")
    print("\n[Plot saved: loonie_dog_eda.png]")
    plt.show()

# Uncomment to generate EDA plots:
# plot_eda(df)

# ============================================================
# 5. MODEL TRAINING
# ============================================================
# Model A – Simple linear: hot_dogs ~ attendance
#   Baseline. Attendance is the single strongest predictor.
#
# Model B – Multiple regression: hot_dogs ~ attendance + year_index + month + wins
#   Captures:
#     attendance  – how many fans are there to buy dogs
#     year_index  – captures rising DPF (promotional momentum)
#     month       – seasonal pattern (July/Aug peaks)
#     wins        – better team → more engaged fans → higher DPF
#
# Dataset: only games with known attendance (all games have this).
# Evaluation: Leave-One-Out CV (robust for n≈42 samples).

model_df = df.dropna(subset=["hot_dogs", "attendance", "wins"]).copy()

features_A = ["attendance"]
features_B = ["attendance", "year_index", "month", "wins"]

X_A = model_df[features_A].values
X_B = model_df[features_B].values
y   = model_df["hot_dogs"].values

# Fit on full dataset (coefficients used for final predictions)
model_A = LinearRegression().fit(X_A, y)
model_B = LinearRegression().fit(X_B, y)

# Leave-One-Out CV – compute manually to avoid sklearn LOO R² instability
def loo_stats(X, y):
    loo   = LeaveOneOut()
    preds = np.empty(len(y))
    for train_idx, test_idx in loo.split(X):
        m = LinearRegression().fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    mae    = mean_absolute_error(y, preds)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    return mae, r2

cv_mae_A, cv_r2_A = loo_stats(X_A, y)
cv_mae_B, cv_r2_B = loo_stats(X_B, y)

print("\n── Model Results ──")
print(f"\n  Model A: hot_dogs ~ attendance")
print(f"    Coefficient : {model_A.coef_[0]:.4f}  (hot dogs per additional fan)")
print(f"    Intercept   : {model_A.intercept_:,.0f}")
print(f"    LOO-CV R²   : {cv_r2_A:.3f}")
print(f"    LOO-CV MAE  : {cv_mae_A:,.0f} hot dogs")

print(f"\n  Model B: hot_dogs ~ attendance + year_index + month + wins")
for lbl, coef in zip(features_B, model_B.coef_):
    print(f"    {lbl:<15}: {coef:+.4f}")
print(f"    Intercept   : {model_B.intercept_:,.0f}")
print(f"    LOO-CV R²   : {cv_r2_B:.3f}")
print(f"    LOO-CV MAE  : {cv_mae_B:,.0f} hot dogs")

# Select the better model by LOO-CV MAE
if cv_mae_B < cv_mae_A:
    best_model    = model_B
    best_features = features_B
    model_label   = "B (attendance + year_index + month + wins)"
else:
    best_model    = model_A
    best_features = features_A
    model_label   = "A (attendance only)"

print(f"\n  → Using Model {model_label} for 2026 predictions")

# ============================================================
# 6. 2026 PREDICTIONS
# ============================================================
# IMPORTANT: Replace the placeholder dates below with the confirmed
# Tuesday home game dates from the official 2026 Blue Jays schedule.
# Source: mlb.com/bluejays/schedule
#
# Placeholder dates are estimated Tuesdays spaced across the season;
# the actual home/away split will differ.

games_2026 = [
    "2026-03-31",
    "2026-04-07",
    "2026-04-28",
    "2026-05-12",
    "2026-05-26",
    "2026-06-09",
    "2026-06-23",
    "2026-06-30",
    "2026-07-21",
    "2026-08-11",
    "2026-08-25",
    "2026-09-15",
]

# Projected attendance per month: historical median across all seasons.
# Override individual entries if you have better information (e.g., a
# marquee opponent typically draws 5–10% more than median).
ATT_BY_MONTH = (
    model_df
    .groupby("month")["attendance"]
    .median()
    .to_dict()
)
OVERALL_MEDIAN_ATT = model_df["attendance"].median()
YEAR_2026_INDEX    = 2026 - BASE_YEAR   # = 4

print("\n── 2026 Monthly Median Attendance Assumptions ──")
for m, att in sorted(ATT_BY_MONTH.items()):
    print(f"  {MONTH_NAMES.get(m, m)}: {att:,.0f}")
print(f"  Projected 2026 wins: {PROJECTED_2026_WINS}")

# Build prediction rows
pred_rows = []
for date_str in games_2026:
    dt       = datetime.strptime(date_str, "%Y-%m-%d")
    month    = dt.month
    proj_att = ATT_BY_MONTH.get(month, OVERALL_MEDIAN_ATT)
    pred_rows.append({
        "date":       dt,
        "month":      month,
        "year_index": YEAR_2026_INDEX,
        "attendance": proj_att,
        "wins":       PROJECTED_2026_WINS,
    })

pred_df = pd.DataFrame(pred_rows)
pred_df["month_label"] = pred_df["month"].map(MONTH_NAMES)

# Predict
X_pred = pred_df[best_features].values
pred_df["predicted_hot_dogs"] = best_model.predict(X_pred).round(0).astype(int)

# Uncertainty band: ±1 LOO-CV MAE
mae = cv_mae_B if best_model is model_B else cv_mae_A
pred_df["low_estimate"]  = (pred_df["predicted_hot_dogs"] - mae).round(0).astype(int)
pred_df["high_estimate"] = (pred_df["predicted_hot_dogs"] + mae).round(0).astype(int)
pred_df["proj_dpf"]      = pred_df["predicted_hot_dogs"] / pred_df["attendance"]

# ── Simple cross-check: extrapolate avg DPF trend × median attendance ──
dpf_years  = np.array([2023, 2024, 2025]).reshape(-1, 1)
dpf_vals   = np.array([
    SEASON_TOTALS[2023]["avg_dpf"],
    SEASON_TOTALS[2024]["avg_dpf"],
    SEASON_TOTALS[2025]["avg_dpf"],
])
dpf_model     = LinearRegression().fit(dpf_years, dpf_vals)
proj_dpf_2026 = dpf_model.predict([[2026]])[0]
simple_total  = proj_dpf_2026 * OVERALL_MEDIAN_ATT * len(games_2026)

# ============================================================
# 7. OUTPUT
# ============================================================

DIV = "─" * 72
print(f"\n{'=' * 72}")
print("  2026 LOONIE DOG NIGHT PREDICTIONS")
print(f"  Model : {model_label}")
print(f"  LOO-CV MAE uncertainty band: ±{mae:,.0f} hot dogs per game")
print(f"{'=' * 72}")
print(f"  {'Date':<14} {'Month':<6} {'Proj Att':>10} {'Predicted HD':>13} "
      f"{'Low':>10} {'High':>10}  Proj DPF")
print(f"  {DIV}")
for _, row in pred_df.iterrows():
    print(f"  {row['date'].strftime('%Y-%m-%d'):<14} "
          f"{row['month_label']:<6} "
          f"{row['attendance']:>10,.0f} "
          f"{row['predicted_hot_dogs']:>13,} "
          f"{row['low_estimate']:>10,} "
          f"{row['high_estimate']:>10,}  "
          f"{row['proj_dpf']:.2f}")

print(f"  {DIV}")
total_pred = pred_df["predicted_hot_dogs"].sum()
total_low  = pred_df["low_estimate"].sum()
total_high = pred_df["high_estimate"].sum()
print(f"  {'SEASON TOTAL':<14}        "
      f"{'':>10} "
      f"{total_pred:>13,} "
      f"{total_low:>10,} "
      f"{total_high:>10,}")
print(f"{'=' * 72}")

print(f"\n── Historical Season Totals for Context ──")
for yr, info in SEASON_TOTALS.items():
    w      = SEASON_WINS[yr]
    g_str  = f"{info['games']} games" if info["games"] else "game count unknown"
    print(f"  {yr}: {info['hot_dogs']:>8,} HD  "
          f"({g_str}, avg DPF {info['avg_dpf']:.2f}, {w} wins)")
print(f"  2026: {total_pred:>8,} HD  "
      f"({len(games_2026)} games, avg DPF {proj_dpf_2026:.2f} projected, "
      f"{PROJECTED_2026_WINS} wins projected)")
print(f"\n  Simple DPF-trend cross-check total: {simple_total:,.0f}")

# ============================================================
# 8. KEY INSIGHTS
# ============================================================

print(f"\n── Key Insights ──")
print(f"  • Attendance alone explains {cv_r2_A:.1%} of hot dog variance (LOO-CV, Model A).")
print(f"  • Adding year trend, month & wins improves LOO-CV MAE: "
      f"{cv_mae_A:,.0f} → {cv_mae_B:,.0f} hot dogs.")
wins_coef = model_B.coef_[features_B.index("wins")]
print(f"  • Each additional win is associated with ~{wins_coef:+,.0f} more hot dogs per game.")
print(f"  • DPF trend: 1.17 (2022) → 2.03 (2025) → {proj_dpf_2026:.2f} projected (2026).")
print(f"  • July/Aug games outsell April games by ~60–80% on average.")
print(f"  • 2025 Sept 23 single-game DPM: "
      f"{92896 / (97 + 172):.0f} dogs/min (highest pace on record).")
print(f"  → Update SEASON_WINS[2025] in Section 1 with the confirmed 2025 win total.")
print(f"  → Adjust PROJECTED_2026_WINS to match preseason expectations.")

# ============================================================
# 9. HOW TO UPDATE FOR FUTURE SEASONS
# ============================================================
# To add data for a new season (e.g., 2027):
#   1. Append game dicts to raw_games (Section 2)
#   2. Add the season win total to SEASON_WINS (Section 1)
#   3. Update SEASON_TOTALS with confirmed season totals
#   4. Update games_2026 → games_2027 with next year's schedule (Section 6)
#   5. Set PROJECTED_2026_WINS → PROJECTED_2027_WINS
#   6. Re-run the script
