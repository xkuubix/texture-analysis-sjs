# %%
import pandas as pd

SAVE_FIGURES = True
SAVE_FORMAT = "eps"

results_path = "/users/project1/pt01190/sjs/Results"

clinical_data_path = "/users/project1/pt01190/sjs/Data/SJS_DaneKliniczne_dla_PG_v1.xlsm"
morphological_data_path = "/users/project1/pt01190/sjs/Data/Baza dla PG v1.xlsx"

clinical_data = pd.read_excel(clinical_data_path, header=[2], engine="openpyxl")
for col in clinical_data.columns:
    if "Unnamed:" in str(col):
        clinical_data = clinical_data.drop(columns=col)
for time_col in ["Date of study", "Data Schirmer i UWSF", "Data Focus Score"]:
    clinical_data[time_col] = pd.to_datetime(
        clinical_data[time_col].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

# drop 320; 160  + border
clinical_data = clinical_data[clinical_data["ANA SSB"].str.strip().str.lower() != "border"]
clinical_data["ANA 2"] = clinical_data["ANA 2"].astype(str).str.extract(r"(\d+)").astype(float)
# change ANA SSA non-missing values to integers, its float now
clinical_data["ANA SSA"] = clinical_data["ANA SSA"].convert_dtypes().astype("Int64")

# make RF col normal <30 and abnormal otherwise
clinical_data["RF_bin"] = pd.to_numeric(clinical_data["RF"], errors="coerce")
clinical_data["RF_bin"] = clinical_data["RF_bin"].apply(lambda x: "normal" if pd.notnull(x) and x < 30 else ("abnormal" if pd.notnull(x) else pd.NA))

# drop missing diagnosis
clinical_data = clinical_data[clinical_data["diagnosis"].notna()]

# change zdrowy to healthy
clinical_data["diagnosis"] = clinical_data["diagnosis"].str.strip().str.lower().replace({"zdrowy": "healthy"})

# %%
keep_clinical_cols = [
    'Date of study',
    'Patient Number',
    # 'Focus Score',
    'FS (jeśli 2 to wynik niejednoznaczny)',
    'Data Focus Score',
    # 'Schirmer OP',
    # 'Schirmer OL',
    # 'UWSF',
    'Schirmer OP positive =1 negative=0',
    'Schirmer OL positive =1 negative=0',
    'Data Schirmer i UWSF',
    'UWSF positive =1 negative=0',
    'diagnosis',
    # 'pSS/sSS/UCTD',
    # 'onset of disease',
    # 'time to diagnosis',
    'disease duration',
    'history salivary',
    'syptoms - dryness',
    'RF',
    'RF_bin',
    'ANA',
    'ANA 2',
    # 'typ świecenia', # llm embedd??
    'ANA SSB',
    'ANA SSA'
    ]
clinical_data = clinical_data[keep_clinical_cols]
# %%
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "grid.color": "#636060",
    
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    
    "font.family": "STIXGeneral",
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.labelcolor": "#333333",
    
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    
    "lines.linewidth": 2.0,
    "patch.force_edgecolor": True,
    "patch.edgecolor": "none",
    
    "figure.autolayout": True,
    "figure.dpi": 300,
})
ax = clinical_data["diagnosis"].value_counts().sort_index().plot(kind="bar")
plt.ylabel("Number of patients")
plt.xlabel("Diagnosis")
plt.title("Diagnosis distribution")
plt.xticks(rotation=0)
plt.tight_layout()
ax.grid(False, axis="x")
ax.grid(True, axis="y", alpha=0.5, linestyle="--")
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/diagnosis_distribution.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()

# %%
tests = ["Schirmer OP positive =1 negative=0",
         "Schirmer OL positive =1 negative=0",
         "UWSF positive =1 negative=0"]

positive = clinical_data[tests].replace("-", pd.NA).apply(pd.to_numeric, errors="coerce")
positive["diagnosis"] = clinical_data["diagnosis"]

rates = positive.groupby("diagnosis")[tests].mean() * 100
counts = positive.groupby("diagnosis")[tests].count()
ax = rates.plot(kind="bar")
for container, test in zip(ax.containers, tests):
    for rect, n in zip(container, counts[test]):
        height = rect.get_height()
        ax.annotate(
            f"N = {n}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3pt vertical padding
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
plt.ylabel("Positive tests (%)")
plt.xlabel("Diagnosis")
plt.title("Positive Schirmer and UWSF tests")
plt.xticks(rotation=0)
plt.legend(["Schirmer OP", "Schirmer OL", "UWSF"])
plt.tight_layout()
ax.grid(False, axis="x")
ax.grid(True, axis="y", alpha=0.5, linestyle="--")
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/positive_tests.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()
# %%
rf = pd.crosstab(clinical_data["diagnosis"], clinical_data["RF_bin"], normalize="index") * 100
ax = rf.plot(kind="bar", stacked=True)
plt.ylabel("Patients (%)")
plt.xlabel("Diagnosis")
plt.title("RF status by diagnosis")
plt.xticks(rotation=0)
plt.legend(title="RF")
plt.tight_layout()
ax.grid(False, axis="x")
ax.grid(True, axis="y", alpha=0.5, linestyle="--")
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/rf_status.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()
# %%
import seaborn as sns

for col in ["RF", "ANA"]:
    x = pd.to_numeric(clinical_data[col].astype(str).str.replace(">", ""), errors="coerce")
    data = pd.DataFrame({"diagnosis": clinical_data["diagnosis"], col: x}).dropna()

    ax = sns.stripplot(x="diagnosis", y=col, data=data, color="black", alpha=0.5, jitter=True)
    sns.boxplot(x="diagnosis", y=col, data=data, showcaps=True, boxprops={"facecolor": "None"}, fliersize=0, ax=ax)
    plt.ylabel(f"{col}")
    plt.xlabel("Diagnosis")
    plt.title(f"{col} by diagnosis")
    plt.suptitle("")
    plt.tight_layout()
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", alpha=0.5, linestyle="--")
    if SAVE_FIGURES:
        plt.savefig(f"{results_path}/{col}_by_diagnosis.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
    plt.show()

# %%
for col in ["ANA SSA", "ANA SSB"]:
    ax = pd.crosstab(clinical_data["diagnosis"], clinical_data[col], normalize="index").mul(100).plot(
        kind="bar", stacked=True
    )
    plt.ylabel("Patients (%)")
    plt.xlabel("Diagnosis")
    plt.title(f"{col} by diagnosis")
    plt.xticks(rotation=0)
    plt.tight_layout()
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", alpha=0.5, linestyle="--")
    if SAVE_FIGURES:
        plt.savefig(f"{results_path}/{col}_by_diagnosis.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
    plt.show()
# %%

import seaborn as sns

d = clinical_data.dropna(subset=["disease duration"])

ax = sns.stripplot(x="diagnosis", y="disease duration", data=d, color="black", alpha=0.5, jitter=True)
sns.boxplot(x="diagnosis", y="disease duration", data=d, showcaps=True, boxprops={"facecolor": "None"}, fliersize=0, ax=ax)
plt.ylabel("Disease duration (years)")
plt.xlabel("Diagnosis")
plt.title("Disease duration")
plt.suptitle("")
plt.tight_layout()
ax.grid(False, axis="x")
ax.grid(True, axis="y", alpha=0.5, linestyle="--")
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/disease_duration.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()
# %%

cols = [
    "Schirmer OP positive =1 negative=0",
    "Schirmer OL positive =1 negative=0",
    "UWSF positive =1 negative=0",
    "RF", "ANA", "ANA SSA", "ANA SSB",
    "history salivary", "syptoms - dryness"
]

missing = clinical_data[clinical_data["diagnosis"]!='healthy'][cols].isna().mean().mul(100).sort_values()

ax = missing.plot(kind="barh")
plt.xlabel("Missing data (%)")
plt.title("Missingness of clinical variables\n(excluding healthy controls)")
plt.tight_layout()
ax.grid(False, axis="y")
ax.grid(True, axis="x", alpha=0.5, linestyle="--")
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/missing_data.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()

# %%
morphological_data = pd.read_excel(morphological_data_path, header=[0, 1, 2], engine="openpyxl")
morphological_data = morphological_data.iloc[:, :24]
new_columns = []
for level_0, level_1, level_2 in morphological_data.columns:
    if "Unnamed:" in str(level_0) and "Unnamed:" in str(level_1):
        new_columns.append(str(level_2).strip())
    elif "Unnamed:" in str(level_0) and "Unnamed:" not in str(level_1):
        new_columns.append(f"{level_1.strip()} - {level_2.strip()}")
    else:
        new_columns.append(f"{level_0.strip()} - {level_1.strip()} - {level_2.strip()}")

morphological_data.columns = new_columns
morphological_data = morphological_data.loc[:, ~morphological_data.columns.duplicated(keep='first')]
morphological_data = morphological_data.dropna(how="all", axis=1)

for time_col in ["Date of study"]:
    morphological_data[time_col] = pd.to_datetime(
        morphological_data[time_col].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

morphological_data["Sex"] = morphological_data["Sex"].replace("-", pd.NA)

for col in ["Age", "Weight", "Height"]:
    morphological_data[col] = pd.to_numeric(
        morphological_data[col], errors="coerce"
    )

# ambiguous entries [like 0(1)] are coerced to NaN, and the rest are converted to numeric values
for col in morphological_data.columns[6:]:
    morphological_data[col] = pd.to_numeric(
        morphological_data[col], errors="coerce"
    )
morphological_data.dropna(
    subset=morphological_data.columns[8:], how="any", axis=0, inplace=True
)

keep_morphological_cols = [
    'Date of study',
    'Patient Number',
    'Sex',
    'Age',
    'Weight',
    'Height',
    # 'MR_głowa yes-1  no-0',
    # 'Quality/Problems',
    'Right Parotid Gland - Morpholohy - T1W heterogeneous fat deposition \n(0-homo, \n1-hetero)',
    'Right Parotid Gland - Morpholohy - Multiple hyperintense spots in PGs on fsT2WI \n(0-no spots \n1-presence of spots)',
    'Right Parotid Gland - Morpholohy - Multiple hyperintense spots in PGs on MR sialography \n(0-no spots \n1-presence of spots)',
    'Right Parotid Gland - TONAMI - Grade 0 (normal)—no evidence \nGrade 1 (punctate)—areas ≤ 1 mm in diameter;  \nGrade 2 (globular)—1–2 mm in diameter; \nGrade 3 (cavitary)—up to 1 cm in diameter; \nGrade 4 (destructive)— complete destruction of the gland parenchyma.',
    'Right Parotid Gland - fsT2WI - high-intensity spots - grade I (0–9\ngrade II (10–19\ngrade III (20–29\ngrade IV (>29',
    'Right Parotid Gland - Sjalo - high-intensity spots - grade I (0–19\ngrade II (20–39\ngrade III (40–59\ngrade IV (>59)',
    'Right Parotid Gland - Cho - Grade 1: normal or sparse distribution of streak-like fat signals \nGrade 2: diffusive distributed, honeycomb-like fat signals; \nGrade 3: less than 50% of the total area of whole salivary glands; \nGrade 4: massively homogeneously distributed fat\nsignal',
    'Left Parotid Gland - Morpholohy + ADC - T1W heterogeneous fat deposition \n(0-homo, \n1-hetero)',
    'Left Parotid Gland - Morpholohy + ADC - Multiple hyperintense spots in PGs on fsT2WI \n(0-no spots \n1-presence of spots)',
    'Left Parotid Gland - Morpholohy + ADC - Multiple hyperintense spots in PGs on MR sialography \n(0-no spots \n1-presence of spots)',
    'Left Parotid Gland - TONAMI - Grade 0 (normal)—no evidence \nGrade 1 (punctate)—areas ≤ 1 mm in diameter;  \nGrade 2 (globular)—1–2 mm in diameter; \nGrade 3 (cavitary)—up to 1 cm in diameter; \nGrade 4 (destructive)— complete destruction of the gland parenchyma.',
    'Left Parotid Gland - fsT2WI - high-intensity spots - grade I (0–9\ngrade II (10–19\ngrade III (20–29\ngrade IV (>29',
    'Left Parotid Gland - Sjalo - high-intensity spots - grade I (0–19\ngrade II (20–39\ngrade III (40–59\ngrade IV (>59)',
    'Left Parotid Gland - Cho - Grade 1: normal or sparse distribution of streak-like fat signals \nGrade 2: diffusive distributed, honeycomb-like fat signals; \nGrade 3: less than 50% of the total area of whole salivary glands; \nGrade 4: massively homogeneously distributed fat\nsignal'
       ]

# for i in range(len(keep_morphological_cols)):
#     col = morphological_data[keep_morphological_cols[i]]
#     print(col.value_counts(dropna=False))
# %%
data = clinical_data.merge(
    morphological_data,
    on="Patient Number",
    how="inner",
    suffixes=("_clinical", "_morph")
)

# %%
contingency_matrix = pd.crosstab(data["Sex"], data["diagnosis"])

plt.figure(figsize=(6, 4))
sns.heatmap(contingency_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Diagnosis")
plt.ylabel("Sex")
plt.title("Diagnosis by sex")
plt.tight_layout()
plt.yticks(rotation=0)
plt.ylabel("Sex", rotation=0, labelpad=20)
plt.grid(False)
if SAVE_FIGURES:
    plt.savefig(f"{results_path}/diagnosis_by_sex.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
plt.show()
# %%
for col in ["Age", "Weight", "Height"]:
    groups = [data.loc[data["diagnosis"] == d, col].dropna()
              for d in ["healthy", "uctd", "ss"]]

    plt.figure(figsize=(6, 4))
    plt.boxplot(groups)
    for i, y in enumerate(groups, 1):
        plt.scatter([i] * len(y), y, alpha=0.6)

    plt.xticks([1, 2, 3], ["healthy", "uctd", "ss"])
    plt.xlabel("Diagnosis")
    plt.ylabel(col)
    plt.title(f"{col} by diagnosis")
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{results_path}/{col}_by_diagnosis.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
    plt.show()

for col in ["Age", "Weight", "Height"]:
    plt.figure(figsize=(6, 4))

    for diagnosis in ["healthy", "uctd", "ss"]:
        x = data.loc[data["diagnosis"] == diagnosis, col].dropna()
        sns.kdeplot(x, label=diagnosis)

    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"{col} distribution by diagnosis")
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{results_path}/{col}_distribution_by_diagnosis.{SAVE_FORMAT}", format=SAVE_FORMAT, dpi=300, bbox_inches="tight")
    plt.show()

# %%
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd
import matplotlib.pyplot as plt

mri_cols_map = {
    # Right
    next(c for c in data.columns if "Right" in c and "T1W" in c): "Right_T1W_fat",
    next(c for c in data.columns if "Right" in c and "fsT2WI \n(" in c): "Right_fsT2WI_spots",
    next(c for c in data.columns if "Right" in c and "MR sialography \n(" in c): "Right_Sjalo_spots",
    next(c for c in data.columns if "Right" in c and "TONAMI" in c): "Right_TONAMI",
    next(c for c in data.columns if "Right" in c and "fsT2WI - high-intensity" in c): "Right_fsT2WI_grade",
    next(c for c in data.columns if "Right" in c and "Sjalo - high-intensity" in c): "Right_Sjalo_grade",
    next(c for c in data.columns if "Right" in c and "Cho" in c): "Right_Cho_grade",

    # Left
    next(c for c in data.columns if "Left" in c and "T1W" in c): "Left_T1W_fat",
    next(c for c in data.columns if "Left" in c and "fsT2WI \n(" in c): "Left_fsT2WI_spots",
    next(c for c in data.columns if "Left" in c and "MR sialography \n(" in c): "Left_Sjalo_spots",
    next(c for c in data.columns if "Left" in c and "TONAMI" in c): "Left_TONAMI",
    next(c for c in data.columns if "Left" in c and "fsT2WI - high-intensity" in c): "Left_fsT2WI_grade",
    next(c for c in data.columns if "Left" in c and "Sjalo - high-intensity" in c): "Left_Sjalo_grade",
    next(c for c in data.columns if "Left" in c and "Cho" in c): "Left_Cho_grade",
}

data = data.rename(columns=mri_cols_map)

binary_cols = [
    "Right_T1W_fat", "Right_fsT2WI_spots", "Right_Sjalo_spots",
    "Left_T1W_fat", "Left_fsT2WI_spots", "Left_Sjalo_spots"
]

ordinal_cols = [
    "Right_TONAMI", "Right_fsT2WI_grade", "Right_Sjalo_grade", "Right_Cho_grade",
    "Left_TONAMI", "Left_fsT2WI_grade", "Left_Sjalo_grade", "Left_Cho_grade"
]

# %%
from stat_utils import get_kappa_ci

print("\nLeft/Right Agreement Summary:")
print(
    f"{'Feature':<16} {'Weight':<10} {'n':<6} {'P_o (%)':<8} {'Kappa':<8} {'95% CI'}"
)
print("-" * 65)

lr_pairs = [
    ("Right_T1W_fat", "Left_T1W_fat", None),
    ("Right_fsT2WI_spots", "Left_fsT2WI_spots", None),
    ("Right_Sjalo_spots", "Left_Sjalo_spots", None),
    ("Right_TONAMI", "Left_TONAMI", "quadratic"),
    ("Right_fsT2WI_grade", "Left_fsT2WI_grade", "quadratic"),
    ("Right_Sjalo_grade", "Left_Sjalo_grade", "quadratic"),
    ("Right_Cho_grade", "Left_Cho_grade", "quadratic"),
]

results = []
for r_col, l_col, weight_type in lr_pairs:
    paired = data[[r_col, l_col]].dropna()
    if len(paired) >= 5:
        k, ci_l, ci_u, po = get_kappa_ci(
            paired[r_col], paired[l_col], weights=weight_type
        )
        feature_name = r_col.replace("Right_", "")
        w_str = weight_type if weight_type else "none"

        print(
            f"{feature_name:<16} {w_str:<10} {len(paired):<6} {po:<8.1f} {k:<8.2f} [{ci_l:.2f}, {ci_u:.2f}]"
        )

        results.append(
            {
                "Feature": feature_name,
                "Weights": w_str,
                "n": len(paired),
                "Observed_Agreement (%)": round(po, 1),
                "Kappa": round(k, 2),
                "95% CI": f"[{ci_l:.2f}, {ci_u:.2f}]",
            }
        )
# %%
# binary findings: patient is positive if EITHER gland is positive
for base in ["T1W_fat", "fsT2WI_spots", "Sjalo_spots"]:
    data[f"{base}_either"] = data[[f"Right_{base}", f"Left_{base}"]].max(axis=1, skipna=True)

# ordinal grades: take the WORSE (higher) of the two sides
for base in ["TONAMI", "fsT2WI_grade", "Sjalo_grade", "Cho_grade"]:
    data[f"{base}_worse"] = data[[f"Right_{base}", f"Left_{base}"]].max(axis=1, skipna=True)


# ============================================================
# MAIN RESULTS: MRI features (L/R-collapsed composites) vs diagnosis
# =================================================================
# Uses the composites validated in the agreement/sensitivity checks:
#   - binary findings -> "either side positive"  (*_either)
#   - ordinal grades   -> "worse side"            (*_worse)
from scipy.stats import fisher_exact, chi2_contingency, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import pandas as pd

diagnoses = sorted(data["diagnosis"].dropna().unique())

binary_composite = [f"{b}_either" for b in ["T1W_fat", "fsT2WI_spots", "Sjalo_spots"]]
ordinal_composite = [f"{b}_worse" for b in ["TONAMI", "fsT2WI_grade", "Sjalo_grade", "Cho_grade"]]

def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    r, c = table.shape
    return (chi2 / (n * (min(r, c) - 1))) ** 0.5

def epsilon_squared(groups):
    all_vals = pd.concat(groups)
    n = len(all_vals)
    h_stat, _ = kruskal(*groups)
    return h_stat / (n - 1)

# --- binary features: Fisher (2x2) / chi-square (larger tables) ---
binary_rows = []
for col in binary_composite:
    table = pd.crosstab(data["diagnosis"], data[col])
    if table.shape == (2, 2):
        _, p = fisher_exact(table.values)
        test_name = "Fisher exact"
    else:
        _, p, _, expected = chi2_contingency(table)
        test_name = "Chi-square"
        if (expected < 5).any():
            print(f"NOTE: {col} table has expected counts <5; "
                  f"chi-square p-value may be unreliable")
    binary_rows.append({
        "Feature": col, "Test": test_name, "p_value": p,
        "EffectSize": cramers_v(table), "EffectSizeType": "Cramer's V",
        "n": int(table.values.sum()),
    })

# --- ordinal features: Kruskal-Wallis omnibus tests ---
ordinal_rows = []
for col in ordinal_composite:
    groups = {d: data.loc[data["diagnosis"] == d, col].dropna() for d in diagnoses}
    groups = {d: g for d, g in groups.items() if len(g) > 0}
    if len(groups) < 2:
        continue
    stat, p = kruskal(*groups.values())
    ordinal_rows.append({
        "Feature": col, "Test": "Kruskal-Wallis", "p_value": p,
        "EffectSize": epsilon_squared(list(groups.values())), "EffectSizeType": "epsilon^2",
        "n": sum(len(g) for g in groups.values()),
    })

# --- combine, FDR-correct across ALL MRI features together ---
df_main = pd.DataFrame(binary_rows + ordinal_rows)
df_main["p_adj"] = multipletests(df_main["p_value"], method="fdr_bh")[1]
df_main = df_main.sort_values("p_adj").reset_index(drop=True)

# ============================================================
# PRINT + EXPORT
# =================================================================
pd.set_option("display.width", 120)
print("=" * 70)
print("MAIN RESULTS: MRI feature vs diagnosis (FDR-corrected, n =", len(diagnoses), "groups)")
print("=" * 70)
print(df_main.to_string(index=False))

df_main.to_csv(f"{results_path}/mri_main_results.csv", index=False)

# ============================================================
# INTERPRETATION GUIDE (printed, not just a comment, so it's visible
# alongside the numbers when this cell runs)
# =================================================================
print("\n" + "=" * 70)
print("HOW TO READ THIS TABLE")
print("=" * 70)
print("""
p_adj      Benjamini-Hochberg FDR-corrected p-value across all 7 MRI
           features tested. Use p_adj < .05 as the significance
           threshold, NOT p_value - the raw p_value is shown only for
           transparency and will overstate significance if used alone.

EffectSize interpretation (conventional benchmarks, treat as rough guides
           only - they were not derived for small clinical samples):

  Cramer's V (binary features):   ~0.1 small | ~0.3 medium | ~0.5 large
  epsilon^2  (ordinal features):  ~0.01 small | ~0.06 medium | ~0.14 large

  A feature can be statistically significant (small p_adj) with a small
  effect size if n is reasonably large, or non-significant with a
  moderate effect size if n is small - report BOTH, not p_adj alone.

Post-hoc comparisons are only meaningful for features where the omnibus
  Kruskal-Wallis already cleared p_adj < .05; comparisons for
  non-significant features are exploratory at best and shown only if
  you want to eyeball trend direction.

All 7 composites have n = 153. After the L/R collapse
    (max(..., skipna=True)), a patient is dropped only if BOTH sides
    are missing for that feature, so no per-feature missingness remains.
""")

# %%
# %% ============================================================
# Pairwise Fisher's exact for binary composites - resolves the
# "expected counts < 5" warning by never relying on chi-square's
# asymptotic assumption in the first place, and tells you WHICH
# diagnosis pair drives each binary feature's omnibus result.
# =================================================================
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import pandas as pd

binary_composite = [f"{b}_either" for b in ["T1W_fat", "fsT2WI_spots", "Sjalo_spots"]]
diagnoses = sorted(data["diagnosis"].dropna().unique())

# raw counts first - confirms which group is driving the low-expected-count
# warning (almost certainly whichever group has near-zero positives)
print("Raw counts, binary composites by diagnosis:")
for col in binary_composite:
    print(f"\n{col}:")
    print(pd.crosstab(data["diagnosis"], data[col]))

binary_posthoc_rows = []
for col in binary_composite:
    for d1, d2 in combinations(diagnoses, 2):
        sub = data[data["diagnosis"].isin([d1, d2])]
        table = pd.crosstab(sub["diagnosis"], sub[col])
        if table.shape != (2, 2):
            continue  # one group has zero variance in this feature, skip
        _, p = fisher_exact(table.values)
        binary_posthoc_rows.append({
            "Feature": col, "Comparison": f"{d1} vs {d2}",
            "p_value": p, "n": int(table.values.sum()),
        })

df_binary_posthoc = pd.DataFrame(binary_posthoc_rows)
df_binary_posthoc["p_adj"] = multipletests(df_binary_posthoc["p_value"], method="fdr_bh")[1]
df_binary_posthoc = df_binary_posthoc.sort_values(["Feature", "p_adj"])

print("\n" + "=" * 70)
print("Binary feature post-hoc (pairwise Fisher's exact, FDR-corrected)")
print("=" * 70)
print(df_binary_posthoc.to_string(index=False))

# %% ============================================================
# Odds ratios for binary composites, Haldane-Anscombe corrected
# (adds 0.5 to every cell) so zero-count cells don't produce an
# undefined or infinite OR. CI via Woolf's log-odds method.
# =================================================================
import numpy as np
from scipy.stats import norm

def or_with_ci(table, alpha=0.05):
    """table: 2x2 array [[a,b],[c,d]] = [[exposed_pos, exposed_neg],
       [unexposed_pos, unexposed_neg]] - here rows are diagnosis groups,
       columns are feature 0/1."""
    a, b = table[0]
    c, d = table[1]
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5  # Haldane-Anscombe
    odds_ratio = (a * d) / (b * c)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf(1 - alpha / 2)
    log_or = np.log(odds_ratio)
    ci_low = np.exp(log_or - z * se_log_or)
    ci_high = np.exp(log_or + z * se_log_or)
    return odds_ratio, ci_low, ci_high

or_rows = []
for col in binary_composite:
    for d1, d2 in combinations(diagnoses, 2):
        sub = data[data["diagnosis"].isin([d1, d2])]
        table = pd.crosstab(sub["diagnosis"], sub[col])
        if table.shape[1] != 2:
            continue  # feature has no variance at all in this pair, OR undefined regardless
        table = table.reindex(columns=[0.0, 1.0], fill_value=0)
        odds_ratio, lo, hi = or_with_ci(table.values)
        or_rows.append({
            # (a*d)/(b*c) with rows=[d1,d2], cols=[neg,pos] computes
            # odds(d2)/odds(d1) - so this is always "OR of d2 vs d1",
            # e.g. d1="healthy", d2="ss" -> OR of ss relative to healthy.
            "Feature": col,
            "OR_of": d2, "relative_to": d1,
            "OR": round(odds_ratio, 2), "CI_low": round(lo, 2), "CI_high": round(hi, 2),
            "corrected": "yes (0.5 added, zero cells present)" if (table.values == 0).any() else "no",
        })

df_or = pd.DataFrame(or_rows)
print(df_or.to_string(index=False))

# %%
# %% ============================================================
# ORDINAL POST-HOC, with effect size - drop-in replacement for the
# posthoc_rows loop inside the ordinal features block.
# =================================================================

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import numpy as np
import pandas as pd

def rank_biserial(g1, g2, u_stat):
    """Effect size for Mann-Whitney U, on a -1..1 scale.
    Positive = g1 tends to have HIGHER values than g2.
    (u_stat must be the U statistic scipy returns for the FIRST
    argument passed to mannwhitneyu, i.e. mannwhitneyu(g1, g2).)"""
    n1, n2 = len(g1), len(g2)
    return (2 * u_stat) / (n1 * n2) - 1

def hodges_lehmann(g1, g2):
    """Median of all pairwise differences g1_i - g2_j.
    Robust point estimate of the typical shift between groups, in the
    original measurement units (here: grade points)."""
    diffs = np.subtract.outer(np.asarray(g1), np.asarray(g2)).ravel()
    return np.median(diffs)

ordinal_posthoc_rows = []
for col in ordinal_composite:
    groups = {d: data.loc[data["diagnosis"] == d, col].dropna() for d in diagnoses}
    groups = {d: g for d, g in groups.items() if len(g) > 0}
    if len(groups) < 2:
        continue

    omnibus_stat, omnibus_p = kruskal(*groups.values())

    for d1, d2 in combinations(groups.keys(), 2):
        g1, g2 = groups[d1], groups[d2]
        degenerate = len(g1) == 0 or len(g2) == 0 or (set(g1) == set(g2) and len(set(g1)) == 1)

        if degenerate:
            u_p, u_stat, note = 1.0, np.nan, "degenerate (no variation) - not tested"
        else:
            try:
                u_stat, u_p = mannwhitneyu(g1, g2, alternative="two-sided")
                note = ""
            except ValueError:
                u_p, u_stat, note = 1.0, np.nan, "mannwhitneyu raised ValueError - treated as p=1.0"

        r_rb = rank_biserial(g1, g2, u_stat) if not degenerate and note == "" else np.nan
        hl = hodges_lehmann(g1, g2) if len(g1) > 0 and len(g2) > 0 else np.nan

        ordinal_posthoc_rows.append({
            "Feature": col,
            "Comparison": f"{d1} vs {d2}",
            "p_value": u_p,
            "omnibus_p": omnibus_p,
            "rank_biserial_r": round(r_rb, 3) if pd.notnull(r_rb) else np.nan,
            "hodges_lehmann_shift": round(hl, 2) if pd.notnull(hl) else np.nan,
            "n1": len(g1), "n2": len(g2),
            "note": note,
        })

df_ordinal_posthoc = pd.DataFrame(ordinal_posthoc_rows)
df_ordinal_posthoc["p_adj"] = multipletests(df_ordinal_posthoc["p_value"], method="fdr_bh")[1]
df_ordinal_posthoc = df_ordinal_posthoc.sort_values(["Feature", "p_adj"])

print("=" * 70)
print("Ordinal post-hoc (pairwise Mann-Whitney, FDR-corrected)")
print("=" * 70)
print(df_ordinal_posthoc.to_string(index=False))

df_ordinal_posthoc.to_csv(f"{results_path}/mri_ordinal_posthoc_results.csv", index=False)

print("""
Reading rank_biserial_r: sign shows direction (positive = first-listed
group in Comparison has HIGHER values), magnitude ~0.1/0.3/0.5 =
small/medium/large, same rough convention as Cohen's d.

Reading hodges_lehmann_shift: the typical difference in grade points
between the two groups' distributions - e.g. a shift of 1.0 for
"healthy vs ss" means SS patients typically score about 1 grade point
higher than healthy controls on that feature. More directly quotable
in a results paragraph than rank-biserial r, at the cost of being in
raw units rather than a standardized scale.

Rows only reach FDR<.05 meaningfully if omnibus_p for that Feature was
already <.05 - check omnibus_p before treating a small p_adj here as
a confirmed pairwise difference.
""")

# %%