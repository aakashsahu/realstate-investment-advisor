"""
Real Estate Investment Advisor - Exploratory Data Analysis (single, self-contained file)

Run:  python eda.py
Needs: pandas, numpy, matplotlib, seaborn
Looks for india_housing_prices.csv in ./data/ or next to this file.
Creates: data/cleaned_housing.csv, figures/*.png, reports/eda_findings.json
"""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11})

BASE = os.path.dirname(os.path.abspath(__file__))
for d in ("data", "figures", "reports"):
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
CSV = next((p for p in (os.path.join(BASE, "data", "india_housing_prices.csv"), os.path.join(BASE, "india_housing_prices.csv")) if os.path.exists(p)), None)
if CSV is None:
    raise SystemExit("india_housing_prices.csv not found - put it in the data/ folder.")

CURRENT_YEAR = 2025
AMENITIES = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
F = {}  # findings collected for the report


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "figures", f"{name}.png")); plt.close()


# ------------------------------------------------------------------ 1. LOAD + QUALITY CHECKS
raw = pd.read_csv(CSV)
F.update(rows_raw=len(raw), columns=raw.shape[1], missing_total=int(raw.isna().sum().sum()),
         duplicates=int(raw.drop(columns="ID").duplicated().sum()))
print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns | missing: {F['missing_total']} | duplicates: {F['duplicates']}")

# ------------------------------------------------------------------ 2. CLEANING
df = raw.drop_duplicates(subset=[c for c in raw.columns if c != "ID"]).copy()
df["Floor_Invalid"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)      # data-quality flag
df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])               # cap impossible floors
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
F["floor_invalid_pct"] = round(df.Floor_Invalid.mean() * 100, 1)

# ------------------------------------------------------------------ 3. FEATURE ENGINEERING
df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
for a in AMENITIES:
    df[f"Has_{a}"] = df["Amenities"].str.contains(a).astype(int)
df["Amenity_Count"] = df[[f"Has_{a}" for a in AMENITIES]].sum(axis=1)
df["Transport_Score"] = df["Public_Transport_Accessibility"].map({"Low": 0, "Medium": 1, "High": 2})
df["School_Density_Score"] = df["Nearby_Schools"] / 10
df["Infra_Score"] = (df["Nearby_Schools"] / 10 + df["Nearby_Hospitals"] / 10 + df["Transport_Score"] / 2) / 3
df["Parking"] = (df["Parking_Space"] == "Yes").astype(int)
df["Is_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
df["City_Median_PPSF"] = df["City"].map(df.groupby("City")["Price_per_SqFt_INR"].median())
df["PPSF_vs_City"] = df["Price_per_SqFt_INR"] / df["City_Median_PPSF"]

# Target 1: feature-based annual growth assumption -> price after 5 years (assumption, not observed data)
df["Growth_Rate"] = (0.06 + 0.02 * df["Transport_Score"] / 2 + 0.02 * df["Infra_Score"] + 0.01 * df["Amenity_Count"] / 5
                     - 0.0004 * df["Age_of_Property"] + 0.005 * df["Is_Ready"]).clip(0.03, 0.12)
df["Future_Price_5y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5
# Target 2: Good Investment = cheaper than city median AND quality score >= 3 (of 6)
df["Quality_Score"] = ((df["BHK"] >= 3).astype(int) + df["Is_Ready"] + (df["Transport_Score"] == 2).astype(int)
                       + df["Parking"] + (df["Security"] == "Yes").astype(int) + (df["Infra_Score"] >= 0.5).astype(int))
df["Good_Investment"] = ((df["PPSF_vs_City"] <= 1.0) & (df["Quality_Score"] >= 3)).astype(int)
df.to_csv(os.path.join(BASE, "data", "cleaned_housing.csv"), index=False)
F.update(rows_clean=len(df), good_rate=round(df.Good_Investment.mean() * 100, 1), growth_mean=round(df.Growth_Rate.mean() * 100, 2))
PPSF = df["Price_per_SqFt_INR"]

# ------------------------------------------------------------------ 4. EDA (20 questions)
# Q1-2 distributions
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
sns.histplot(df.Price_in_Lakhs, bins=40, ax=ax[0]); ax[0].set_title("Q1 Price distribution (Lakhs)")
sns.histplot(df.Size_in_SqFt, bins=40, ax=ax[1], color="C1"); ax[1].set_title("Q2 Size distribution (sq ft)"); save("q01_02_distributions")
F.update(price_mean=round(df.Price_in_Lakhs.mean(), 1), price_median=round(df.Price_in_Lakhs.median(), 1), size_mean=round(df.Size_in_SqFt.mean()))

# Q3 PPSF by property type
plt.figure(figsize=(6, 3.4)); sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_INR", showfliers=False)
plt.title("Q3 Price per sq ft (INR) by property type"); save("q03_ppsf_type")
F["ppsf_by_type"] = df.groupby("Property_Type").Price_per_SqFt_INR.mean().round(0).to_dict()

# Q4 size vs price
s = df.sample(min(8000, len(df)), random_state=1); plt.figure(figsize=(6, 3.6))
plt.scatter(s.Size_in_SqFt, s.Price_in_Lakhs, s=4, alpha=.4); plt.xlabel("Size (sq ft)"); plt.ylabel("Price (Lakhs)")
plt.title("Q4 Size vs price (sample)"); save("q04_size_price")
F["corr_size_price"] = round(df.Size_in_SqFt.corr(df.Price_in_Lakhs), 3)

# Q5 outliers (IQR rule)
def iqr_out(x):
    q1, q3 = x.quantile([.25, .75]); k = 1.5 * (q3 - q1)
    return int(((x < q1 - k) | (x > q3 + k)).sum())
F.update(outliers_ppsf=iqr_out(PPSF), outliers_size=iqr_out(df.Size_in_SqFt), outliers_price=iqr_out(df.Price_in_Lakhs))
plt.figure(figsize=(6, 2.6)); sns.boxplot(x=PPSF); plt.title("Q5 Price per sq ft outliers (INR)"); save("q05_outliers")

# Q6 avg PPSF by state
st = df.groupby("State").Price_per_SqFt_INR.mean().sort_values()
plt.figure(figsize=(6.5, 5)); st.plot.barh(); plt.title("Q6 Avg price per sq ft by state (INR)"); save("q06_state_ppsf")
F["state_ppsf_range"] = {"lowest": [st.index[0], round(st.iloc[0])], "highest": [st.index[-1], round(st.iloc[-1])]}

# Q7 avg price by city
ct = df.groupby("City").Price_in_Lakhs.mean().sort_values()
pd.concat([ct.head(8), ct.tail(8)]).plot.barh(figsize=(6.5, 4.5), color="C2")
plt.title("Q7 Avg price by city: lowest 8 vs highest 8 (Lakhs)"); plt.xlim(ct.min() - 5, None); save("q07_city_price")
F["city_price_range"] = {"lowest": [ct.index[0], round(ct.iloc[0], 1)], "highest": [ct.index[-1], round(ct.iloc[-1], 1)],
                         "spread_pct": round((ct.max() / ct.min() - 1) * 100, 1)}

# Q8 median age by locality
la = df.groupby("Locality").Age_of_Property.median()
plt.figure(figsize=(6, 3.2)); sns.histplot(la, bins=15); plt.title("Q8 Median property age by locality (years)"); save("q08_locality_age")
F["locality_median_age_range"] = [float(la.min()), float(la.max())]

# Q9 BHK distribution across cities
top = df.City.value_counts().head(10).index
pd.crosstab(df[df.City.isin(top)].City, df.BHK, normalize="index").plot.bar(stacked=True, figsize=(7, 3.6), colormap="viridis")
plt.title("Q9 BHK mix, 10 largest cities"); plt.legend(title="BHK", bbox_to_anchor=(1, 1)); save("q09_bhk_city")

# Q10 top-5 expensive localities (no date column -> build-era used as proxy for trend)
lp = df.groupby("Locality").Price_per_SqFt_INR.mean().sort_values(ascending=False).head(5)
F["top5_localities_ppsf"] = {k: round(v) for k, v in lp.items()}
d5 = df[df.Locality.isin(lp.index)].copy()
d5["Build_Era"] = pd.cut(d5.Year_Built, [1989, 2000, 2010, 2023], labels=["1990-2000", "2001-2010", "2011-2023"])
plt.figure(figsize=(6.5, 3.4)); sns.pointplot(data=d5, x="Build_Era", y="Price_per_SqFt_INR", hue="Locality", errorbar=None)
plt.title("Q10 Top-5 localities: PPSF by build era (proxy for trend)"); save("q10_top_localities")

# Q11 correlation heatmap
num = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
       "Floor_No", "Total_Floors", "Amenity_Count", "Transport_Score", "Infra_Score"]
c = df[num].corr(); plt.figure(figsize=(7, 5.6))
sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 6}); plt.title("Q11 Correlation matrix"); save("q11_corr")
F["corr_with_price"] = c["Price_in_Lakhs"].drop("Price_in_Lakhs").round(3).to_dict()

# Q12-13 schools / hospitals vs PPSF
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
sns.barplot(data=df, x="Nearby_Schools", y="Price_per_SqFt_INR", errorbar=None, ax=ax[0], color="C0"); ax[0].set_title("Q12 PPSF by nearby schools")
sns.barplot(data=df, x="Nearby_Hospitals", y="Price_per_SqFt_INR", errorbar=None, ax=ax[1], color="C3"); ax[1].set_title("Q13 PPSF by nearby hospitals"); save("q12_13_schools_hospitals")
F["corr_ppsf_schools"] = round(PPSF.corr(df.Nearby_Schools), 3); F["corr_ppsf_hospitals"] = round(PPSF.corr(df.Nearby_Hospitals), 3)

# Q14-15 furnishing / facing
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
sns.barplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", errorbar=None, ax=ax[0]); ax[0].set_title("Q14 Avg price by furnishing")
sns.barplot(data=df, x="Facing", y="Price_per_SqFt_INR", errorbar=None, ax=ax[1], color="C4"); ax[1].set_title("Q15 Avg PPSF by facing"); save("q14_15_furnish_facing")
F["price_by_furnishing"] = df.groupby("Furnished_Status").Price_in_Lakhs.mean().round(1).to_dict()
F["ppsf_by_facing"] = df.groupby("Facing").Price_per_SqFt_INR.mean().round(0).to_dict()

# Q16-17 owner type / availability
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
df.Owner_Type.value_counts().plot.bar(ax=ax[0], rot=0); ax[0].set_title("Q16 Listings by owner type")
df.Availability_Status.value_counts().plot.bar(ax=ax[1], rot=0, color="C1"); ax[1].set_title("Q17 Listings by availability"); save("q16_17_owner_avail")
F["owner_counts"] = df.Owner_Type.value_counts().to_dict(); F["availability_counts"] = df.Availability_Status.value_counts().to_dict()

# Q18-19 parking / amenities
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", errorbar=None, ax=ax[0]); ax[0].set_title("Q18 Avg price by parking")
am = pd.Series({a: PPSF[df[f"Has_{a}"] == 1].mean() - PPSF[df[f"Has_{a}"] == 0].mean() for a in AMENITIES})
am.plot.bar(ax=ax[1], rot=30, color="C2"); ax[1].set_title("Q19 PPSF difference: has vs lacks amenity (INR)"); save("q18_19_parking_amenities")
F["price_by_parking"] = df.groupby("Parking_Space").Price_in_Lakhs.mean().round(1).to_dict(); F["amenity_ppsf_difference"] = am.round(0).to_dict()

# Q20 public transport vs PPSF and investment potential
order = ["Low", "Medium", "High"]; fig, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt_INR", order=order, errorbar=None, ax=ax[0]); ax[0].set_title("Q20 PPSF by transport access")
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=order, errorbar=None, ax=ax[1], color="C1"); ax[1].set_title("Good-investment rate by transport"); save("q20_transport")
F["good_rate_by_transport"] = df.groupby("Public_Transport_Accessibility").Good_Investment.mean().round(3).to_dict()

# ------------------------------------------------------------------ 5. SAVE FINDINGS
with open(os.path.join(BASE, "reports", "eda_findings.json"), "w") as f:
    json.dump(F, f, indent=1, default=str)
print(json.dumps(F, indent=1, default=str))
print("\nDone: data/cleaned_housing.csv, figures/ (20 questions), reports/eda_findings.json")
