Shubh03, [3/13/26 1:35 PM]
%%writefile app.py
# app.py
import os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# ----------------------------
# Config
# ----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def model_paths(sector):
    return (
        os.path.join(MODEL_DIR, f"credit_risk_model_{sector}.pkl"),
        os.path.join(MODEL_DIR, f"feature_columns_{sector}.pkl"),
        os.path.join(MODEL_DIR, f"district_defaults_{sector}.pkl"),
    )

TARGET_NAME = "default"

FEATURES_ALLOWLIST = [
    "liabilities","education_level","migrant_score",
    "cibil_score","itr_filed","sip_investment","insurance_paid",
    "upi_txn_freq","bill_payment","ecommerce_spend","mobile_usage",
    "pmkisan_score","svanidhi_score","jan_dhan_score",
    "on_time_ratio","assets_amount","skill_level",
    "telecom_density","district_cd_ratio","npa_agri","internet_penetration",
    "geo_stability","branch_density","digital_penetration","digital_banking"
]

# ----------------------------
# Helpers
# ----------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
    )
    return df

def sector_feature_mask(sector):
    feats = FEATURES_ALLOWLIST.copy()
    if sector == "Agriculture":
        for f in ["cibil_score","itr_filed","sip_investment","insurance_paid","svanidhi_score"]:
            if f in feats: feats.remove(f)
    elif sector == "Informal":
        for f in ["cibil_score","itr_filed","sip_investment","insurance_paid","pmkisan_score"]:
            if f in feats: feats.remove(f)
    elif sector == "Service":
        for f in ["pmkisan_score","svanidhi_score"]:
            if f in feats: feats.remove(f)
    return feats

def train_and_save(df: pd.DataFrame, sector: str):
    df = clean_cols(df)
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found.")

    feats = sector_feature_mask(sector)
    work = df[feats + [TARGET_NAME] + ["district"]].copy()

    for c in feats + [TARGET_NAME]:
        if work[c].dtype == "object":
            work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[feats]
    y = work[TARGET_NAME].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.9, random_state=42
        )),
    ])
    pipe.fit(Xtr, ytr)

    yprob = pipe.predict_proba(Xte)[:,1]
    acc = accuracy_score(yte, pipe.predict(Xte))
    auc = roc_auc_score(yte, yprob)

    district_defaults = work.groupby("district")[[
        "geo_stability","npa_agri","telecom_density","internet_penetration",
        "district_cd_ratio","branch_density","digital_penetration","digital_banking"
    ]].median(numeric_only=True).to_dict(orient="index")

    model_path, features_path, district_path = model_paths(sector)
    with open(model_path, "wb") as f: pickle.dump(pipe, f)
    with open(features_path, "wb") as f: pickle.dump(feats, f)
    with open(district_path, "wb") as f: pickle.dump(district_defaults, f)

    return acc, auc

def load_model(sector: str):

Shubh03, [3/13/26 1:35 PM]
model_path, features_path, district_path = model_paths(sector)
    with open(model_path, "rb") as f: pipe = pickle.load(f)
    with open(features_path, "rb") as f: feats = pickle.load(f)
    if os.path.exists(district_path):
        with open(district_path, "rb") as f: district_defaults = pickle.load(f)
    else:
        district_defaults = {}
    return pipe, feats, district_defaults

def lti_multiplier(lti: float) -> float:
    """Monotonic adjustment: lower LTI = safer, higher = riskier."""
    if lti < 0.5: return 0.85
    elif lti < 1: return 1.0
    elif lti < 2: return 1.2
    else: return 1.5

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Jharkhand Credit Risk — Sector Models + District Aware", layout="wide")
st.title("🏦 Jharkhand Borrower Credit Risk — Sector Models + District Aware")

sector = st.selectbox("Sector", ["Agriculture","Informal","Service"])
uploaded = st.sidebar.file_uploader("Upload training CSV (first run)", type=["csv"])
retrain = st.sidebar.checkbox("Retrain all sector models?", value=False)

# load/train
if retrain:
    if uploaded is None:
        st.warning("⚠️ Upload dataset to retrain.")
        st.stop()
    df_uploaded = pd.read_csv(uploaded)
    results = {}
    for s in ["Agriculture","Informal","Service"]:
        acc, auc = train_and_save(df_uploaded, s)
        results[s] = (acc, auc)
    st.success("✅ Trained all sector models successfully!")
    st.write(pd.DataFrame(results, index=["Accuracy","AUC"]).T)
try:
    model, model_features, district_defaults = load_model(sector)
    st.success(f"✅ Loaded {sector} model.")
except Exception:
    st.error(f"No saved model found for {sector}. Retrain with dataset.")
    st.stop()

# ----------------------------
# Sector rules summary table
# ----------------------------
st.subheader("📑 Sector Feature Rules")
rules = {
    "Agriculture":"Schemes important (PM-Kisan, Jan Dhan). Formal excluded.",
    "Informal":"Schemes important (SVANidhi, Jan Dhan). Formal excluded.",
    "Service":"Formal (CIBIL/ITR/SIP/Insurance) important. Schemes excluded.",
}
st.table(pd.DataFrame.from_dict(rules, orient="index", columns=["Model Focus"]))

# ----------------------------
# Borrower Inputs
# ----------------------------
st.subheader("👤 Borrower Inputs")

district = st.selectbox("District", sorted(district_defaults.keys()) if district_defaults else ["Ranchi"])

dvals = district_defaults.get(district, {
    "geo_stability":0.6,"npa_agri":0.15,"telecom_density":0.75,"internet_penetration":0.65,
    "district_cd_ratio":0.55,"branch_density":0.5,"digital_penetration":0.65,"digital_banking":0.5
})

income = st.number_input("Income (₹/mo)", 1000, 100000, 5000, 500)
loan_amount = st.number_input("Loan Amount (₹)", 1000, 500000, 10000, 1000)
liabilities = st.slider("Dependents (0–5)", 0, 5, 2)
education_level = st.slider("Education Level", 0, 5, 2)
migrant_score = st.slider("Migrant Score", 0.0, 1.0, 0.5)
assets_amount = st.number_input("Assets (₹)", 0, 1000000, 5000, 500)
skill_level = st.slider("Skill Level (0–2)", 0, 2, 1)
on_time_ratio = st.slider("On-Time Payment Ratio", 0.0, 1.0, 0.7)

cibil_score = st.number_input("CIBIL Score", 300, 900, 650)
itr_filed = st.selectbox("ITR Filed?", ["Yes","No"])
sip_investment = st.number_input("SIP Investment (₹/mo)", 0, 100000, 0, 500)
insurance_paid = st.number_input("Insurance Paid (₹/yr)", 0, 100000, 0, 500)

upi_txn_freq = st.number_input("UPI Txn / month", 0, 1000, 10, 1)
bill_payment = st.number_input("Bill Payment (₹/mo)", 0, 100000, 600, 500)
ecommerce_spend = st.number_input("E-commerce Spend (₹/mo)", 0, 100000, 600, 500)
mobile_usage = st.number_input("Mobile Usage (GB/mo)", 0, 1000, 20, 1)

pmkisan_score = st.slider("PM-Kisan Score", 0.0, 1.0, 0.5)
svanidhi_score = st.slider("SVANidhi Score", 0.0, 1.0, 0.5)
jan_dhan_score = st.slider("Jan Dhan Score", 0.0, 1.0, 0.7)

Shubh03, [3/13/26 1:35 PM]
geo_stability = st.number_input("Geo Stability", 0.0, 1.0, float(dvals["geo_stability"]), 0.01)
npa_agri = st.number_input("NPA Agri", 0.0, 1.0, float(dvals["npa_agri"]), 0.01)
telecom_density = st.number_input("Telecom Density", 0.0, 1.0, float(dvals["telecom_density"]), 0.01)
internet_penetration = st.number_input("Internet Penetration", 0.0, 1.0, float(dvals["internet_penetration"]), 0.01)
district_cd_ratio = st.number_input("District CD Ratio", 0.0, 1.0, float(dvals["district_cd_ratio"]), 0.01)
branch_density = st.number_input("Branch Density", 0.0, 1.0, float(dvals["branch_density"]), 0.01)
digital_penetration = st.number_input("Digital Penetration", 0.0, 1.0, float(dvals["digital_penetration"]), 0.01)
digital_banking = st.number_input("Digital Banking", 0.0, 1.0, float(dvals["digital_banking"]), 0.01)

# ----------------------------
# Build input row
# ----------------------------
input_row = {
    "liabilities": liabilities,
    "education_level": education_level,
    "migrant_score": migrant_score,
    "cibil_score": cibil_score,
    "itr_filed": 1 if itr_filed=="Yes" else 0,
    "sip_investment": sip_investment,
    "insurance_paid": insurance_paid,
    "upi_txn_freq": upi_txn_freq,
    "bill_payment": bill_payment,
    "ecommerce_spend": ecommerce_spend,
    "mobile_usage": mobile_usage,
    "pmkisan_score": pmkisan_score,
    "svanidhi_score": svanidhi_score,
    "jan_dhan_score": jan_dhan_score,
    "on_time_ratio": on_time_ratio,
    "assets_amount": assets_amount,
    "skill_level": skill_level,
    "telecom_density": telecom_density,
    "district_cd_ratio": district_cd_ratio,
    "npa_agri": npa_agri,
    "internet_penetration": internet_penetration,
    "geo_stability": geo_stability,
    "branch_density": branch_density,
    "digital_penetration": digital_penetration,
    "digital_banking": digital_banking,
}
X_input = pd.DataFrame([input_row], columns=model_features)

# ----------------------------
# Predict + SHAP
# ----------------------------
if st.button("🔮 Predict Risk"):
    try:
        raw_pd = float(model.predict_proba(X_input)[0,1])

        # Apply LTI multiplier
        lti = loan_amount / (income + 1)
        final_pd = float(np.clip(raw_pd * lti_multiplier(lti), 1e-4, 0.9999))

        st.metric("Raw PD", f"{raw_pd*100:.2f}%")
        st.metric("LTI Multiplier", f"{lti_multiplier(lti):.2f}x")
        st.success(f"🎯 Final Probability of Default: {final_pd*100:.2f}%")

        # SHAP
        imp = model.named_steps["imp"]
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["gb"]

        X_input_tr = scaler.transform(imp.transform(X_input))
        explainer = shap.TreeExplainer(clf)
        phi = explainer.shap_values(X_input_tr)[0]

        # SHAP Waterfall
        shap.initjs()
        exp = shap.Explanation(values=phi, base_values=explainer.expected_value,
                               data=X_input.iloc[0].values, feature_names=model_features)
        fig1 = plt.figure(figsize=(8,5))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig1, use_container_width=True)

        # Group contributions
        groups = {
            "capacity":["liabilities","education_level","migrant_score",
                        "on_time_ratio","assets_amount","skill_level"],
            "formal":["cibil_score","itr_filed","sip_investment","insurance_paid"],
            "digital":["upi_txn_freq","bill_payment","ecommerce_spend","mobile_usage",
                       "digital_penetration","digital_banking"],
            "schemes":["pmkisan_score","svanidhi_score","jan_dhan_score"],
            "district":["geo_stability","npa_agri","telecom_density","internet_penetration",
                        "district_cd_ratio","branch_density"],
        }
        contrib = {}
        for g, flist in groups.items():
            contrib[g] = np.sum([abs(phi[model_features.index(f)]) for f in flist if f in model_features])

Shubh03, [3/13/26 1:35 PM]
weights = {
            "Agriculture": {"capacity":0.8, "digital":1.2, "schemes":2.0, "district":1.0, "formal":0.0},
            "Informal":    {"capacity":0.8, "digital":1.2, "schemes":1.8, "district":1.0, "formal":0.0},
            "Service":     {"capacity":0.9, "digital":1.1, "schemes":0.0, "district":1.0, "formal":1.2},
        }
        w = weights.get(sector, {g:1.0 for g in groups})
        for g in contrib:
            contrib[g] *= w.get(g,1.0)

        total = sum(contrib.values())
        contrib_norm = {k:v/total for k,v in contrib.items() if total>0}

        fig2, ax = plt.subplots(figsize=(6,4))
        pd.Series(contrib_norm).plot.bar(ax=ax)
        ax.set_ylabel("Normalized Contribution")
        ax.set_title(f"Feature Group Contributions (weighted for {sector})")
        st.pyplot(fig2, use_container_width=True)

        # ----------------------------
        # Natural language summary
        # ----------------------------
        top_group = max(contrib_norm, key=contrib_norm.get)
        st.markdown(f"""
        ### 📝 Model Interpretation
        This borrower has an estimated {final_pd*100:.2f}% probability of default.
        
        - The most influential factor group for this case is {top_group.capitalize()}.  
        - Sector-specific rules place higher weight on: {rules[sector]}  
        - Overall, repayment capacity, digital activity, and scheme participation balance together to drive the score.  
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")