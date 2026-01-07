import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import plot_importance
import matplotlib.pyplot as plt
import io

# ======================
# CONFIG PAGE (must run before other Streamlit calls)
# ======================
st.set_page_config(
    page_title="LoanRisk News Analytics",
    layout="wide"
)

# ======================
# LOAD MODEL & ENCODER
# ======================
@st.cache(allow_output_mutation=True)
def load_artifacts():
    try:
        m = joblib.load("xgb_model.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        m = None
    try:
        enc = joblib.load("encoder.pkl")
    except Exception as e:
        st.error(f"Gagal memuat encoder: {e}")
        enc = {}
    return m, enc

model, encoders = load_artifacts()

st.title("📊 LoanRisk News Analytics")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi **risiko gagal bayar kredit** menggunakan **XGBoost**.
Fitur baru: preset input, prediksi batch (upload CSV), dan unduh hasil prediksi.
""")

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("🧾 Input Data Nasabah")

# --- use session state keys for preset support
def user_input():
    Age = st.sidebar.number_input("Age", 18, 70, key="Age")
    Income = st.sidebar.number_input("Income", 0, key="Income")
    LoanAmount = st.sidebar.number_input("Loan Amount", 0, key="LoanAmount")
    CreditScore = st.sidebar.number_input("Credit Score", 300, 850, key="CreditScore")
    MonthsEmployed = st.sidebar.number_input("Months Employed", 0, key="MonthsEmployed")
    NumCreditLines = st.sidebar.number_input("Number of Credit Lines", 0, key="NumCreditLines")
    InterestRate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, key="InterestRate")
    LoanTerm = st.sidebar.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], key="LoanTerm")
    DTIRatio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, key="DTIRatio")

    Education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"], key="Education")
    EmploymentType = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"], key="EmploymentType")
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="MaritalStatus")
    HasMortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"], key="HasMortgage")
    HasDependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"], key="HasDependents")
    LoanPurpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Medical"], key="LoanPurpose")
    HasCoSigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"], key="HasCoSigner")

    data = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "Education": Education,
        "EmploymentType": EmploymentType,
        "MaritalStatus": MaritalStatus,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "LoanPurpose": LoanPurpose,
        "HasCoSigner": HasCoSigner
    }

    return pd.DataFrame([data])


# --- Preset buttons
st.sidebar.markdown("---")
st.sidebar.subheader("Contoh Preset")
if st.sidebar.button("Profil Berisiko Tinggi"):
    st.session_state.update({
        "Age": 20,
        "Income": 0,
        "LoanAmount": 30000,
        "CreditScore": 300,
        "MonthsEmployed": 0,
        "NumCreditLines": 15,
        "InterestRate": 30.0,
        "LoanTerm": 60,
        "DTIRatio": 1.0,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Personal",
        "HasCoSigner": "No"
    })
if st.sidebar.button("Profil Aman"):
    st.session_state.update({
        "Age": 45,
        "Income": 8000,
        "LoanAmount": 5000,
        "CreditScore": 750,
        "MonthsEmployed": 120,
        "NumCreditLines": 3,
        "InterestRate": 6.5,
        "LoanTerm": 36,
        "DTIRatio": 0.2,
        "Education": "Master",
        "EmploymentType": "Salaried",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "No",
        "LoanPurpose": "Education",
        "HasCoSigner": "Yes"
    })

input_df = user_input()

# ======================
# ENCODING INPUT
# ======================
def safe_transform(series, encoder):
    # Defensive: ensure encoder is usable
    if encoder is None:
        return series

    if not hasattr(encoder, "transform"):
        return series

    # Try several ways to call transform (1D / 2D inputs)
    attempts = []
    attempts.append(lambda s: encoder.transform(s))
    attempts.append(lambda s: encoder.transform(s.values))
    attempts.append(lambda s: encoder.transform(s.values.reshape(-1, 1)))

    last_exc = None
    for fn in attempts:
        try:
            res = fn(series)
            # handle sparse matrix
            try:
                from scipy import sparse
                if sparse.issparse(res):
                    res = res.toarray()
            except Exception:
                pass

            arr = np.asarray(res)
            if arr.ndim == 2:
                if arr.shape[1] == 1:
                    return pd.Series(arr.ravel(), index=series.index)
                else:
                    st.warning(f"Encoder untuk kolom menghasilkan {arr.shape[1]} kolom; menggunakan kolom pertama.")
                    return pd.Series(arr[:, 0], index=series.index)
            else:
                return pd.Series(arr, index=series.index)
        except Exception as e:
            last_exc = e
            continue

    # fallback: if encoder exposes classes_, map values
    if hasattr(encoder, "classes_"):
        try:
            mapping = {c: i for i, c in enumerate(encoder.classes_)}
            return series.map(lambda x: mapping.get(x, -1)).astype(int)
        except Exception:
            pass

    # final fallback: warn and return original series
    st.warning(f"Gagal mentransform kolom dengan encoder ({last_exc}); menggunakan nilai asli.")
    return series


def prepare_input_for_model(df, model):
    """Align dataframe columns to model expected features. Fill missing with 0 and drop extras."""
    if model is None:
        return df, "Model not loaded"

    expected = None
    try:
        booster = model.get_booster()
        expected = booster.feature_names
    except Exception:
        pass

    if expected is None:
        # try sklearn attribute
        expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        return df, "Tidak dapat menentukan fitur model; menggunakan dataframe asli."

    # expected may be list-like or numpy array
    expected = list(expected)

    # create aligned df
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    aligned = df.copy()
    # add missing with zeros
    for m in missing:
        aligned[m] = 0

    # drop extras
    aligned = aligned[[c for c in expected if c in aligned.columns]]

    msg = []
    if missing:
        msg.append(f"Menambahkan kolom hilang: {missing}")
    if extra:
        msg.append(f"Menghapus kolom tambahan: {extra}")

    return aligned, "; ".join(msg) if msg else "Columns aligned"

def display_progress(prob):
    """Display progress bar accepting numpy types; scale to 0-100 integer."""
    if prob is None:
        return
    try:
        val = float(prob)
    except Exception:
        return
    val = max(0.0, min(1.0, val))
    try:
        st.progress(int(val * 100))
    except Exception:
        # final fallback: ignore progress
        return

for col, encoder in encoders.items():
    if col in input_df.columns:
        try:
            transformed = safe_transform(input_df[col], encoder)
            # only assign if lengths match
            if isinstance(transformed, pd.Series) and len(transformed) == len(input_df):
                input_df[col] = transformed
            else:
                st.warning(f"Transform untuk kolom '{col}' mengembalikan bentuk tak terduga; menggunakan nilai asli.")
        except Exception as e:
            st.warning(f"Gagal mentransform kolom '{col}': {e}; menggunakan nilai asli.")

# ======================
# PREDIKSI
# ======================
# --- Layout: tabs for single / batch / model info
tab1, tab2, tab3 = st.tabs(["Single Predict", "Batch Predict", "Model Info"])

with tab1:
    st.subheader("Prediksi Untuk 1 Nasabah")
    st.write(input_df)
    if st.button("🔍 Prediksi Risiko"):
        if model is None:
            st.error("Model belum tersedia.")
        else:
            try:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
            except Exception as e:
                # try aligning columns and retry
                aligned, msg = prepare_input_for_model(input_df, model)
                if msg:
                    st.warning(msg)
                try:
                    pred = model.predict(aligned)[0]
                    prob = model.predict_proba(aligned)[0][1]
                except Exception as e2:
                    st.error(f"Prediksi gagal: {e2}")
                    pred = None
                    prob = None

            if pred is not None:
                st.subheader("📌 Hasil Prediksi")
                st.metric(label="Probabilitas Gagal Bayar", value=f"{prob:.2%}")
                display_progress(prob)

                if pred == 1:
                    st.error(f"⚠️ Berisiko Gagal Bayar (Probabilitas: {prob:.2%})")
                else:
                    st.success(f"✅ Tidak Berisiko Gagal Bayar (Probabilitas: {prob:.2%})")

with tab2:
    st.subheader("Prediksi Batch (Upload CSV)")
    st.markdown("Unggah file CSV yang berisi kolom input yang sama seperti form sidebar.")
    uploaded = st.file_uploader("Pilih CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(batch_df.head())

            # apply encoders to matching columns
            for col, encoder in encoders.items():
                if col in batch_df.columns:
                    try:
                        transformed = safe_transform(batch_df[col], encoder)
                        if isinstance(transformed, pd.Series) and len(transformed) == len(batch_df):
                            batch_df[col] = transformed
                        else:
                            st.warning(f"Transform untuk kolom '{col}' pada batch mengembalikan bentuk tak terduga; menggunakan nilai asli.")
                    except Exception as e:
                        st.warning(f"Gagal mentransform kolom '{col}' pada batch: {e}; menggunakan nilai asli.")

            if model is None:
                st.error("Model belum tersedia untuk prediksi batch.")
            else:
                try:
                    preds = model.predict(batch_df)
                    probs = model.predict_proba(batch_df)[:, 1]
                except Exception:
                    aligned_batch, msg = prepare_input_for_model(batch_df, model)
                    if msg:
                        st.warning(msg)
                    try:
                        preds = model.predict(aligned_batch)
                        probs = model.predict_proba(aligned_batch)[:, 1]
                        # if aligned changed column order, map back to original index
                        batch_df = aligned_batch.copy()
                    except Exception as e:
                        st.error(f"Prediksi batch gagal: {e}")
                        preds = None
                        probs = None
                if preds is not None:
                    batch_df["pred_risk"] = preds
                    batch_df["probability"] = probs
                else:
                    st.error("Prediksi batch tidak dapat diselesaikan.")
                st.write(batch_df.head())

                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("Unduh Hasil Prediksi (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# ======================
# FEATURE IMPORTANCE
# ======================
with tab3:
    st.subheader("📊 Feature Importance Model")
    # Try to extract importance in multiple ways
    try:
        booster = model.get_booster()
        imp = booster.get_score(importance_type="gain")
        imp_df = pd.DataFrame.from_dict(imp, orient="index", columns=["gain"]).reset_index()
        imp_df.columns = ["feature", "gain"]
        imp_df = imp_df.sort_values("gain", ascending=False).head(15)
        st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        try:
            fi = model.feature_importances_
            imp_df = pd.DataFrame({"feature": range(len(fi)), "importance": fi}).sort_values("importance", ascending=False).head(15)
            st.bar_chart(imp_df.set_index("feature"))
        except Exception as e:
            st.info("Tidak dapat menampilkan feature importance untuk model ini.")
