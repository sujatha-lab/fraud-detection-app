# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:20:26 2025

@author: grace
"""

# app.py
import streamlit as st
st.set_page_config(page_title="Fraud Detection - Supervised vs Unsupervised", layout="wide")

import pandas as pd
import numpy as np
import time
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

# optional heavy libs — catch import errors and show friendly message
missing = []
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
    missing.append("imbalanced-learn (imblearn)")

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
except Exception:
    missing.append("scikit-learn")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
    missing.append("xgboost")

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
    missing.append("lightgbm")

# ---------- helper plotting functions ----------
def fig_countplot(y, title="Class distribution"):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=y, palette="Set2", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

def fig_heatmap(df, title="Correlation heatmap"):
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def fig_confusion(cm, model_name):
    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig

def fig_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, linewidth=3, label=f'{label} (AUC={roc_auc_score(y_true, y_score):.4f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ---------- evaluation ----------
def evaluate_model(y_test, y_pred, y_proba, model_name):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    lift = recall / (sum(y_test) / len(y_test)) if len(y_test)>0 else np.nan

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC-ROC": auc,
        "Specificity": specificity,
        "Lift": lift,
        "Confusion Matrix": cm,
        "y_proba": y_proba
    }

# ---------- UI ----------
st.title("Fraud Detection — Supervised vs Unsupervised")
st.markdown("Upload `creditcard.csv` or use a local fallback. Choose models and settings in the sidebar.")

if missing:
    st.warning("Some optional libraries are missing: " + ", ".join(missing) + 
               ". The app will still run with available models. Install them for full functionality.")

# Sidebar controls
st.sidebar.header("Dataset & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV (creditcard dataset)", type=["csv"])
use_local = st.sidebar.checkbox("Use local file `creditcard.csv` in repo (if present)", value=False)
sample_size = st.sidebar.slider("Sample size for debugging (0 = use full dataset)", min_value=0, max_value=20000, value=10000, step=1000)
use_smote = st.sidebar.checkbox("Apply SMOTE to training data (supervised)", value=True)
n_estimators = st.sidebar.slider("n_estimators (for tree models)", 10, 200, 50, step=10)

models_to_run = st.sidebar.multiselect("Models to run", 
                                       ["RandomForest", "XGBoost", "LightGBM", "IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
                                       default=["RandomForest","XGBoost","IsolationForest"])

run_button = st.sidebar.button("Run experiment")

# ---------- load data ----------
@st.cache_data(show_spinner=False)
def load_csv_from_upload_or_path(uploaded_file, use_local_flag):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if use_local_flag:
        try:
            return pd.read_csv(r"C:\Users\grace\Downloads\creditcard (1).csv")  # place the CSV in repo root
        except Exception:
            return None
    return None

data = load_csv_from_upload_or_path(uploaded, use_local)

if data is None:
    st.info("No dataset loaded. Upload a CSV or check 'Use local file' and ensure `creditcard.csv` exists in your repo.")
    st.stop()

st.write("Dataset shape:", data.shape)
st.dataframe(data.head())

# allow sampling (non-destructive)
if sample_size > 0 and sample_size < len(data):
    st.info(f"Using a random sample of {sample_size} rows for debugging.")
    data_sample = data.sample(sample_size, random_state=42)
else:
    data_sample = data

# features & target
if "Class" not in data_sample.columns:
    st.error("The dataset must contain a 'Class' column (0 = legit, 1 = fraud).")
    st.stop()

X = data_sample.drop("Class", axis=1)
y = data_sample["Class"]

# show original distributions + correlation
st.subheader("Original class distribution")
st.pyplot(fig_countplot(y, title="Class Distribution — Full/Sampled Dataset"))

st.subheader("Correlation heatmap (sample)")
st.pyplot(fig_heatmap(data_sample, title="Correlation Heatmap — Sample"))

# split and run only when user asks
if run_button:
    with st.spinner("Training models — this may take a while depending on sample size and models selected..."):
        # train-test split (stratify)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # apply SMOTE if user chose and library available
        if use_smote and SMOTE is not None and ("RandomForest" in models_to_run or "XGBoost" in models_to_run or "LightGBM" in models_to_run):
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train.copy(), y_train.copy()

        results = []

        # Random Forest
        if "RandomForest" in models_to_run:
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42)
            rf.fit(X_train_res, y_train_res)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:,1]
            results.append(evaluate_model(y_test, y_pred, y_proba, "Random Forest"))

        # XGBoost
        if "XGBoost" in models_to_run and XGBClassifier is not None:
            xgb = XGBClassifier(n_estimators=n_estimators, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train_res, y_train_res)
            y_pred = xgb.predict(X_test)
            y_proba = xgb.predict_proba(X_test)[:,1]
            results.append(evaluate_model(y_test, y_pred, y_proba, "XGBoost"))

        # LightGBM
        if "LightGBM" in models_to_run and LGBMClassifier is not None:
            lgbm = LGBMClassifier(n_estimators=n_estimators, max_depth=6, random_state=42)
            lgbm.fit(X_train_res, y_train_res)
            y_pred = lgbm.predict(X_test)
            y_proba = lgbm.predict_proba(X_test)[:,1]
            results.append(evaluate_model(y_test, y_pred, y_proba, "LightGBM"))

        # Isolation Forest
        if "IsolationForest" in models_to_run:
            if_model = IsolationForest(contamination=0.001, random_state=42)
            if_model.fit(X_train)
            y_pred_if = if_model.predict(X_test)
            y_pred_if = np.where(y_pred_if == -1, 1, 0)
            y_proba_if = -if_model.decision_function(X_test)
            results.append(evaluate_model(y_test, y_pred_if, y_proba_if, "Isolation Forest"))

        # LOF
        if "LocalOutlierFactor" in models_to_run:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True)
            lof.fit(X_train)
            y_pred_lof = lof.predict(X_test)
            y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
            y_proba_lof = -lof.decision_function(X_test)
            results.append(evaluate_model(y_test, y_pred_lof, y_proba_lof, "Local Outlier Factor"))

        # One-Class SVM
        if "OneClassSVM" in models_to_run:
            ocsvm = OneClassSVM(nu=0.001, kernel="rbf", gamma=0.01)
            ocsvm.fit(X_train)
            y_pred_oc = ocsvm.predict(X_test)
            y_pred_oc = np.where(y_pred_oc == -1, 1, 0)
            y_proba_oc = -ocsvm.decision_function(X_test)
            results.append(evaluate_model(y_test, y_pred_oc, y_proba_oc, "One-Class SVM"))

    # show results
    if results:
        results_df = pd.DataFrame(results).drop(columns=["y_proba", "Confusion Matrix"])
        results_df = results_df.set_index("Model")
        st.subheader("Model metrics summary")
        st.dataframe(results_df.style.format("{:.4f}"))

        # Plots: ROC for supervised models (if present)
        st.subheader("ROC Curves (supervised models)")
        sup_results = [r for r in results if r["Model"] in ("Random Forest","XGBoost","LightGBM")]
        for r in sup_results:
            if r["y_proba"] is not None and not np.isnan(r["AUC-ROC"]):
                st.pyplot(fig_roc(y_test, r["y_proba"], r["Model"]))

        st.subheader("Confusion matrices")
        for r in results:
            st.pyplot(fig_confusion(r["Confusion Matrix"], r["Model"]))

        st.success("Done. If app is slow or memory-heavy, reduce sample size and n_estimators, or remove heavy models (XGBoost/LightGBM).")
    else:
        st.info("No models ran. Pick models in the sidebar and press Run.")
