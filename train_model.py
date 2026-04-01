"""
train_model.py
Training Random Forest untuk deteksi love bombing
Output: model_love_bombing.pkl + metrics_report.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = [
    "msg_per_day_week1",
    "msg_per_day_week4",
    "praise_ratio",
    "avg_response_time_min",
    "response_time_std",
    "emotional_intensity_score",
    "commitment_pressure_ratio",
    "isolation_attempt_count",
    "avg_msg_length",
    "msg_length_variance",
    "escalation_speed_days",
    "consistency_score",
    "night_msg_ratio",
    "apology_count",
    "future_planning_ratio",
]

FEATURE_LABELS = {
    "msg_per_day_week1": "Pesan/hari (minggu 1)",
    "msg_per_day_week4": "Pesan/hari (minggu 4)",
    "praise_ratio": "Rasio pujian",
    "avg_response_time_min": "Waktu respons rata2 (menit)",
    "response_time_std": "Variansi waktu respons",
    "emotional_intensity_score": "Intensitas emosi",
    "commitment_pressure_ratio": "Tekanan komitmen",
    "isolation_attempt_count": "Percobaan isolasi",
    "avg_msg_length": "Panjang pesan rata2",
    "msg_length_variance": "Variansi panjang pesan",
    "escalation_speed_days": "Kecepatan eskalasi (hari)",
    "consistency_score": "Konsistensi perilaku",
    "night_msg_ratio": "Pesan tengah malam",
    "apology_count": "Jumlah permintaan maaf",
    "future_planning_ratio": "Rasio perencanaan masa depan",
}


def load_data(path: str) -> tuple:
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Label distribusi:\n{df['label'].value_counts().to_string()}")
    
    X = df[FEATURE_COLS]
    y = df["label"]
    return X, y


def train_and_evaluate(X, y) -> dict:
    print("\nSplit data 80/20 stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Cross validation
    print("Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Evaluasi test set
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n=== Hasil Evaluasi ===")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Normal', 'Love Bombing'])}")
    
    # Feature importance
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Simpan model + scaler
    print("\nMenyimpan model...")
    model_data = {
        "model": rf,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
    }
    with open("model_love_bombing.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("  ✓ model_love_bombing.pkl")
    
    # Simpan metrics
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": feat_imp[["feature", "label", "importance"]].to_dict("records"),
        "roc_curve": {
            "fpr": roc_curve(y_test, y_prob)[0].tolist(),
            "tpr": roc_curve(y_test, y_prob)[1].tolist(),
        }
    }
    with open("metrics_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ metrics_report.json")
    
    # Plot
    generate_plots(cm, feat_imp, y_test, y_prob, metrics)
    
    return metrics


def generate_plots(cm, feat_imp, y_test, y_prob, metrics):
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')
    
    colors = ['#6c63ff', '#ff6584', '#43e97b', '#fa8231']
    
    # 1. Confusion Matrix
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax,
                xticklabels=['Normal', 'Love Bombing'],
                yticklabels=['Normal', 'Love Bombing'],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title('Confusion Matrix', color='white', fontsize=14, pad=12)
    ax.set_xlabel('Predicted', color='#aaa', fontsize=11)
    ax.set_ylabel('Actual', color='#aaa', fontsize=11)
    ax.tick_params(colors='#ccc')
    
    # 2. Feature Importance (top 10)
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    top10 = feat_imp.head(10)
    bars = ax.barh(range(len(top10)), top10['importance'], color=colors[0], alpha=0.8)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['label'], fontsize=9, color='#ccc')
    ax.set_xlabel('Importance', color='#aaa', fontsize=11)
    ax.set_title('Top 10 Feature Importance', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='#ccc', axis='x')
    ax.spines[:].set_visible(False)
    
    # 3. ROC Curve
    ax = axes[2]
    ax.set_facecolor('#1a1a2e')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[0], lw=2, label=f'AUC = {metrics["auc_roc"]:.4f}')
    ax.plot([0, 1], [0, 1], color='#555', lw=1, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.15, color=colors[0])
    ax.set_xlabel('False Positive Rate', color='#aaa', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='#aaa', fontsize=11)
    ax.set_title('ROC Curve', color='white', fontsize=14, pad=12)
    ax.legend(loc='lower right', facecolor='#2a2a3e', labelcolor='white', fontsize=11)
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#333')
    
    plt.tight_layout(pad=2)
    plt.savefig("model_plots.png", dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  ✓ model_plots.png")


if __name__ == "__main__":
    X, y = load_data("dataset_love_bombing.csv")
    metrics = train_and_evaluate(X, y)
    print("\n✓ Training selesai!")
