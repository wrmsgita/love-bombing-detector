"""
app.py
Streamlit web app: Love Bombing Detector
Upload chat CSV atau input manual → prediksi + akurasi model
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import base64
import io
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import roc_curve
from PIL import Image
import anthropic

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Love Bombing Detector 💕",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* ══ BACKGROUND ══ */
.stApp {
    background: linear-gradient(145deg, #fff0f8 0%, #f5f0ff 35%, #f0f5ff 65%, #f0fff8 100%);
    background-attachment: fixed;
}
.main { background: transparent; }

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff5fb 0%, #f8f0ff 100%);
    border-right: 2px solid #ffd6ea;
}
section[data-testid="stSidebar"] * { color: #5a3d6e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #8B5CF6 !important; }

/* ══ GLOBAL TEXT ══ */
h1, h2, h3, h4 { color: #4a2c6e !important; }
p, li, span, div { color: #5a3d6e; }
label { color: #7c5a9e !important; }

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.7);
    border-radius: 20px;
    padding: 6px;
    border: 2px solid #ffd6ea;
    box-shadow: 0 4px 20px rgba(255,133,161,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    color: #a07cc5;
    font-weight: 700;
    font-size: 0.92rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff85a1, #a78bff) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(167,139,255,0.35);
}

/* ══ CARDS ══ */
.metric-card {
    background: white;
    border: 2px solid #ffd6ea;
    border-radius: 24px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(255,133,161,0.1);
}
.metric-card:hover {
    border-color: #a78bff;
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(167,139,255,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff85a1, #a78bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.82rem;
    color: #a07cc5;
    margin-top: 4px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ══ RISK BADGES ══ */
.risk-low     { background: #dcfce7; border: 2px solid #4ade80; color: #16a34a; }
.risk-medium  { background: #fef9c3; border: 2px solid #fbbf24; color: #b45309; }
.risk-high    { background: #fee2e2; border: 2px solid #f87171; color: #dc2626; }
.risk-critical{ background: #fce7f3; border: 2px solid #f472b6; color: #be185d; }
.risk-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 50px;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 1.5px;
}

/* ══ RESULT CARD ══ */
.result-card {
    background: white;
    border-radius: 28px;
    padding: 32px;
    border: 2px solid #ffd6ea;
    margin-top: 16px;
    box-shadow: 0 8px 40px rgba(255,133,161,0.12);
}

/* ══ SCORE BAR ══ */
.score-bar-container {
    background: #f3e8ff;
    border-radius: 50px;
    height: 14px;
    overflow: hidden;
    margin-top: 8px;
}

/* ══ BUTTONS ══ */
.stButton > button {
    background: linear-gradient(135deg, #ff85a1, #a78bff);
    color: white !important;
    border: none;
    border-radius: 50px;
    padding: 10px 28px;
    font-weight: 800;
    font-size: 1rem;
    width: 100%;
    transition: all 0.25s;
    box-shadow: 0 4px 16px rgba(167,139,255,0.3);
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(167,139,255,0.4);
    opacity: 0.92;
}

/* ══ DIVIDER ══ */
.section-divider {
    border: none;
    border-top: 2px dashed #ffd6ea;
    margin: 28px 0;
}

/* ══ INFO / WARNING BOXES ══ */
.warning-box {
    background: #fff0f4;
    border: 2px solid #ffb3c6;
    border-radius: 18px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c0134d;
}
.info-box {
    background: #f5f0ff;
    border: 2px solid #c4b5fd;
    border-radius: 18px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #6d28d9;
}

/* ══ DATAFRAME ══ */
.stDataFrame { border-radius: 16px; overflow: hidden; }

/* ══ SLIDERS ══ */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #ff85a1, #a78bff) !important;
}

/* ══ EXPANDER ══ */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 16px !important;
    border: 2px solid #ffd6ea !important;
    color: #7c5a9e !important;
    font-weight: 700 !important;
}

/* ════════════════════════════════
   KAWAII ANIMATIONS
   ════════════════════════════════ */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-10px); }
}
@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  14%  { transform: scale(1.2); }
  28%  { transform: scale(1); }
  42%  { transform: scale(1.12); }
  70%  { transform: scale(1); }
}
@keyframes gradientFlow {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes shimmer {
  0%   { left: -100%; }
  100% { left: 200%; }
}
@keyframes bounce-in {
  0%   { transform: scale(0.7); opacity: 0; }
  70%  { transform: scale(1.05); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes wiggle {
  0%, 100% { transform: rotate(-3deg); }
  50%       { transform: rotate(3deg); }
}

/* ══ KAWAII HEADER ══ */
.kawaii-header { text-align: center; padding: 24px 0 12px; }
.kawaii-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff85a1, #a78bff, #60c6ff, #ff85a1);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 4s ease infinite;
    margin-bottom: 6px;
}
.kawaii-subtitle { color: #a07cc5; font-size: 1rem; font-weight: 600; }
.floating-mascot {
    font-size: 3.8rem;
    display: block;
    animation: float 3s ease-in-out infinite;
    margin-bottom: 8px;
    filter: drop-shadow(0 6px 12px rgba(167,139,255,0.3));
}

/* ══ UPLOAD ZONE ══ */
.upload-zone {
    background: linear-gradient(135deg, #fff5fb, #f5f0ff, #f0f8ff);
    border: 3px dashed #d8b4fe;
    border-radius: 28px;
    padding: 44px 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.upload-zone::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    animation: shimmer 3s infinite;
}
.upload-zone:hover {
    border-color: #a78bff;
    background: linear-gradient(135deg, #ffe4f3, #ede9fe, #e0f2fe);
    box-shadow: 0 8px 40px rgba(167,139,255,0.2);
}
.upload-icon { font-size: 3.5rem; margin-bottom: 12px; animation: float 2.5s ease-in-out infinite; display: block; }
.upload-text { color: #7c5a9e; font-size: 1.1rem; font-weight: 800; }
.upload-hint { color: #a07cc5; font-size: 0.85rem; margin-top: 8px; font-weight: 600; }

/* ══ KAWAII CARDS ══ */
.kawaii-card {
    background: white;
    border: 2px solid #e9d5ff;
    border-radius: 20px;
    padding: 18px;
    margin: 8px 0;
    transition: all 0.3s;
    box-shadow: 0 4px 16px rgba(167,139,255,0.08);
    color: #5a3d6e;
}
.kawaii-card:hover {
    border-color: #a78bff;
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(167,139,255,0.18);
}

/* ══ RESULT HEADER ══ */
.kawaii-result-header {
    background: linear-gradient(135deg, #fff5fb, #f5f0ff, #f0f8ff);
    border: 2px solid #e9d5ff;
    border-radius: 28px;
    padding: 36px 24px;
    text-align: center;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(167,139,255,0.12);
    animation: bounce-in 0.5s ease;
}
.kawaii-result-header::after {
    content: '✨';
    position: absolute;
    top: 12px; right: 16px;
    font-size: 1.4rem;
    animation: spin-slow 4s linear infinite;
}
.kawaii-score {
    font-size: 4.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff85a1, #a78bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.kawaii-score-label {
    color: #a07cc5;
    font-size: 0.85rem;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ══ KAWAII BADGES ══ */
.kawaii-badge {
    display: inline-block;
    padding: 8px 28px;
    border-radius: 50px;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 2px;
    margin: 12px 0;
}
.kawaii-low      { background: #dcfce7; border: 2px solid #4ade80; color: #15803d; }
.kawaii-medium   { background: #fef9c3; border: 2px solid #facc15; color: #a16207; }
.kawaii-high     { background: #fee2e2; border: 2px solid #f87171; color: #b91c1c; }
.kawaii-critical { background: #fce7f3; border: 2px solid #f472b6; color: #be185d; }

/* ══ MSG BUBBLES ══ */
.msg-bubble {
    background: #f5f0ff;
    border-left: 4px solid #c4b5fd;
    border-radius: 0 16px 16px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: #5a3d6e;
}
.msg-bubble.sender {
    border-left-color: #ff85a1;
    background: #fff0f8;
    color: #7c1d4a;
}

/* ══ FEATURE CHIPS ══ */
.feature-chip {
    display: inline-block;
    background: #f5f0ff;
    border: 2px solid #e9d5ff;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #7c5a9e;
    margin: 3px;
    font-weight: 700;
}
.feature-chip.warning {
    background: #fff0f4;
    border-color: #ffb3c6;
    color: #be185d;
}

/* ══ KAWAII TIPS ══ */
.kawaii-tip {
    background: linear-gradient(135deg, #fff5fb, #f5f0ff);
    border: 2px solid #e9d5ff;
    border-radius: 18px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #7c5a9e;
    font-weight: 600;
}

/* ══ ANALYZE BUTTON ══ */
.analyze-btn > button {
    background: linear-gradient(135deg, #ff85a1, #a78bff, #60c6ff) !important;
    background-size: 200% 200% !important;
    animation: gradientFlow 3s ease infinite !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.15rem !important;
    border-radius: 50px !important;
    padding: 14px 0 !important;
    border: none !important;
    letter-spacing: 1px;
    box-shadow: 0 6px 24px rgba(167,139,255,0.4) !important;
}

/* ══ PLATFORM RADIO PILLS ══ */
div[data-testid="stRadio"] > label {
    display: none;
}
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    background: transparent !important;
}
div[data-testid="stRadio"] > div > label {
    background: white !important;
    border: 2px solid #e9d5ff !important;
    border-radius: 50px !important;
    padding: 8px 20px !important;
    cursor: pointer !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    color: #7c5a9e !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(167,139,255,0.1) !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] > div > label:hover {
    border-color: #a78bff !important;
    background: #f5f0ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(167,139,255,0.2) !important;
}
div[data-testid="stRadio"] > div > label[data-baseweb="radio"]:has(input:checked),
div[data-testid="stRadio"] > div > label[aria-checked="true"] {
    background: linear-gradient(135deg, #ff85a1, #a78bff) !important;
    border-color: transparent !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(167,139,255,0.4) !important;
}
div[data-testid="stRadio"] > div > label > div:first-child {
    display: none !important;
}

/* Hide the default radio circle dot */
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* ══ CLEAN SECTION LABEL ══ */
.section-label {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c4b5fd;
    margin-bottom: 10px;
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model_love_bombing.pkl"):
        return None
    with open("model_love_bombing.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    if not os.path.exists("metrics_report.json"):
        return None
    with open("metrics_report.json", "r") as f:
        return json.load(f)

model_data = load_model()
metrics = load_metrics()


# ─── Helper functions ─────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> tuple[str, str]:
    if prob < 0.3:   return "LOW", "risk-low"
    if prob < 0.55:  return "MEDIUM", "risk-medium"
    if prob < 0.80:  return "HIGH", "risk-high"
    return "CRITICAL", "risk-critical"

def get_risk_color(level: str) -> str:
    return {"LOW": "#43e97b", "MEDIUM": "#fa8231", "HIGH": "#ff6565", "CRITICAL": "#ff3232"}.get(level, "#fff")

def predict_single(features: dict) -> tuple[float, str, str]:
    if model_data is None:
        return 0.0, "UNKNOWN", "risk-low"
    X = pd.DataFrame([features])[model_data["feature_cols"]]
    X_scaled = model_data["scaler"].transform(X)
    prob = model_data["model"].predict_proba(X_scaled)[0][1]
    level, css = get_risk_level(prob)
    return prob, level, css

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if model_data is None:
        return df
    feature_cols = model_data["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    
    if missing:
        st.warning(f"Kolom tidak ditemukan, diisi default 0: {missing}")
        for col in missing:
            df[col] = 0
    
    X = df[feature_cols]
    X_scaled = model_data["scaler"].transform(X)
    probs = model_data["model"].predict_proba(X_scaled)[:, 1]
    preds = model_data["model"].predict(X_scaled)
    
    df = df.copy()
    df["love_bombing_probability"] = probs
    df["prediction"] = preds
    df["risk_level"] = [get_risk_level(p)[0] for p in probs]
    return df


def plot_confusion_matrix(cm_data: list) -> plt.Figure:
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#fff5fb')
    ax.set_facecolor('#fff5fb')
    cmap = sns.color_palette(["#fce7f3", "#f9a8d4", "#ec4899", "#be185d"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax,
                xticklabels=['Normal', 'Love Bombing'],
                yticklabels=['Normal', 'Love Bombing'],
                cbar=False, annot_kws={"size": 14, "weight": "bold", "color": "#4a2c6e"})
    ax.set_title('Confusion Matrix 🌸', color='#7c5a9e', fontsize=13, pad=12, fontweight='bold')
    ax.set_xlabel('Predicted', color='#a07cc5', fontsize=10)
    ax.set_ylabel('Actual', color='#a07cc5', fontsize=10)
    ax.tick_params(colors='#7c5a9e')
    for spine in ax.spines.values():
        spine.set_color('#e9d5ff')
    plt.tight_layout()
    return fig


def plot_feature_importance(feat_data: list) -> plt.Figure:
    top10 = feat_data[:10]
    labels = [d["label"] for d in top10]
    values = [d["importance"] for d in top10]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#f5f0ff')
    ax.set_facecolor('#f5f0ff')

    palette = ["#ff85a1", "#f472b6", "#e879f9", "#c084fc",
               "#a78bff", "#818cf8", "#60a5fa", "#38bdf8",
               "#34d399", "#4ade80"]
    bars = ax.barh(range(len(top10)), values, color=palette[:len(top10)],
                   alpha=0.88, height=0.62)
    for bar in bars:
        bar.set_linewidth(0)

    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=9, color='#5a3d6e', fontweight='bold')
    ax.set_xlabel('Importance ✨', color='#a07cc5', fontsize=10)
    ax.set_title('Top 10 Feature Importance 💡', color='#7c5a9e', fontsize=13, pad=12, fontweight='bold')
    ax.tick_params(colors='#a07cc5', axis='x')
    ax.spines[:].set_visible(False)
    ax.set_facecolor('#f5f0ff')
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr: list, tpr: list, auc: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#fff5fb')
    ax.set_facecolor('#fff5fb')
    ax.plot(fpr, tpr, color='#a78bff', lw=2.5, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], color='#e9d5ff', lw=1.5, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#a78bff')
    ax.set_xlabel('False Positive Rate', color='#a07cc5', fontsize=10)
    ax.set_ylabel('True Positive Rate', color='#a07cc5', fontsize=10)
    ax.set_title('ROC Curve 📈', color='#7c5a9e', fontsize=13, pad=12, fontweight='bold')
    ax.legend(loc='lower right', facecolor='white', edgecolor='#e9d5ff',
              labelcolor='#7c5a9e', fontsize=10)
    ax.tick_params(colors='#a07cc5')
    for spine in ax.spines.values():
        spine.set_color('#e9d5ff')
    plt.tight_layout()
    return fig


# ─── Claude Vision: Analyze Chat Image ──────────────────────────────────────
PLATFORM_PROMPTS = {
    "WhatsApp":     "tampilan chat WhatsApp (bubble hijau/putih, foto profil bulat, nama di atas bubble)",
    "Instagram DM": "tampilan Instagram Direct Message (bubble ungu/hitam, DM IG)",
    "Telegram":     "tampilan Telegram (bubble biru/putih, ikon pesawat)",
    "Line":         "tampilan LINE (bubble hijau, sticker LINE)",
    "Twitter/X DM": "tampilan Twitter atau X Direct Message",
    "SMS/iMessage": "tampilan SMS atau iMessage Apple (bubble biru/hijau)",
    "Auto-detect":  "platform chat apapun (WhatsApp, Instagram, Telegram, dll)",
}

KAWAII_MASCOTS = {
    "LOW":      "🐱",
    "MEDIUM":   "🐰",
    "HIGH":     "🦊",
    "CRITICAL": "🚨",
}

def analyze_chat_image(image_bytes: bytes, platform: str, media_type: str = "image/jpeg") -> dict:
    """Kirim screenshot ke Claude Vision → ekstrak fitur love bombing."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    platform_desc = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["Auto-detect"])

    system_prompt = """Kamu adalah ahli psikologi digital dan deteksi love bombing.
Tugasmu: analisis screenshot chat dan ekstrak fitur-fitur kuantitatif untuk model ML deteksi love bombing.
SELALU jawab dalam format JSON yang valid, tidak ada teks lain di luar JSON."""

    user_prompt = f"""Ini adalah screenshot {platform_desc}.

Analisis percakapan ini dan kembalikan JSON dengan format PERSIS seperti ini:

{{
  "platform_detected": "nama platform yang terdeteksi",
  "sender_name": "nama/username pengirim utama (yang mungkin melakukan love bombing, bukan si penerima)",
  "msg_count_visible": <jumlah pesan yang terlihat>,
  "extracted_messages": [
    {{"sender": "nama", "text": "isi pesan", "time": "waktu jika ada"}}
  ],
  "features": {{
    "msg_per_day_week1": <estimasi pesan per hari 0-100, tinggi jika banyak pesan dalam waktu singkat>,
    "msg_per_day_week4": <estimasi pesan per hari minggu ke-4, biasanya lebih rendah jika love bombing>,
    "praise_ratio": <0.0-1.0, proporsi pesan berisi pujian berlebihan/love bombing words>,
    "avg_response_time_min": <estimasi waktu respons menit, rendah jika selalu cepat balas>,
    "response_time_std": <variansi waktu respons>,
    "emotional_intensity_score": <0-10, intensitas emosi rata-rata>,
    "commitment_pressure_ratio": <0.0-1.0, proporsi pesan yang memaksa komitmen>,
    "isolation_attempt_count": <0-40, jumlah upaya isolasi dari orang lain>,
    "avg_msg_length": <panjang karakter rata-rata per pesan>,
    "msg_length_variance": <variansi panjang pesan>,
    "escalation_speed_days": <estimasi hari sampai sangat intens, makin kecil makin merah>,
    "consistency_score": <0-10, 10=konsisten, rendah jika mood swing>,
    "night_msg_ratio": <0.0-1.0, proporsi pesan malam hari>,
    "apology_count": <jumlah permintaan maaf>,
    "future_planning_ratio": <0.0-1.0, proporsi membahas rencana masa depan bersama>
  }},
  "red_flags": ["daftar tanda bahaya yang terlihat dalam pesan"],
  "analysis_summary": "Ringkasan analisis dalam Bahasa Indonesia, 2-3 kalimat, jelaskan pola yang terdeteksi",
  "confidence": <0.0-1.0, seberapa yakin kamu dengan analisis ini berdasarkan kejelasan gambar>
}}

Jika gambar buram/tidak jelas/bukan screenshot chat, tetap kembalikan JSON tapi dengan confidence rendah dan features semua 0."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2500,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    # Bersihkan jika ada markdown code block
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px;">
        <div style="font-size:3rem; animation: float 3s ease-in-out infinite; display:inline-block;">🌸</div>
        <div style="font-size:1.2rem; font-weight:900; background:linear-gradient(135deg,#ff85a1,#a78bff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top:6px;">
            Love Bombing Detector
        </div>
        <div style="font-size:0.8rem; color:#a07cc5; margin-top:4px; font-weight:600;">
            Jaga hatimu dari manipulasi 💕
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:2px dashed #ffd6ea; margin:12px 0;">', unsafe_allow_html=True)

    if model_data:
        st.markdown("""
        <div style="background:#dcfce7; border:2px solid #4ade80; border-radius:16px;
                    padding:10px 14px; text-align:center; color:#15803d; font-weight:800;">
            ✅ Model siap digunakan!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fee2e2; border:2px solid #f87171; border-radius:16px;
                    padding:10px 14px; text-align:center; color:#b91c1c; font-weight:800;">
            ❌ Model belum ada<br>
            <small style="font-weight:600;">Jalankan train_model.py dulu~</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border:1.5px solid #f0e4ff; margin:14px 0;">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem; font-weight:800; letter-spacing:2.5px;
                text-transform:uppercase; color:#c4b5fd; margin-bottom:12px;">
        15 Fitur Dianalisis
    </div>
    """, unsafe_allow_html=True)

    features_sidebar = [
        ("Frekuensi pesan",        "#ff85a1"),
        ("Intensitas emosi",        "#f472b6"),
        ("Rasio pujian",            "#c084fc"),
        ("Tekanan komitmen",        "#a78bff"),
        ("Waktu respons",           "#818cf8"),
        ("Upaya isolasi",           "#60a5fa"),
        ("Konsistensi perilaku",    "#34d399"),
        ("Pesan tengah malam",      "#facc15"),
        ("Kecepatan eskalasi",      "#fb923c"),
    ]
    for label, color in features_sidebar:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px;
                    padding:5px 0; font-size:0.85rem;">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:{color}; flex-shrink:0;"></div>
            <span style="color:#5a3d6e; font-weight:600;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border:1.5px solid #f0e4ff; margin:14px 0;">', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#faf5ff; border:1.5px solid #e9d5ff; border-radius:14px;
                padding:12px 14px; font-size:0.82rem; color:#a07cc5; font-weight:600;
                text-align:center; line-height:1.6;">
        ⚠️ Untuk tujuan edukasi &amp; penelitian
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 24px;">
    <div style="font-size:0.9rem; letter-spacing:10px; color:#d8b4fe; margin-bottom:10px; font-weight:700;">
        ✿ ♡ ✿ ♡ ✿
    </div>
    <h1 style="font-size:3rem; font-weight:900;
               background:linear-gradient(135deg,#ff85a1,#a78bff,#60c6ff,#ff85a1);
               background-size:300% 300%;
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               animation: gradientFlow 4s ease infinite; margin-bottom:10px; line-height:1.2;">
        🌸 Love Bombing Detector 🌸
    </h1>
    <div style="display:inline-block; background:linear-gradient(135deg,#fff5fb,#f5f0ff);
                border:2px solid #e9d5ff; border-radius:50px; padding:8px 24px;
                color:#7c5a9e; font-size:0.95rem; font-weight:700;">
        ✨ Deteksi pola manipulasi hubungan pakai Machine Learning ✨
    </div>
    <div style="font-size:0.9rem; letter-spacing:10px; color:#d8b4fe; margin-top:10px; font-weight:700;">
        ✿ ♡ ✿ ♡ ✿
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analisis Manual", "📁 Upload CSV", "📸 Scan Chat", "📊 Akurasi Model", "📖 Panduan"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Analisis Manual
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 16px;">
        <span style="font-size:1.6rem; font-weight:900; color:#7c5a9e;">🔍 Analisis Manual Percakapan</span><br>
        <span style="font-size:0.9rem; color:#a07cc5; font-weight:600;">Isi slider berdasarkan pola chat yang kamu amati 💬</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Isi fitur berdasarkan analisis percakapan selama <b>30 hari pertama</b> hubungan.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📱 Pola Pesan**")
        msg_w1 = st.slider("Pesan/hari minggu pertama", 0.0, 100.0, 15.0, 0.5)
        msg_w4 = st.slider("Pesan/hari minggu keempat", 0.0, 100.0, 10.0, 0.5)
        avg_len = st.slider("Panjang pesan rata-rata (karakter)", 10.0, 600.0, 80.0, 5.0)
        len_var = st.slider("Variansi panjang pesan", 0.0, 300.0, 50.0, 5.0)
        night_ratio = st.slider("Rasio pesan tengah malam (22:00-05:00)", 0.0, 1.0, 0.1, 0.01)
    
    with col2:
        st.markdown("**💬 Konten & Emosi**")
        praise = st.slider("Rasio pujian dalam pesan", 0.0, 1.0, 0.1, 0.01)
        emotion = st.slider("Skor intensitas emosi (0-10)", 0.0, 10.0, 5.0, 0.1)
        commit = st.slider("Tekanan komitmen (rasio)", 0.0, 1.0, 0.05, 0.01)
        future = st.slider("Rasio rencana masa depan", 0.0, 1.0, 0.1, 0.01)
        apology = st.slider("Jumlah permintaan maaf (30 hari)", 0, 80, 3, 1)
    
    with col3:
        st.markdown("**⏱️ Pola Waktu & Perilaku**")
        resp_time = st.slider("Waktu respons rata-rata (menit)", 0.1, 120.0, 20.0, 0.5)
        resp_std = st.slider("Variansi waktu respons", 0.0, 60.0, 15.0, 0.5)
        escalation = st.slider("Kecepatan eskalasi (hari)", 1.0, 180.0, 45.0, 1.0)
        consistency = st.slider("Konsistensi perilaku (0=inkonsisten, 10=konsisten)", 0.0, 10.0, 7.0, 0.1)
        isolation = st.slider("Percobaan isolasi dari orang lain", 0, 40, 0, 1)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    if st.button("🔍 Analisis Sekarang", use_container_width=True):
        if model_data is None:
            st.error("Model belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
        else:
            features = {
                "msg_per_day_week1": msg_w1,
                "msg_per_day_week4": msg_w4,
                "praise_ratio": praise,
                "avg_response_time_min": resp_time,
                "response_time_std": resp_std,
                "emotional_intensity_score": emotion,
                "commitment_pressure_ratio": commit,
                "isolation_attempt_count": isolation,
                "avg_msg_length": avg_len,
                "msg_length_variance": len_var,
                "escalation_speed_days": escalation,
                "consistency_score": consistency,
                "night_msg_ratio": night_ratio,
                "apology_count": apology,
                "future_planning_ratio": future,
            }
            
            prob, level, css = predict_single(features)
            color = get_risk_color(level)
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(f"""
                <div style="text-align:center; padding:12px 0;">
                    <div style="font-size:0.85rem; color:#a07cc5; font-weight:700;
                                letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">
                        🎯 Risk Level
                    </div>
                    <span class="risk-badge {css}">{level}</span>
                    <div style="font-size:3.5rem; font-weight:900;
                                background:linear-gradient(135deg,#ff85a1,#a78bff);
                                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                                margin:16px 0 4px; line-height:1;">
                        {prob*100:.1f}%
                    </div>
                    <div style="font-size:0.9rem; color:#a07cc5; font-weight:700;">
                        Probabilitas Love Bombing
                    </div>
                    <div class="score-bar-container" style="margin-top:16px; max-width:280px; margin-left:auto; margin-right:auto;">
                        <div style="height:100%; width:{prob*100:.1f}%;
                                    background:linear-gradient(90deg,#ff85a1,#a78bff);
                                    border-radius:50px; transition:width 0.6s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Interpretasi
            st.markdown("**📋 Interpretasi:**")
            
            interpretations = []
            if msg_w1 > 30:
                interpretations.append(f"⚠️ Frekuensi pesan minggu pertama sangat tinggi ({msg_w1:.0f}x/hari)")
            if praise > 0.4:
                interpretations.append(f"⚠️ Rasio pujian abnormal ({praise*100:.0f}% pesan)")
            if resp_time < 3:
                interpretations.append(f"⚠️ Waktu respons terlalu cepat (rata-rata {resp_time:.1f} menit)")
            if escalation < 10:
                interpretations.append(f"⚠️ Eskalasi emosi sangat cepat ({escalation:.0f} hari)")
            if consistency < 4:
                interpretations.append(f"⚠️ Perilaku sangat inkonsisten (skor {consistency:.1f}/10)")
            if isolation > 5:
                interpretations.append(f"⚠️ Upaya isolasi terdeteksi ({isolation}x dalam 30 hari)")
            if commit > 0.3:
                interpretations.append(f"⚠️ Tekanan komitmen tinggi ({commit*100:.0f}% pesan)")
            
            if not interpretations:
                st.success("✓ Tidak ada pola mencurigakan terdeteksi pada percakapan ini.")
            else:
                for i in interpretations:
                    st.warning(i)
            
            if level in ["HIGH", "CRITICAL"]:
                st.markdown('<div class="warning-box">🚨 <b>Perhatian:</b> Pola percakapan ini menunjukkan indikator love bombing yang signifikan. Disarankan untuk waspada dan berkonsultasi dengan orang tepercaya.</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Upload CSV
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 16px;">
        <span style="font-size:1.6rem; font-weight:900; color:#7c5a9e;">📁 Upload Dataset CSV</span><br>
        <span style="font-size:0.9rem; color:#a07cc5; font-weight:600;">Analisis banyak data sekaligus~ 🗂️</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Format CSV yang dibutuhkan"):
        sample_df = pd.DataFrame([{
            "msg_per_day_week1": 45.2,
            "msg_per_day_week4": 5.1,
            "praise_ratio": 0.72,
            "avg_response_time_min": 1.2,
            "response_time_std": 0.5,
            "emotional_intensity_score": 8.9,
            "commitment_pressure_ratio": 0.45,
            "isolation_attempt_count": 12,
            "avg_msg_length": 280.0,
            "msg_length_variance": 120.0,
            "escalation_speed_days": 3.0,
            "consistency_score": 2.5,
            "night_msg_ratio": 0.55,
            "apology_count": 28,
            "future_planning_ratio": 0.68,
        }])
        st.dataframe(sample_df)
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button("⬇️ Download template CSV", csv_sample, "template_love_bombing.csv", "text/csv")
    
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.info(f"✓ File diupload: {len(df_raw):,} baris, {len(df_raw.columns)} kolom")
        
        if st.button("🔍 Analisis Semua Data"):
            if model_data is None:
                st.error("Model belum tersedia.")
            else:
                with st.spinner("Memproses..."):
                    df_result = predict_batch(df_raw)
                
                st.markdown("### Hasil Prediksi")
                
                c1, c2, c3, c4 = st.columns(4)
                total = len(df_result)
                lb_count = (df_result["prediction"] == 1).sum()
                lb_pct = lb_count / total * 100
                avg_prob = df_result["love_bombing_probability"].mean()
                
                for col, val, label in [
                    (c1, f"{total:,}", "Total Data"),
                    (c2, f"{lb_count:,}", "Terdeteksi LB"),
                    (c3, f"{lb_pct:.1f}%", "Persentase LB"),
                    (c4, f"{avg_prob*100:.1f}%", "Avg Probability"),
                ]:
                    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Risk distribution
                risk_counts = df_result["risk_level"].value_counts()
                st.markdown("**Distribusi Risk Level:**")
                level_colors = {"LOW":"#4ade80","MEDIUM":"#facc15","HIGH":"#f87171","CRITICAL":"#f472b6"}
                level_bg     = {"LOW":"#dcfce7","MEDIUM":"#fef9c3","HIGH":"#fee2e2","CRITICAL":"#fce7f3"}
                for lvl in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    count = risk_counts.get(lvl, 0)
                    pct = count / total * 100
                    clr = level_colors[lvl]
                    bg  = level_bg[lvl]
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
                        <span style="width:90px;background:{bg};border:2px solid {clr};border-radius:50px;
                                     padding:3px 10px;font-weight:800;font-size:0.82rem;color:{clr};text-align:center;">
                            {lvl}
                        </span>
                        <div style="flex:1;background:#f3e8ff;border-radius:50px;height:12px;">
                            <div style="width:{pct:.1f}%;background:linear-gradient(90deg,{clr}88,{clr});
                                        height:100%;border-radius:50px;"></div>
                        </div>
                        <span style="color:#7c5a9e;font-size:0.9rem;font-weight:700;width:80px;">
                            {count:,} ({pct:.0f}%)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**Data Hasil Prediksi:**")
                display_cols = ["love_bombing_probability", "prediction", "risk_level"] + \
                               [c for c in df_result.columns if c not in ["love_bombing_probability","prediction","risk_level","label"]]
                st.dataframe(df_result[display_cols].head(200), use_container_width=True)
                
                csv_out = df_result.to_csv(index=False)
                st.download_button("⬇️ Download Hasil Lengkap", csv_out, "hasil_prediksi.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Scan Chat Image (KAWAII ✨)
# ══════════════════════════════════════════════════════════════════════
with tab3:
    # ── Header ──
    st.markdown("""
    <div style="text-align:center; padding:20px 0 8px;">
        <div style="font-size:3rem; animation:float 3s ease-in-out infinite;
                    display:inline-block; margin-bottom:10px;">📸</div>
        <div style="font-size:1.8rem; font-weight:900;
                    background:linear-gradient(135deg,#ff85a1,#a78bff,#60c6ff);
                    background-size:200% 200%; -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    animation:gradientFlow 4s ease infinite; margin-bottom:6px;">
            Scan Screenshot Chat
        </div>
        <div style="font-size:0.9rem; color:#a07cc5; font-weight:600;">
            Upload foto chat untuk deteksi pola love bombing otomatis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Platform selector ──
    st.markdown("""
    <div style="text-align:center; margin:16px 0 4px;">
        <span style="font-size:0.75rem; font-weight:800; letter-spacing:2.5px;
                     text-transform:uppercase; color:#c4b5fd;">
            Platform Chat
        </span>
    </div>
    """, unsafe_allow_html=True)

    PLATFORM_OPTIONS = [
        "WhatsApp", "Instagram DM", "Telegram",
        "Line", "Twitter / X", "iMessage", "Auto-detect",
    ]
    # Map display name → key used in PLATFORM_PROMPTS
    PLATFORM_MAP = {
        "WhatsApp":     "WhatsApp",
        "Instagram DM": "Instagram DM",
        "Telegram":     "Telegram",
        "Line":         "Line",
        "Twitter / X":  "Twitter/X DM",
        "iMessage":     "SMS/iMessage",
        "Auto-detect":  "Auto-detect",
    }

    _radio_col = st.columns([1, 6, 1])[1]
    with _radio_col:
        selected_label = st.radio(
            "platform",
            options=PLATFORM_OPTIONS,
            index=PLATFORM_OPTIONS.index(
                st.session_state.get("scan_platform_label", "Auto-detect")
            ),
            horizontal=True,
            label_visibility="collapsed",
        )
    st.session_state["scan_platform_label"] = selected_label
    selected_platform = PLATFORM_MAP[selected_label]

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Upload zone ──
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📸</div>
        <div class="upload-text">Drop screenshot chat di sini~</div>
        <div class="upload-hint">✨ Mendukung WhatsApp · Instagram · Telegram · Line · dan lainnya ✨</div>
        <div class="upload-hint" style="margin-top:8px;">Format: JPG · PNG · WEBP · max 10MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Pilih gambar screenshot chat",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_img:
        # Preview image
        img_bytes = uploaded_img.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Determine media type
        ext = uploaded_img.name.rsplit(".", 1)[-1].lower()
        media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/jpeg")

        col_prev, col_info = st.columns([1, 1])
        with col_prev:
            st.markdown("""
            <div style="font-size:0.75rem; font-weight:800; letter-spacing:2px;
                        text-transform:uppercase; color:#c4b5fd; margin-bottom:8px;">
                Preview
            </div>
            """, unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        with col_info:
            w, h = pil_img.size
            size_kb = len(img_bytes) / 1024
            st.markdown(f"""
            <div class="kawaii-card" style="margin-top:0;">
                <div style="font-size:0.75rem; font-weight:800; letter-spacing:2px;
                            text-transform:uppercase; color:#c4b5fd; margin-bottom:14px;">
                    Info File
                </div>
                <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                    <tr>
                        <td style="color:#a07cc5; font-weight:700; padding:5px 0; width:40%;">Nama</td>
                        <td style="color:#4a2c6e; font-weight:800;">{uploaded_img.name}</td>
                    </tr>
                    <tr>
                        <td style="color:#a07cc5; font-weight:700; padding:5px 0;">Resolusi</td>
                        <td style="color:#4a2c6e; font-weight:800;">{w} × {h} px</td>
                    </tr>
                    <tr>
                        <td style="color:#a07cc5; font-weight:700; padding:5px 0;">Ukuran</td>
                        <td style="color:#4a2c6e; font-weight:800;">{size_kb:.1f} KB</td>
                    </tr>
                    <tr>
                        <td style="color:#a07cc5; font-weight:700; padding:5px 0;">Platform</td>
                        <td>
                            <span style="background:linear-gradient(135deg,#ff85a1,#a78bff);
                                         color:white; padding:2px 12px; border-radius:50px;
                                         font-size:0.82rem; font-weight:800;">
                                {selected_platform}
                            </span>
                        </td>
                    </tr>
                </table>
            </div>
            <div class="kawaii-tip" style="margin-top:10px;">
                💡 Screenshot jelas &amp; tidak terpotong = hasil lebih akurat
            </div>
            <div class="kawaii-tip">
                🔒 Gambar hanya diproses sementara, tidak disimpan
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        btn_col = st.columns([1, 2, 1])[1]
        with btn_col:
            st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
            do_analyze = st.button("🔮 Analisa Sekarang! 💖", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if do_analyze:
            with st.spinner("🌸 Lagi baca chat-nya... sebentar ya~ ✨"):
                try:
                    result = analyze_chat_image(img_bytes, selected_platform, media_type)
                except json.JSONDecodeError as e:
                    st.error(f"😿 Hmm, Claude-nya galau... coba lagi ya! (JSON error: {e})")
                    st.stop()
                except Exception as e:
                    st.error(f"😿 Ada error nih: {e}")
                    st.stop()

            # ── Run ML model on extracted features ──
            features_raw = result.get("features", {})
            model_prob, model_level, model_css = None, None, None
            if model_data and features_raw:
                try:
                    prob_ml, lvl_ml, css_ml = predict_single(features_raw)
                    model_prob = prob_ml
                    model_level = lvl_ml
                    model_css = css_ml
                except Exception:
                    pass

            # Use model result if available, else estimate from confidence
            final_level = model_level or "MEDIUM"
            final_prob  = model_prob if model_prob is not None else result.get("confidence", 0.5)
            kawaii_level_css = {
                "LOW": "kawaii-low", "MEDIUM": "kawaii-medium",
                "HIGH": "kawaii-high", "CRITICAL": "kawaii-critical",
            }.get(final_level, "kawaii-medium")
            mascot = KAWAII_MASCOTS.get(final_level, "🐱")
            color  = get_risk_color(final_level)

            level_messages = {
                "LOW":      ("Aman nih kayaknya~ 🍀", "Pola percakapan terlihat normal dan sehat! Tetap jaga diri ya 💚"),
                "MEDIUM":   ("Hm, agak curiga nih... 🤔", "Ada beberapa pola yang perlu diperhatikan. Tetap waspada ya 💛"),
                "HIGH":     ("Waduh, bahaya nih! 😰", "Terdeteksi pola love bombing yang signifikan. Please be careful 🧡"),
                "CRITICAL": ("BAHAYA BANGET!! 🚨", "Pola love bombing sangat kuat terdeteksi! Segera minta bantuan terpercaya 🔴"),
            }
            level_title, level_desc = level_messages.get(final_level, ("", ""))

            st.markdown("---")

            # ── Main result card ──
            st.markdown(f"""
            <div class="kawaii-result-header">
                <div style="font-size:4rem; animation: float 3s ease-in-out infinite; display:inline-block;">{mascot}</div>
                <div style="color:#c9b8ff; font-size:1.1rem; font-weight:600; margin:8px 0 4px;">{level_title}</div>
                <div class="kawaii-score">{final_prob*100:.1f}%</div>
                <div class="kawaii-score-label">Probabilitas Love Bombing</div>
                <div style="margin:12px 0;">
                    <span class="kawaii-badge {kawaii_level_css}">{final_level}</span>
                </div>
                <div style="background:rgba(255,255,255,0.07);border-radius:50px;height:12px;margin:16px auto;max-width:320px;overflow:hidden;">
                    <div style="height:100%;width:{final_prob*100:.1f}%;background:linear-gradient(90deg,{color}88,{color});border-radius:50px;transition:width 1s ease;"></div>
                </div>
                <div style="color:#aaa;font-size:0.9rem;">{level_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Details row ──
            d1, d2 = st.columns([1, 1])

            with d1:
                # Extracted messages
                st.markdown("""
                <div style="font-size:0.72rem; font-weight:800; letter-spacing:2.5px;
                            text-transform:uppercase; color:#c4b5fd; margin-bottom:10px;">
                    Pesan Terdeteksi
                </div>
                """, unsafe_allow_html=True)

                msgs = result.get("extracted_messages", [])
                sender_name = result.get("sender_name", "Pengirim")
                plat_detected = result.get("platform_detected", selected_platform)
                total_visible = result.get("msg_count_visible", len(msgs))

                st.markdown(f"""
                <div class="kawaii-card" style="margin-bottom:10px;">
                    <span class="feature-chip">🌐 {plat_detected}</span>
                    <span class="feature-chip">👤 {sender_name}</span>
                    <span class="feature-chip">💬 {total_visible} pesan terlihat</span>
                </div>
                """, unsafe_allow_html=True)

                if msgs:
                    for msg in msgs[:12]:
                        sender = msg.get("sender", "?")
                        text   = msg.get("text", "")
                        time_  = msg.get("time", "")
                        is_main = sender.lower() == sender_name.lower() if sender_name else True
                        bubble_class = "msg-bubble sender" if is_main else "msg-bubble"
                        time_str = f'<span style="color:#666;font-size:0.75rem;"> · {time_}</span>' if time_ else ""
                        st.markdown(f"""
                        <div class="{bubble_class}">
                            <span style="color:#c9b8ff;font-size:0.78rem;font-weight:600;">{sender}{time_str}</span><br>
                            {text}
                        </div>
                        """, unsafe_allow_html=True)
                    if len(msgs) > 12:
                        st.markdown(f'<div style="color:#666;font-size:0.85rem;text-align:center;padding:8px;">... dan {len(msgs)-12} pesan lainnya</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="kawaii-tip">😿 Pesan tidak berhasil diekstrak — coba gambar yang lebih jelas ya~</div>', unsafe_allow_html=True)

            with d2:
                # Red flags
                st.markdown("""
                <div style="font-size:0.72rem; font-weight:800; letter-spacing:2.5px;
                            text-transform:uppercase; color:#c4b5fd; margin-bottom:10px;">
                    Red Flags
                </div>
                """, unsafe_allow_html=True)

                red_flags = result.get("red_flags", [])
                if red_flags:
                    for flag in red_flags:
                        st.markdown(f"""
                        <div style="background:#fff0f4; border:2px solid #ffb3c6;
                                    border-radius:16px; padding:10px 14px; margin:6px 0;
                                    color:#be185d; font-size:0.9rem; font-weight:700;">
                            🚩 {flag}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#dcfce7; border:2px solid #4ade80;
                                border-radius:16px; padding:14px; color:#15803d;
                                text-align:center; font-size:0.95rem; font-weight:800;">
                        🍀 Tidak ada red flag yang terdeteksi!
                    </div>
                    """, unsafe_allow_html=True)

                # Analysis summary
                summary = result.get("analysis_summary", "")
                if summary:
                    st.markdown(f"""
                    <div style="margin-top:16px;">
                        <div style="color:#7c5a9e; font-weight:800; font-size:1rem; margin-bottom:10px;">
                            🧠 Ringkasan Analisis AI
                        </div>
                        <div class="kawaii-card" style="font-size:0.92rem; line-height:1.8; color:#5a3d6e;">
                            {summary}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence
                conf = result.get("confidence", 0)
                conf_color = "#4ade80" if conf > 0.7 else "#facc15" if conf > 0.4 else "#f87171"
                conf_bg    = "#dcfce7" if conf > 0.7 else "#fef9c3" if conf > 0.4 else "#fee2e2"
                st.markdown(f"""
                <div style="margin-top:14px;">
                    <div style="color:#a07cc5; font-size:0.82rem; font-weight:700; margin-bottom:6px;">
                        🎯 Tingkat Keyakinan AI
                    </div>
                    <div style="background:#f3e8ff; border-radius:50px; height:10px; overflow:hidden;">
                        <div style="height:100%; width:{conf*100:.0f}%;
                                    background:linear-gradient(90deg,{conf_color}88,{conf_color});
                                    border-radius:50px;"></div>
                    </div>
                    <div style="background:{conf_bg}; border:1px solid {conf_color}; border-radius:50px;
                                display:inline-block; padding:2px 12px; margin-top:6px;
                                font-size:0.82rem; font-weight:800; color:{conf_color};">
                        {conf*100:.0f}% yakin
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Feature breakdown ──
            st.markdown("---")
            st.markdown("""
            <div style="font-size:0.72rem; font-weight:800; letter-spacing:2.5px;
                        text-transform:uppercase; color:#c4b5fd; margin-bottom:12px;">
                Detail Fitur yang Diekstrak
            </div>
            """, unsafe_allow_html=True)

            feat_labels = {
                "msg_per_day_week1":          ("📱 Pesan/hari minggu 1",       lambda v: v > 25),
                "msg_per_day_week4":          ("📉 Pesan/hari minggu 4",       lambda v: False),
                "praise_ratio":               ("💝 Rasio pujian",              lambda v: v > 0.35),
                "avg_response_time_min":      ("⚡ Waktu respons (menit)",     lambda v: v < 3),
                "emotional_intensity_score":  ("🔥 Intensitas emosi /10",      lambda v: v > 7),
                "commitment_pressure_ratio":  ("💍 Tekanan komitmen",          lambda v: v > 0.25),
                "isolation_attempt_count":    ("🚧 Upaya isolasi",             lambda v: v > 4),
                "escalation_speed_days":      ("⚡ Eskalasi (hari)",           lambda v: v < 10),
                "consistency_score":          ("🎭 Konsistensi /10",           lambda v: v < 4),
                "night_msg_ratio":            ("🌙 Pesan malam",               lambda v: v > 0.4),
                "apology_count":              ("🙏 Permintaan maaf",           lambda v: v > 15),
                "future_planning_ratio":      ("💫 Rencana masa depan",        lambda v: v > 0.45),
            }

            fcols = st.columns(3)
            feat_items = [(k, v) for k, v in features_raw.items() if k in feat_labels]
            for i, (fkey, fval) in enumerate(feat_items):
                label, is_warning = feat_labels[fkey]
                chip_class = "feature-chip warning" if is_warning(fval) else "feature-chip"
                warn_icon  = "⚠️ " if is_warning(fval) else ""
                if isinstance(fval, float):
                    disp = f"{fval:.2f}"
                else:
                    disp = str(fval)
                with fcols[i % 3]:
                    st.markdown(f'<span class="{chip_class}">{warn_icon}{label}: <b>{disp}</b></span>', unsafe_allow_html=True)

            # ── Action advice ──
            st.markdown("---")
            if final_level == "LOW":
                st.markdown("""
                <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
                            border:2px solid #4ade80; border-radius:24px; padding:24px; text-align:center;
                            box-shadow:0 4px 20px rgba(74,222,128,0.15);">
                    <div style="font-size:2.5rem; margin-bottom:8px;">🍀✨🌿</div>
                    <div style="color:#15803d; font-weight:900; font-size:1.15rem; margin:8px 0;">
                        Percakapan Terlihat Sehat~
                    </div>
                    <div style="color:#16a34a; font-size:0.92rem; font-weight:600;">
                        Tetap jaga batas diri dan komunikasi terbuka ya! 💚
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif final_level == "MEDIUM":
                st.markdown("""
                <div style="background:linear-gradient(135deg,#fefce8,#fef9c3);
                            border:2px solid #facc15; border-radius:24px; padding:24px; text-align:center;
                            box-shadow:0 4px 20px rgba(250,204,21,0.15);">
                    <div style="font-size:2.5rem; margin-bottom:8px;">🐰💛🌻</div>
                    <div style="color:#a16207; font-weight:900; font-size:1.15rem; margin:8px 0;">
                        Ada yang Perlu Diperhatikan~
                    </div>
                    <div style="color:#b45309; font-size:0.92rem; font-weight:600;">
                        Percayai instingmu dan diskusikan dengan orang terpercaya 🌻
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif final_level in ["HIGH", "CRITICAL"]:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#fff0f4,#fce7f3);
                            border:2px solid #f472b6; border-radius:24px; padding:28px; text-align:center;
                            box-shadow:0 4px 20px rgba(244,114,182,0.2);">
                    <div style="font-size:2.5rem; animation:heartbeat 1.5s infinite; display:inline-block;">🚨</div>
                    <div style="color:#be185d; font-weight:900; font-size:1.15rem; margin:12px 0;">
                        Pola Love Bombing Signifikan Terdeteksi!
                    </div>
                    <div style="color:#9d174d; font-size:0.92rem; font-weight:700; line-height:2.2;">
                        💬 Ceritakan ke teman / keluarga terpercaya<br>
                        🧠 Pertimbangkan konsultasi dengan psikolog<br>
                        🚪 Ingat: kamu berhak menentukan batasmu sendiri<br>
                        📞 Hotline konseling:
                        <span style="background:#be185d; color:white; padding:2px 10px;
                                     border-radius:50px; font-weight:900;">119 ext 8</span>
                        &nbsp;Into The Light Indonesia
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif not uploaded_img:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px;
                    background:linear-gradient(135deg,#fff5fb,#f5f0ff,#f0f8ff);
                    border:2px dashed #e9d5ff; border-radius:28px; margin-top:16px;">
            <div style="font-size:4rem; animation:float 3s ease-in-out infinite;
                        display:inline-block; margin-bottom:16px;">📸</div>
            <div style="font-size:1.05rem; color:#7c5a9e; font-weight:800;">
                Upload screenshot chat di atas untuk mulai~ ✨
            </div>
            <div style="font-size:0.88rem; margin-top:10px; color:#a07cc5; font-weight:600;">
                AI akan baca isi chatnya dan deteksi pola love bombing 🔍💕
            </div>
            <div style="font-size:1.2rem; letter-spacing:6px; margin-top:14px; opacity:0.6;">
                🌸 💜 💙 💛 💚
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Akurasi Model
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 16px;">
        <span style="font-size:1.6rem; font-weight:900; color:#7c5a9e;">📊 Performa Model</span><br>
        <span style="font-size:0.9rem; color:#a07cc5; font-weight:600;">Seberapa jago si AI kita? 🤖✨</span>
    </div>
    """, unsafe_allow_html=True)
    
    if metrics is None:
        st.warning("Metrics belum ada. Jalankan `python train_model.py` untuk melatih model.")
    else:
        # Metric cards row 1
        cols = st.columns(4)
        metric_items = [
            ("Accuracy", f"{metrics['accuracy']*100:.2f}%", "Kebenaran prediksi keseluruhan"),
            ("Precision", f"{metrics['precision']*100:.2f}%", "Ketepatan deteksi love bombing"),
            ("Recall", f"{metrics['recall']*100:.2f}%", "Kemampuan menemukan semua kasus LB"),
            ("F1-Score", f"{metrics['f1_score']*100:.2f}%", "Keseimbangan Precision & Recall"),
        ]
        for col, (title, val, desc) in zip(cols, metric_items):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{title}</div></div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Metric cards row 2
        cols2 = st.columns(4)
        m2 = [
            ("AUC-ROC", f"{metrics['auc_roc']:.4f}", "Area Under Curve"),
            ("CV F1 Mean", f"{metrics['cv_f1_mean']*100:.2f}%", "Cross-validation (5-fold)"),
            ("CV F1 Std", f"±{metrics['cv_f1_std']*100:.2f}%", "Stabilitias model"),
            ("Test Samples", f"{metrics['n_test']:,}", "Jumlah data uji"),
        ]
        for col, (title, val, desc) in zip(cols2, m2):
            col.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.7rem">{val}</div><div class="metric-label">{title}<br><span style="font-size:0.75rem; color:#c4b5fd;">{desc}</span></div></div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Plots
        pc1, pc2, pc3 = st.columns(3)
        
        with pc1:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()
        
        with pc2:
            st.markdown("**Feature Importance**")
            fig_fi = plot_feature_importance(metrics["feature_importance"])
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()
        
        with pc3:
            st.markdown("**ROC Curve**")
            fig_roc = plot_roc_curve(
                metrics["roc_curve"]["fpr"],
                metrics["roc_curve"]["tpr"],
                metrics["auc_roc"]
            )
            st.pyplot(fig_roc, use_container_width=True)
            plt.close()
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Feature importance table
        st.markdown("**Detail Feature Importance:**")
        fi_df = pd.DataFrame(metrics["feature_importance"])
        fi_df["importance_pct"] = (fi_df["importance"] * 100).round(2)
        fi_df = fi_df[["label", "importance_pct"]].rename(columns={"label": "Fitur", "importance_pct": "Importance (%)"})
        st.dataframe(fi_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Panduan
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📖 Cara Penggunaan")
    
    st.markdown("""
    #### Langkah Persiapan
    
    **1. Install Dependensi**
    ```bash
    pip install anthropic pandas scikit-learn streamlit matplotlib seaborn Pillow
    ```

    **2. Generate Dataset (100.000 data)**
    ```bash
    python generate_dataset.py
    ```
    
    **3. Training Model**
    ```bash
    python train_model.py
    ```

    **4. Jalankan Web App**
    ```bash
    streamlit run app.py
    ```

    **5. Fitur Scan Chat (Tab 📸)**
    - Upload screenshot chat dari WhatsApp, Instagram DM, Telegram, Line, dll
    - Pilih platform atau biarkan Auto-detect
    - Claude Vision AI akan menganalisis gambar dan mengekstrak pola pesan
    - Butuh `ANTHROPIC_API_KEY` yang valid di environment variable
    
    ---
    
    #### Deploy ke Hugging Face Spaces (Gratis)
    
    1. Buat akun di [huggingface.co](https://huggingface.co)
    2. Buat Space baru → pilih **Streamlit**
    3. Upload semua file:
       - `app.py`
       - `model_love_bombing.pkl`
       - `metrics_report.json`
       - `requirements.txt`
    4. App otomatis live dalam beberapa menit
    
    ---
    
    #### Penjelasan Fitur
    """)
    
    feature_data = [
        ("msg_per_day_week1", "Pesan/hari minggu 1", "Frekuensi pesan di minggu pertama – love bomber sangat aktif di awal"),
        ("praise_ratio", "Rasio pujian", "Proporsi pesan yang mengandung pujian berlebihan"),
        ("avg_response_time_min", "Waktu respons", "Love bomber selalu merespons sangat cepat untuk menciptakan ketergantungan"),
        ("emotional_intensity_score", "Intensitas emosi", "Seberapa intens emosi yang ditunjukkan per pesan"),
        ("commitment_pressure_ratio", "Tekanan komitmen", "Seberapa sering memaksa komitmen (kenalan 2 hari tapi minta jadi pacar)"),
        ("isolation_attempt_count", "Percobaan isolasi", "Upaya memisahkan korban dari teman/keluarga"),
        ("escalation_speed_days", "Kecepatan eskalasi", "Berapa hari sampai hubungan menjadi sangat intens – makin cepat makin merah"),
        ("consistency_score", "Konsistensi perilaku", "Perilaku konsisten atau berubah drastis – love bomber berubah setelah 'dapat'"),
        ("night_msg_ratio", "Pesan malam", "Pesan tengah malam berlebihan bisa menandakan kontrol"),
    ]
    
    ft_df = pd.DataFrame(feature_data, columns=["Fitur", "Label", "Penjelasan"])
    st.dataframe(ft_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ---
    #### ⚠️ Disclaimer
    
    Sistem ini **bukan alat diagnosis** hubungan. Hasil prediksi bersifat indikatif berdasarkan pola statistik.
    Selalu konsultasikan dengan profesional (psikolog/konselor) untuk situasi yang membutuhkan penanganan serius.
    Dataset yang digunakan adalah data sintetis untuk tujuan penelitian.
    """)
