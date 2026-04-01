# 🛡️ Love Bombing Detector

Sistem deteksi pola love bombing dari data percakapan menggunakan Machine Learning (Random Forest).

## Fitur
- Generate 100.000 synthetic dataset via Claude API
- Training Random Forest dengan evaluasi lengkap
- Web app Streamlit dengan:
  - Analisis manual (input slider)
  - Batch prediction (upload CSV)
  - Dashboard akurasi model (confusion matrix, ROC, feature importance)

## Cara Pakai

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key Anthropic
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Generate dataset (100.000 samples)
```bash
python generate_dataset.py
```
> Catatan: Kalau tidak punya API key, set `USE_API = False` di generate_dataset.py  
> untuk pakai generator programatik (gratis, instant, tidak perlu API)

### 4. Training model
```bash
python train_model.py
```
Output: `model_love_bombing.pkl`, `metrics_report.json`, `model_plots.png`

### 5. Jalankan web app
```bash
streamlit run app.py
```

---

## Deploy ke Hugging Face Spaces (Gratis)

1. Daftar di https://huggingface.co
2. New Space → Streamlit → nama repo bebas
3. Upload file: `app.py`, `model_love_bombing.pkl`, `metrics_report.json`, `requirements.txt`
4. App live otomatis 🎉

---

## Struktur File
```
love_bombing_detector/
├── generate_dataset.py     # Generate 100k synthetic data via Claude API
├── train_model.py          # Training + evaluasi model
├── app.py                  # Streamlit web app
├── requirements.txt
├── README.md
├── dataset_love_bombing.csv     # (generated)
├── model_love_bombing.pkl       # (generated)
├── metrics_report.json          # (generated)
└── model_plots.png              # (generated)
```

## 15 Fitur yang Dianalisis

| Fitur | Deskripsi |
|-------|-----------|
| msg_per_day_week1 | Pesan/hari minggu pertama |
| msg_per_day_week4 | Pesan/hari minggu keempat |
| praise_ratio | Rasio pujian berlebihan |
| avg_response_time_min | Waktu respons rata-rata |
| response_time_std | Variansi waktu respons |
| emotional_intensity_score | Intensitas emosi (0-10) |
| commitment_pressure_ratio | Tekanan komitmen |
| isolation_attempt_count | Percobaan isolasi |
| avg_msg_length | Panjang pesan rata-rata |
| msg_length_variance | Variansi panjang pesan |
| escalation_speed_days | Kecepatan eskalasi emosi |
| consistency_score | Konsistensi perilaku (0-10) |
| night_msg_ratio | Rasio pesan tengah malam |
| apology_count | Jumlah permintaan maaf berlebihan |
| future_planning_ratio | Rasio rencana masa depan bersama |

---

⚠️ **Disclaimer**: Sistem ini untuk tujuan edukasi & penelitian. Bukan alat diagnosis hubungan.
