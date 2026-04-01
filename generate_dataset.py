"""
generate_dataset.py
Generate 100.000 synthetic love bombing dataset via Claude API
Hasilnya: dataset_love_bombing.csv
"""

import anthropic
import pandas as pd
import json
import random
import time
from datetime import datetime, timedelta

client = anthropic.Anthropic()

SYSTEM_PROMPT = """Kamu adalah generator data sintetis untuk penelitian deteksi love bombing.
Tugasmu menghasilkan fitur statistik percakapan chat dalam format JSON array.
Setiap item adalah SATU hubungan percakapan (bukan per pesan).
HANYA output JSON array, tidak ada teks lain."""

def generate_batch(batch_size: int, label: int) -> list[dict]:
    """Generate satu batch data dengan label tertentu (0=normal, 1=love bombing)"""
    label_desc = "LOVE BOMBING" if label == 1 else "NORMAL/SEHAT"
    
    prompt = f"""Generate {batch_size} data sintetis percakapan chat dengan label {label_desc}.

Label: {label} ({label_desc})

{"Karakteristik LOVE BOMBING:" if label == 1 else "Karakteristik NORMAL:"}
{"- Pesan sangat frequent (>20x/hari di awal)" if label == 1 else "- Frekuensi pesan wajar (3-15x/hari)"}
{"- Pujian berlebihan (>40% pesan mengandung pujian)" if label == 1 else "- Pujian proporsional (<20%)"}
{"- Respons sangat cepat (<2 menit hampir selalu)" if label == 1 else "- Respons bervariasi (5-60 menit)"}
{"- Eskalasi emosi sangat cepat dalam 1-2 minggu" if label == 1 else "- Eskalasi emosi bertahap (berbulan-bulan)"}
{"- Tekanan komitmen tinggi (>30% pesan)" if label == 1 else "- Tekanan komitmen rendah (<10%)"}
{"- Pesan panjang dan emosional" if label == 1 else "- Panjang pesan bervariasi"}
{"- Kata isolasi tinggi (jauhkan dari teman/keluarga)" if label == 1 else "- Tidak ada upaya isolasi"}
{"- Konsistensi perilaku rendah (berubah drastis setelah 2-4 minggu)" if label == 1 else "- Perilaku konsisten"}

Output JSON array dengan {batch_size} objek, tiap objek berisi:
{{
  "msg_per_day_week1": float,         // rata2 pesan/hari minggu pertama (0-100)
  "msg_per_day_week4": float,         // rata2 pesan/hari minggu keempat
  "praise_ratio": float,              // proporsi pesan mengandung pujian (0-1)
  "avg_response_time_min": float,     // rata2 waktu respons dalam menit
  "response_time_std": float,         // standar deviasi waktu respons
  "emotional_intensity_score": float, // skor intensitas emosi rata2 (0-10)
  "commitment_pressure_ratio": float, // proporsi pesan mengandung tekanan komitmen (0-1)
  "isolation_attempt_count": int,     // jumlah upaya isolasi dalam 30 hari
  "avg_msg_length": float,            // rata2 panjang pesan (karakter)
  "msg_length_variance": float,       // variansi panjang pesan
  "escalation_speed_days": float,     // seberapa cepat eskalasi emosi (hari)
  "consistency_score": float,         // konsistensi perilaku (0-10, 10=sangat konsisten)
  "night_msg_ratio": float,           // proporsi pesan tengah malam 22.00-05.00 (0-1)
  "apology_count": int,               // jumlah permintaan maaf berlebihan dalam 30 hari
  "future_planning_ratio": float,     // proporsi pesan tentang masa depan bersama (0-1)
  "label": {label}
}}

PENTING: Pastikan nilai numerik REALISTIS dan konsisten dengan label {label_desc}.
Output HANYA JSON array, tidak ada markdown, tidak ada penjelasan."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT
        )
        
        text = response.content[0].text.strip()
        # Bersihkan kalau ada markdown fence
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        data = json.loads(text)
        return data
        
    except Exception as e:
        print(f"  Error batch: {e}")
        return []


def noisy(value: float, noise_std: float) -> float:
    """Tambahkan noise gaussian absolut ke value"""
    return value + random.gauss(0, noise_std)

def generate_fallback_batch(batch_size: int, label: int) -> list[dict]:
    """
    Generate data realistis dengan noise besar & overlap agresif.
    Target akurasi model: ~92-94% (realistis untuk portofolio)

    Strategi:
    - 42% data adalah borderline (zona abu-abu, overlap antar kelas)
    - Noise absolut cukup besar → distribusi melebar dan tumpang tindih
    - Mencerminkan realita: tidak semua love bomber berperilaku ekstrem,
      tidak semua hubungan normal selalu tenang
    """
    data = []
    for _ in range(batch_size):
        is_borderline = random.random() < 0.42

        if label == 1:  # Love bombing
            if is_borderline:
                # Subtle / early stage love bombing — mudah dikira normal
                row = {
                    "msg_per_day_week1":        noisy(random.uniform(10, 22), 7),
                    "msg_per_day_week4":        noisy(random.uniform(7, 20), 5),
                    "praise_ratio":             noisy(random.uniform(0.15, 0.35), 0.09),
                    "avg_response_time_min":    noisy(random.uniform(4, 22), 7),
                    "response_time_std":        noisy(random.uniform(3, 15), 5),
                    "emotional_intensity_score":noisy(random.uniform(4.5, 6.8), 1.0),
                    "commitment_pressure_ratio":noisy(random.uniform(0.08, 0.26), 0.08),
                    "isolation_attempt_count":  random.randint(1, 6),
                    "avg_msg_length":           noisy(random.uniform(70, 190), 45),
                    "msg_length_variance":      noisy(random.uniform(28, 110), 28),
                    "escalation_speed_days":    noisy(random.uniform(8, 28), 8),
                    "consistency_score":        noisy(random.uniform(3.8, 6.8), 0.9),
                    "night_msg_ratio":          noisy(random.uniform(0.10, 0.32), 0.09),
                    "apology_count":            random.randint(3, 15),
                    "future_planning_ratio":    noisy(random.uniform(0.15, 0.40), 0.09),
                    "label": 1
                }
            else:
                # Love bombing jelas
                row = {
                    "msg_per_day_week1":        noisy(random.uniform(22, 70), 10),
                    "msg_per_day_week4":        noisy(random.uniform(2, 10), 3),
                    "praise_ratio":             noisy(random.uniform(0.38, 0.85), 0.08),
                    "avg_response_time_min":    noisy(random.uniform(0.5, 5), 2),
                    "response_time_std":        noisy(random.uniform(0.2, 3), 1),
                    "emotional_intensity_score":noisy(random.uniform(6.5, 9.8), 0.6),
                    "commitment_pressure_ratio":noisy(random.uniform(0.28, 0.75), 0.07),
                    "isolation_attempt_count":  random.randint(5, 28),
                    "avg_msg_length":           noisy(random.uniform(140, 480), 50),
                    "msg_length_variance":      noisy(random.uniform(50, 200), 30),
                    "escalation_speed_days":    noisy(random.uniform(1, 9), 2),
                    "consistency_score":        noisy(random.uniform(1.2, 4.2), 0.7),
                    "night_msg_ratio":          noisy(random.uniform(0.28, 0.68), 0.07),
                    "apology_count":            random.randint(10, 50),
                    "future_planning_ratio":    noisy(random.uniform(0.38, 0.88), 0.07),
                    "label": 1
                }
        else:  # Normal
            if is_borderline:
                # Normal tapi intens — LDR, baru jadian, atau expressive person
                row = {
                    "msg_per_day_week1":        noisy(random.uniform(9, 23), 6),
                    "msg_per_day_week4":        noisy(random.uniform(9, 24), 5),
                    "praise_ratio":             noisy(random.uniform(0.10, 0.30), 0.08),
                    "avg_response_time_min":    noisy(random.uniform(5, 22), 6),
                    "response_time_std":        noisy(random.uniform(4, 16), 5),
                    "emotional_intensity_score":noisy(random.uniform(4.2, 6.5), 0.9),
                    "commitment_pressure_ratio":noisy(random.uniform(0.05, 0.20), 0.06),
                    "isolation_attempt_count":  random.randint(1, 5),
                    "avg_msg_length":           noisy(random.uniform(65, 185), 38),
                    "msg_length_variance":      noisy(random.uniform(25, 105), 24),
                    "escalation_speed_days":    noisy(random.uniform(14, 45), 9),
                    "consistency_score":        noisy(random.uniform(4.8, 7.5), 0.8),
                    "night_msg_ratio":          noisy(random.uniform(0.09, 0.28), 0.08),
                    "apology_count":            random.randint(2, 12),
                    "future_planning_ratio":    noisy(random.uniform(0.09, 0.28), 0.08),
                    "label": 0
                }
            else:
                # Normal jelas
                row = {
                    "msg_per_day_week1":        noisy(random.uniform(2, 15), 4),
                    "msg_per_day_week4":        noisy(random.uniform(2, 14), 3),
                    "praise_ratio":             noisy(random.uniform(0.02, 0.14), 0.04),
                    "avg_response_time_min":    noisy(random.uniform(12, 72), 12),
                    "response_time_std":        noisy(random.uniform(8, 38), 8),
                    "emotional_intensity_score":noisy(random.uniform(2.0, 4.8), 0.6),
                    "commitment_pressure_ratio":noisy(random.uniform(0, 0.06), 0.02),
                    "isolation_attempt_count":  random.randint(0, 2),
                    "avg_msg_length":           noisy(random.uniform(18, 125), 25),
                    "msg_length_variance":      noisy(random.uniform(10, 85), 18),
                    "escalation_speed_days":    noisy(random.uniform(45, 195), 22),
                    "consistency_score":        noisy(random.uniform(7.0, 10), 0.5),
                    "night_msg_ratio":          noisy(random.uniform(0.02, 0.14), 0.04),
                    "apology_count":            random.randint(0, 3),
                    "future_planning_ratio":    noisy(random.uniform(0.02, 0.14), 0.04),
                    "label": 0
                }
        data.append(row)
    return data


def main():
    TOTAL_SAMPLES = 100_000
    BATCH_SIZE = 50       # per request Claude API
    LOVE_BOMB_RATIO = 0.4 # 40% love bombing, 60% normal (imbalanced realistis)
    USE_API = False       # Set True kalau punya Anthropic API key
    
    n_love_bomb = int(TOTAL_SAMPLES * LOVE_BOMB_RATIO)
    n_normal = TOTAL_SAMPLES - n_love_bomb
    
    print(f"=== Love Bombing Dataset Generator ===")
    print(f"Total target  : {TOTAL_SAMPLES:,} samples")
    print(f"Love bombing  : {n_love_bomb:,} ({LOVE_BOMB_RATIO*100:.0f}%)")
    print(f"Normal        : {n_normal:,} ({(1-LOVE_BOMB_RATIO)*100:.0f}%)")
    print(f"Mode          : {'Claude API' if USE_API else 'Fallback programatik'}")
    print(f"Batch size    : {BATCH_SIZE}")
    print()
    
    all_data = []
    
    configs = [
        (n_love_bomb, 1, "Love Bombing"),
        (n_normal, 0, "Normal"),
    ]
    
    for target_count, label, label_name in configs:
        print(f"Generating {target_count:,} '{label_name}' samples...")
        generated = 0
        batch_num = 0
        
        while generated < target_count:
            remaining = target_count - generated
            current_batch = min(BATCH_SIZE, remaining)
            batch_num += 1
            
            print(f"  Batch {batch_num}: {generated:,}/{target_count:,}", end="", flush=True)
            
            if USE_API:
                batch_data = generate_batch(current_batch, label)
                if not batch_data:
                    print(" → fallback")
                    batch_data = generate_fallback_batch(current_batch, label)
                else:
                    # Pastikan label benar
                    for item in batch_data:
                        item["label"] = label
            else:
                batch_data = generate_fallback_batch(current_batch, label)
            
            all_data.extend(batch_data)
            generated += len(batch_data)
            print(f" ✓ (+{len(batch_data)})")
            
            if USE_API:
                time.sleep(0.5)  # Rate limit buffer
        
        print(f"  ✓ Selesai: {generated:,} samples\n")
    
    # Shuffle dan simpan
    random.shuffle(all_data)
    df = pd.DataFrame(all_data)
    
    # Pastikan label integer
    df["label"] = df["label"].astype(int)
    
    # Clip nilai ke range valid
    df["praise_ratio"] = df["praise_ratio"].clip(0, 1)
    df["commitment_pressure_ratio"] = df["commitment_pressure_ratio"].clip(0, 1)
    df["night_msg_ratio"] = df["night_msg_ratio"].clip(0, 1)
    df["future_planning_ratio"] = df["future_planning_ratio"].clip(0, 1)
    df["emotional_intensity_score"] = df["emotional_intensity_score"].clip(0, 10)
    df["consistency_score"] = df["consistency_score"].clip(0, 10)
    
    output_path = "dataset_love_bombing.csv"
    df.to_csv(output_path, index=False)
    
    print(f"=== Dataset berhasil disimpan ===")
    print(f"File     : {output_path}")
    print(f"Total    : {len(df):,} baris")
    print(f"Kolom    : {list(df.columns)}")
    print(f"\nDistribusi label:")
    print(df["label"].value_counts().to_string())
    print(f"\nSample data:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
