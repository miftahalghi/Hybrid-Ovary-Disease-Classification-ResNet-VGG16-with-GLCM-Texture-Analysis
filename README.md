# OvarAI — Deteksi Kista Ovarium 🔬

Aplikasi web berbasis Streamlit untuk deteksi kondisi ovarium dari citra USG menggunakan **Dual-Branch CNN + GLCM**.

---

## 📁 Struktur Folder

```
streamlit_app/
│
├── app.py                        ← Aplikasi utama Streamlit
├── requirements.txt              ← Dependensi Python
├── README.md                     ← Panduan ini
│
└── models/                       
    ├── best_s1_vgg16_glcm.keras  ← Model Skema 1 (dari Kaggle output)
    ├── best_s2_resnet_glcm.keras ← Model Skema 2 (dari Kaggle output)
    └── glcm_scaler.pkl           ← Scaler GLCM (dari Kaggle output)
```

## 💻 Jalankan Lokal

```bash
# Install dependensi
pip install -r requirements.txt

# Jalankan app
streamlit run app.py
```

---

## 🔧 Konfigurasi Opsional

Buat file `.streamlit/config.toml` untuk kustomisasi:

```toml
[server]
maxUploadSize = 50

[theme]
primaryColor = "#0d9488"
backgroundColor = "#fdfcfb"
secondaryBackgroundColor = "#f0fdfa"
textColor = "#1e293b"
```

---

## 📊 Kelas yang Dideteksi

| Kelas | Label | Keterangan |
|-------|-------|------------|
| `complex_cyst` | Kista Kompleks | Kista dengan komponen internal, perlu evaluasi lanjut |
| `dominant_follicle` | Folikel Dominan | Normal, menandakan ovulasi |
| `healthy` | Ovarium Sehat | Tampilan normal |
| `poly_cyst` | Polikista (PCOS) | Gambaran PCOS, perlu evaluasi hormonal |
| `simple_cyst` | Kista Sederhana | Umumnya jinak, pantau berkala |

---

## ⚠️ Disclaimer

Sistem ini bersifat **bantuan skrining** berbasis AI dan **bukan pengganti diagnosis medis**.
Selalu konsultasikan hasil dengan dokter spesialis kandungan.
