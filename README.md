# 💧 Aplikasi Cek Kelayakan Air Minum (Water Potability Prediction)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> **Aplikasi Berbasis Web untuk Memprediksi Kelayakan Air Minum Menggunakan Algoritma pemodelan Random Forest**

Aplikasi ini dibuat sebagai bagian dari proses Data Mining (Clasification) untuk dataset 'water_potability.csv', serta sebagai pemenuhan syarat Final Showcase Karsakarsa Data Mining Course.

---

## 📌 Fitur

### 1. 🤖 Prediction (Random Forest)
Menggunakan algoritma **Random Forest Classifier** yang telah dilatih untuk mendeteksi pola kompleks dari 9 parameter kualitas air. Sistem juga dilengkapi dengan **Confidence Score** (Tingkat Keyakinan) untuk mengetahui seberapa yakin model terhadap prediksinya.

### 2. 📊 Radar Chart Visualization
Visualisasi interaktif berbentuk jaring laba-laba (Spider Plot) yang membandingkan profil air pengguna (Garis Biru) dengan Standar Aman WHO (Area Hijau). Pengguna dapat dengan cepat melihat parameter mana yang "melonjak" keluar dari batas aman.

### 3. 📝 Laporan Kesehatan Fisik Air
Analisis otomatis berbasis aturan (*rule-based*) yang fokus pada 4 indikator fisik yang paling mudah dikenali oleh indra manusia:
- **pH** (Derajat Keasaman)
- **Hardness** (Kekerasan Air/Kerak)
- **Solids / TDS** (Padatan Terlarut)
- **Turbidity** (Kekeruhan)

---

## 🛠️ Teknologi & Library

Proyek ini dibangun menggunakan ekosistem Python:

- **Streamlit:** Framework utama untuk antarmuka web yang interaktif.
- **Scikit-Learn:** Pembuatan model Machine Learning (Random Forest & SVM).
- **Plotly:** Pembuatan grafik Radar Chart interaktif.
- **Pandas & NumPy:** Manipulasi dan pemrosesan data numerik.
- **Joblib:** Penyimpanan dan pemuatan model (`.pkl`).

---

## 📂 Struktur Repositori

| Nama File | Deskripsi |
| :--- | :--- |
| `Proyek_Klasifikasi_Kelayakan_Air_Minum.ipynb` | File Jupyter Notebook berisi proses analisis data, preprocessing, dan pelatihan model. |
| `README.md` | File README.md |
| `app.py` | Source code utama aplikasi Streamlit. |
| `minmax_scaler.pkl` | StandardScaler untuk normalisasi input data user. |
| `model_random_forest.pkl` | Model Random Forest yang sudah dilatih (Best Model). |
| `requirements.txt` | Daftar library yang dibutuhkan untuk deployment cloud. |
| `water_potability.csv`| Dataset mentah (Sumber: Kaggle). |

---
