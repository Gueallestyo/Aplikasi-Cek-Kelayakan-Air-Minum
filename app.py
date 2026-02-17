import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS MOBILE
# ==========================================
st.set_page_config(
    page_title="Cek Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

# --- CSS Tampilan Rapi di HP ---
st.markdown("""
    <style>
        /* Mengurangi margin/padding berlebih di sisi kiri-kanan layar HP */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        /* Agar judul tidak terlalu besar di layar kecil */
        h1 {
            font-size: 1.8rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model_random_forest.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_resources()

if model is None:
    st.error("❌ Error: File model tidak ditemukan. Pastikan rf_model.pkl dan scaler.pkl ada.")
    st.stop()

# ==========================================
# 3. SIDEBAR (INPUT)
# ==========================================
st.sidebar.header("📝 Panel Input Data Sampel Air")
st.sidebar.markdown("Masukkan data laboratorium sampel air:")

# Default Values (Aman / Rata-rata Dataset)
def_ph = 0.0
def_hard = 0.0
def_solid = 0.0
def_chloro = 7.15
def_sulfate = 336.0
def_conduct = 428.0
def_carbon = 14.0
def_trihalo = 66.0
def_turbid = 3.96

# Input Widgets
ph = st.sidebar.number_input("1. pH Level", 0.0, 14.0, def_ph, help="Netral: 7.0. Aman: 6.5-8.5")
hardness = st.sidebar.number_input("2. Hardness (mg/L)", 0.0, 400.0, def_hard)
solids = st.sidebar.number_input("3. Solids (ppm)", 0.0, 60000.0, def_solid)

st.sidebar.markdown("---")
st.sidebar.markdown("**Parameter Tambahan (Opsional):**")

chloramines = st.sidebar.number_input("4. Chloramines (ppm)", 0.0, 14.0, def_chloro)
sulfate = st.sidebar.number_input("5. Sulfate (mg/L)", 0.0, 500.0, def_sulfate)
conductivity = st.sidebar.number_input("6. Conductivity (μS/cm)", 0.0, 800.0, def_conduct)
organic_carbon = st.sidebar.number_input("7. Organic Carbon (ppm)", 0.0, 30.0, def_carbon)
trihalomethanes = st.sidebar.number_input("8. Trihalomethanes (μg/L)", 0.0, 125.0, def_trihalo)
turbidity = st.sidebar.number_input("9. Turbidity (NTU)", 0.0, 7.0, def_turbid)

# ==========================================
# 4. FUNGSI VISUALISASI (RADAR CHART RESPONSIF)
# ==========================================
def plot_radar_chart(input_values):
    categories = [
        'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    values_user = list(input_values)
    values_limit = [8.5, 300, 25000, 10, 400, 600, 20, 100, 5] 
    
    val_norm_user = []
    for v, l in zip(values_user, values_limit):
        if l == 0: l = 1
        val_norm_user.append(v / l)

    val_norm_limit = [1.0] * 9 

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=val_norm_limit, theta=categories, fill='toself', name='Batas Aman', line_color='green', opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=val_norm_user, theta=categories, fill='toself', name='Sampel Anda', line_color='blue', opacity=0.6))

    # --- UPDATED LAYOUT FOR MOBILE ---
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2.0], tickfont=dict(size=9)),
        ),
        showlegend=True,
        # LEGENDA DI BAWAH (Horizontal) agar grafik melebar penuh di HP
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        # Margin Tipis agar tidak buang tempat di layar kecil
        margin=dict(l=30, r=30, t=30, b=50),
        height=450,
        font=dict(size=11)
    )
    return fig

# ==========================================
# 5. HALAMAN UTAMA
# ==========================================
st.title("💧 Aplikasi Cek Kelayakan Air Minum")
st.markdown("Aplikasi ini menggunakan **Algoritma pemodelan Random Forest** untuk memprediksi layak atau tidaknya air untuk diminum. Silakan masukkan data sampel air hasil uji lab di sidebar sebelah kiri.")

# --- BAGIAN EDUKASI ---
st.info("ℹ️ **Panduan Standar Kualitas Air WHO & Kemenkes (Penjelasan Parameter)**")
with st.expander("Klik di sini untuk melihat Penjelasan Lengkap ke-9 Parameter", expanded=False):
    st.markdown("""
    | Parameter | Rentang Aman (Ideal) | Penjelasan & Bahaya Jika Berlebih |
    | :--- | :--- | :--- |
    | **1. pH** | 6.5 - 8.5 | Derajat keasaman. Jika < 6 (Asam) korosif, jika > 8.5 (Basa) pahit. |
    | **2. Hardness** | < 300 mg/L | Kekerasan air. Tinggi = Berkerak & sabun sulit berbusa. |
    | **3. Solids (TDS)** | < 500 ppm | Padatan terlarut. Tinggi = Air keruh, asin/logam. |
    | **4. Chloramines** | < 4.0 ppm | Disinfektan. Berlebih = Bau menyengat & iritasi. |
    | **5. Sulfate** | < 250 mg/L | Mineral alami. Tinggi = Rasa pahit & diare. |
    | **6. Conductivity** | < 400 μS/cm | Daya hantar listrik (indikator jumlah mineral). |
    | **7. Org. Carbon** | < 2.0 ppm | Sisa material organik. Indikator adanya bakteri. |
    | **8. Trihalomethanes**| < 80 μg/L | Zat sisa klorin. **Berbahaya**: Memicu kanker jangka panjang. |
    | **9. Turbidity** | < 5.0 NTU | Kekeruhan (lumpur/debu). Tempat bakteri bersembunyi. |
    """)

# ==========================================
# 6. LOGIKA & OUTPUT
# ==========================================
st.markdown("---")

# Menggunakan use_container_width=True agar tombol pas di layar HP
if st.button("🔍 CEK KELAYAKAN SAMPEL AIR", type="primary", use_container_width=True):
    
    # --- PROCESS ---
    with st.spinner('Sedang memeriksa ke-9 parameter kimia...'):
        time.sleep(1.0)
    with st.spinner('Mencocokkan dengan pola Algoritma Random Forest...'):
        time.sleep(0.5)
    
    # --- PREDICTION ---
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    input_scaled = scaler.transform(input_data)
    prediksi = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[prediksi] * 100
    
    # --- RESULT HEADER ---
    st.markdown("---")
    # Di HP col1 dan col2 akan otomatis menumpuk (stack)
    col_status, col_meter = st.columns([1.5, 1])

    with col_status:
        st.subheader("Hasil Pengecekan:")
        if prediksi == 1:
            st.success("✅ **STATUS: AIR LAYAK MINUM (POTABLE)**")
            st.markdown("**Keterangan:** Sampel air diprediksi **AMAN** untuk dikonsumsi.")
        else:
            st.error("⛔ **STATUS: AIR TIDAK LAYAK MINUM (NOT POTABLE)**")
            st.markdown("**Keterangan:** Sampel air diprediksi **TIDAK AMAN** / berpotensi berbahaya untuk dikonsumsi.")

    with col_meter:
        st.subheader("Tingkat Keyakinan:")
        st.metric(label="Probabilitas Model", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

    # --- LAPORAN KESEHATAN AIR ---
    st.markdown("---")
    st.subheader("📝 Laporan Kesehatan Air (Parameter Fisik):")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        if 6.5 <= ph <= 8.5:
            st.markdown("✅ **pH (Keasaman):** Normal/Netral (Aman).")
        else:
            st.markdown(f"⚠️ **pH (Keasaman):** Tidak Ideal ({ph}). Standar aman: 6.5 - 8.5.")
        
        if hardness < 300:
            st.markdown("✅ **Kekerasan (Hardness):** Lunak/Wajar (Aman).")
        else:
            st.markdown(f"⚠️ **Kekerasan (Hardness):** Tinggi ({hardness} mg/L). Berisiko menimbulkan kerak.")

    with col_rep2:
        if solids < 1000:
            st.markdown("✅ **Padatan Terlarut (TDS):** Rendah/Wajar.")
        else:
            st.markdown(f"⚠️ **Padatan Terlarut (TDS):** Tinggi ({solids} ppm). Air mungkin berasa asin/logam.")

        if turbidity < 5.0:
            st.markdown("✅ **Kekeruhan:** Jernih (Aman).")
        else:
            st.markdown(f"⚠️ **Kekeruhan:** Tinggi ({turbidity} NTU). Air tampak keruh/kotor.")

    # --- VISUALIZATION ---
    st.markdown("---")
    st.subheader("📊 Visualisasi Profil Sampel Air")
    st.caption("Grafik Biru = Data Sampel Air Anda  |  Grafik Hijau = Batas Wajar.")
    
    fig = plot_radar_chart(input_data[0])
    # PENTING: use_container_width=True agar grafik menyesuaikan lebar HP
    st.plotly_chart(fig, use_container_width=True)
    
    # --- DATA TABLE ---
    st.markdown("---")
    with st.expander("📥 Lihat Rincian Data Input Sampel Air Anda"):
        df_input = pd.DataFrame(input_data, columns=['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Org. Carbon', 'Trihalomethanes', 'Turbidity'])
        # PENTING: use_container_width=True agar tabel bisa discroll horizontal di HP
        st.dataframe(df_input, hide_index=True, use_container_width=True)

else:
    st.info("👈 Silakan masukkan data sampel air di sidebar sebelah kiri, lalu tekan tombol Cek Kelayakan Sampel Air")
