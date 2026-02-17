import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Cek Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

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
# 4. FUNGSI VISUALISASI (RADAR CHART)
# ==========================================
def plot_radar_chart(input_values):
    categories = [
        'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    values_user = list(input_values)
    # Batas Visual (Acuan Grafik Hijau)
    values_limit = [8.5, 300, 25000, 10, 400, 600, 20, 100, 5] 
    
    val_norm_user = []
    for v, l in zip(values_user, values_limit):
        if l == 0: l = 1
        val_norm_user.append(v / l)

    val_norm_limit = [1.0] * 9 

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=val_norm_limit, theta=categories, fill='toself', name='Batas Aman', line_color='green', opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=val_norm_user, theta=categories, fill='toself', name='Sampel Anda', line_color='blue', opacity=0.6))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2.0])),
        showlegend=True, height=400, margin=dict(l=50, r=50, t=20, b=20)
    )
    return fig

# ==========================================
# 5. HALAMAN UTAMA
# ==========================================
st.title("💧 Aplikasi Cek Kelayakan Air Minum")
st.markdown("Aplikasi ini menggunakan **Algoritma pemodelan Random Forest** untuk memprediksi layak atau tidaknya air untuk diminum. Silakan masukkan data sampel air hasil uji lab di sidebar sebelah kiri.")

# --- BAGIAN EDUKASI (FULL 9 PARAMETER) ---
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
    
    # Layout kolom untuk laporan teks
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        # 1. Cek pH
        if 6.5 <= ph <= 8.5:
            st.markdown("✅ **pH (Keasaman):** Normal/Netral (Aman).")
        else:
            st.markdown(f"⚠️ **pH (Keasaman):** Tidak Ideal ({ph}). Standar aman: 6.5 - 8.5.")
        
        # 2. Cek Hardness (Kekerasan)
        if hardness < 300:
            st.markdown("✅ **Kekerasan (Hardness):** Lunak/Wajar (Aman).")
        else:
            st.markdown(f"⚠️ **Kekerasan (Hardness):** Tinggi ({hardness} mg/L). Berisiko menimbulkan kerak.")

    with col_rep2:
        # 3. Cek Solids / TDS (Padatan Terlarut)
        # Catatan: Standar ketat biasanya 500, tapi 1000 masih bisa ditoleransi.
        if solids < 1000:
            st.markdown("✅ **Padatan Terlarut (TDS):** Rendah/Wajar.")
        else:
            st.markdown(f"⚠️ **Padatan Terlarut (TDS):** Tinggi ({solids} ppm). Air mungkin berasa asin/logam.")

        # 4. Cek Turbidity (Kekeruhan)
        if turbidity < 5.0:
            st.markdown("✅ **Kekeruhan:** Jernih (Aman).")
        else:
            st.markdown(f"⚠️ **Kekeruhan:** Tinggi ({turbidity} NTU). Air tampak keruh/kotor.")

    # --- VISUALIZATION ---
    st.markdown("---")
    st.subheader("📊 Visualisasi Profil Sampel Air")
    st.caption("Grafik Biru = Data Sampel Air Anda  |  Grafik Hijau = Batas Wajar.")
    fig = plot_radar_chart(input_data[0])
    st.plotly_chart(fig, use_container_width=True)
    
    # --- DATA TABLE ---
    st.markdown("---")
    with st.expander("📥 Lihat Rincian Data Input Sampel Air Anda"):
        df_input = pd.DataFrame(input_data, columns=['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Org. Carbon', 'Trihalomethanes', 'Turbidity'])
        st.dataframe(df_input, hide_index=True)

else:
    st.info("👈 Silakan masukkan data sampel air di sidebar sebelah kiri, lalu tekan tombol Cek Kelayakan Sampel Air")