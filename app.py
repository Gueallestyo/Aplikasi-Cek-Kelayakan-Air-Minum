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

# --- CSS INJECTION: Agar Tampilan Rapi di HP ---
st.markdown("""
    <style>
        /* Mengurangi padding berlebih di HP */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        /* Judul Responsif */
        h1 { font-size: 1.8rem !important; }
        
        /* Merapikan Text List agar tidak terlalu renggang */
        ul { margin-bottom: 0.5rem !important; }
        li { font-size: 0.95rem !important; }
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
    st.error("❌ Error: File model tidak ditemukan.")
    st.stop()

# ==========================================
# 3. SIDEBAR (INPUT)
# ==========================================
st.sidebar.header("📝 Panel Input Data Sampel")
st.sidebar.markdown("Masukkan data laboratorium:")

# Default Values
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
st.sidebar.markdown("**Parameter Tambahan:**")

chloramines = st.sidebar.number_input("4. Chloramines (ppm)", 0.0, 14.0, def_chloro)
sulfate = st.sidebar.number_input("5. Sulfate (mg/L)", 0.0, 500.0, def_sulfate)
conductivity = st.sidebar.number_input("6. Conductivity (μS/cm)", 0.0, 800.0, def_conduct)
organic_carbon = st.sidebar.number_input("7. Org. Carbon (ppm)", 0.0, 30.0, def_carbon)
trihalomethanes = st.sidebar.number_input("8. Trihalomethanes (μg/L)", 0.0, 125.0, def_trihalo)
turbidity = st.sidebar.number_input("9. Turbidity (NTU)", 0.0, 7.0, def_turbid)

# ==========================================
# 4. FUNGSI VISUALISASI (RESPONSIF MOBILE)
# ==========================================
def plot_radar_chart(input_values):
    categories = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Org. Carbon', 'Trihalo.', 'Turbidity']
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

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2.0], tickfont=dict(size=9)),
        ),
        showlegend=True,
        # LEGENDA DI BAWAH (Agar grafik tidak gepeng di HP)
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=30, b=50),
        height=450,
        font=dict(size=11)
    )
    return fig

# ==========================================
# 5. HALAMAN UTAMA
# ==========================================
st.title("💧 Cek Kelayakan Air Minum")
st.markdown("Prediksi keamanan air minum berbasis AI Random Forest.")

# --- BAGIAN EDUKASI (REVISI: FORMAT LIST AGAR RAPI DI HP) ---
st.info("ℹ️ **Panduan Standar Kualitas Air**")
with st.expander("📖 Klik untuk Membaca Kamus Parameter", expanded=False):
    st.markdown("""
    **1. pH (Derajat Keasaman)**
    * **Ideal:** 6.5 - 8.5
    * **Risiko:** < 6 (Korosif), > 8.5 (Pahit)

    **2. Hardness (Kekerasan)**
    * **Ideal:** < 300 mg/L
    * **Risiko:** Berkerak, sabun sulit berbusa

    **3. Solids (TDS)**
    * **Ideal:** < 500 ppm
    * **Risiko:** Air keruh, rasa asin/logam

    **4. Chloramines**
    * **Ideal:** < 4.0 ppm
    * **Risiko:** Bau menyengat, iritasi

    **5. Sulfate**
    * **Ideal:** < 250 mg/L
    * **Risiko:** Rasa pahit, efek pencahar (diare)

    **6. Conductivity**
    * **Ideal:** < 400 μS/cm
    * **Risiko:** Indikator mineral berlebih

    **7. Organic Carbon**
    * **Ideal:** < 2.0 ppm
    * **Risiko:** Indikator adanya bakteri berbahaya

    **8. Trihalomethanes**
    * **Ideal:** < 80 μg/L
    * **Risiko:** **Karsinogenik** (Pemicu Kanker)

    **9. Turbidity (Kekeruhan)**
    * **Ideal:** < 5.0 NTU
    * **Risiko:** Tempat bakteri bersembunyi
    """)

# ==========================================
# 6. LOGIKA & OUTPUT
# ==========================================
st.markdown("---")

if st.button("🔍 CEK KELAYAKAN SEKARANG", type="primary", use_container_width=True):
    
    with st.spinner('Menganalisis...'):
        time.sleep(1.0)
    
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    input_scaled = scaler.transform(input_data)
    prediksi = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[prediksi] * 100
    
    st.markdown("---")
    
    # --- HASIL ---
    col_status, col_meter = st.columns([1.5, 1])

    with col_status:
        st.subheader("Hasil Analisis:")
        if prediksi == 1:
            st.success("✅ **AIR LAYAK MINUM**")
            st.markdown("Prediksi AI: **AMAN** dikonsumsi.")
        else:
            st.error("⛔ **TIDAK LAYAK MINUM**")
            st.markdown("Prediksi AI: **BERBAHAYA** / Tidak Aman.")

    with col_meter:
        st.subheader("Keyakinan:")
        st.metric(label="Probabilitas", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

    # --- LAPORAN FISIK ---
    st.markdown("---")
    st.subheader("📝 Laporan Fisik:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 6.5 <= ph <= 8.5: st.markdown("✅ **pH:** Normal")
        else: st.markdown(f"⚠️ **pH:** {ph} (Tidak Ideal)")
            
        if hardness < 300: st.markdown("✅ **Hardness:** Aman")
        else: st.markdown(f"⚠️ **Hardness:** {hardness} (Keras)")

    with col2:
        if solids < 1000: st.markdown("✅ **TDS:** Wajar")
        else: st.markdown(f"⚠️ **TDS:** {solids} (Tinggi)")
            
        if turbidity < 5.0: st.markdown("✅ **Kekeruhan:** Jernih")
        else: st.markdown(f"⚠️ **Kekeruhan:** Keruh")

    # --- VISUALISASI ---
    st.markdown("---")
    st.subheader("📊 Visualisasi")
    
    fig = plot_radar_chart(input_data[0])
    st.plotly_chart(fig, use_container_width=True)
    
    # --- DATA TABLE ---
    st.markdown("---")
    with st.expander("📥 Rincian Input"):
        df_input = pd.DataFrame(input_data, columns=['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Org. Carbon', 'Trihalo.', 'Turbidity'])
        st.dataframe(df_input, hide_index=True, use_container_width=True)

else:
    st.info("👈 Masukkan data di menu samping (klik tanda panah > di pojok kiri atas).")
