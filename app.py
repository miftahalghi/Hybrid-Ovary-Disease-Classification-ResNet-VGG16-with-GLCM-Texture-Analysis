import streamlit as st
import numpy as np
import cv2
import joblib
import os
import hashlib
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")


# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="OvarAI — Deteksi Kista Ovarium",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def download_models_from_hf():
    """Download model dari HF Hub jika belum ada di local."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("❌ Package huggingface_hub tidak terinstall. Tambahkan ke requirements.txt")
        st.stop()

    token   = st.secrets.get("HF_TOKEN", None)
    repo_id = st.secrets.get("HF_REPO", None)

    if not repo_id:
        st.error("❌ HF_REPO tidak ditemukan di Streamlit Secrets.")
        st.stop()

    files_needed = [
        "best_s1_vgg16_glcm.keras",
        "best_s2_resnet_glcm.keras",
        "glcm_scaler.pkl",
    ]

    progress = st.empty()
    for i, fname in enumerate(files_needed):
        dest = os.path.join(MODEL_DIR, fname)
        if os.path.exists(dest):
            continue  # sudah ada, skip
        progress.info(f"⏬ Mengunduh {fname} dari Hugging Face... ({i+1}/{len(files_needed)})")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=MODEL_DIR,
                token=token,
            )
        except Exception as e:
            st.error(f"❌ Gagal download {fname}: {e}")
            st.stop()

    progress.empty()
    return True

# Jalankan download saat startup
download_models_from_hf()

# ── CSS Custom ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --teal:     #0d9488;
    --teal-lt:  #ccfbf1;
    --teal-dk:  #0f766e;
    --cream:    #fdfcfb;
    --slate:    #1e293b;
    --muted:    #64748b;
    --border:   #e2e8f0;
    --danger:   #ef4444;
    --warn:     #f59e0b;
    --ok:       #10b981;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--slate);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #134e4a 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    right: -40px; top: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.hero::after {
    content: '';
    position: absolute;
    right: 60px; bottom: -80px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero p { margin: 0; opacity: 0.88; font-size: 1.05rem; font-weight: 300; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

/* ── Upload zone ── */
.upload-area {
    border: 2px dashed var(--teal);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
    background: var(--teal-lt);
    transition: all 0.2s;
}

/* ── Result card ── */
.result-card {
    background: white;
    border-radius: 14px;
    padding: 1.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid var(--border);
}
.result-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.result-card .diagnosis {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: var(--teal-dk);
    margin: 0 0 0.3rem;
    line-height: 1.2;
}
.result-card .confidence {
    font-size: 0.92rem;
    color: var(--muted);
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.55rem;
}
.prob-row .cls-name {
    width: 160px;
    font-size: 0.85rem;
    font-weight: 500;
    flex-shrink: 0;
    color: var(--slate);
}
.prob-bar-wrap {
    flex: 1;
    background: #f1f5f9;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--teal), var(--teal-dk));
    transition: width 0.8s ease;
}
.prob-bar.top { background: linear-gradient(90deg, #0d9488, #134e4a); }
.prob-pct {
    width: 44px;
    text-align: right;
    font-size: 0.83rem;
    font-weight: 600;
    color: var(--slate);
}

/* ── Info chips ── */
.chip {
    display: inline-block;
    background: var(--teal-lt);
    color: var(--teal-dk);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}
.chip.warn { background: #fef3c7; color: #92400e; }
.chip.ok   { background: #d1fae5; color: #065f46; }

/* ── Info box ── */
.info-box {
    background: #f8fafc;
    border-left: 3px solid var(--teal);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--slate);
    margin-top: 1rem;
}
.info-box strong { color: var(--teal-dk); }

/* ── Disclaimer ── */
.disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.84rem;
    color: #92400e;
    line-height: 1.6;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-size: 0.86rem;
}
.sidebar-section h4 {
    font-family: 'DM Serif Display', serif;
    color: var(--teal-dk);
    margin: 0 0 0.6rem;
    font-size: 1.05rem;
}

/* ── Metric pill ── */
.metric-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.84rem;
}
.metric-pill:last-child { border-bottom: none; }
.metric-pill .val { font-weight: 600; color: var(--teal-dk); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 3px; }

/* ── stImage caption ── */
.stImage > div { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "complex_cyst",
    "dominant_follicle",
    "healthy",
    "poly_cyst",
    "simple_cyst",
]

CLASS_INFO = {
    "complex_cyst": {
        "label": "Kista Kompleks",
        "desc": "Kista dengan komponen internal seperti septa, nodul, atau komponen padat. Memerlukan evaluasi lanjutan untuk menyingkirkan keganasan.",
        "severity": "warn",
        "icon": "⚠️",
        "action": "Konsultasikan segera dengan dokter spesialis kandungan.",
    },
    "dominant_follicle": {
        "label": "Folikel Dominan",
        "desc": "Folikel yang tumbuh lebih besar dari lainnya dalam siklus menstruasi normal, menandakan ovulasi yang akan terjadi.",
        "severity": "ok",
        "icon": "✅",
        "action": "Kondisi fisiologis normal. Tidak memerlukan tindakan khusus.",
    },
    "healthy": {
        "label": "Ovarium Sehat",
        "desc": "Tampilan ovarium dalam batas normal tanpa temuan patologis yang bermakna.",
        "severity": "ok",
        "icon": "✅",
        "action": "Lanjutkan pemeriksaan rutin sesuai jadwal.",
    },
    "poly_cyst": {
        "label": "Polikista (PCOS)",
        "desc": "Gambaran ovarium dengan banyak folikel kecil (≥12 folikel, diameter 2–9 mm) yang tersusun di tepi. Terkait dengan sindrom ovarium polikistik.",
        "severity": "warn",
        "icon": "⚠️",
        "action": "Evaluasi hormonal dan konsultasi dokter untuk penanganan PCOS.",
    },
    "simple_cyst": {
        "label": "Kista Sederhana",
        "desc": "Kista berisi cairan jernih dengan dinding tipis dan halus. Umumnya jinak dan sering menghilang sendiri.",
        "severity": "ok",
        "icon": "🔵",
        "action": "Pantau berkala (USG ulang 6–8 minggu). Konsultasi jika membesar.",
    },
}

IMG_SIZE       = 224
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GLCM_DIM       = len(GLCM_DISTANCES) * len(GLCM_ANGLES) * len(GLCM_PROPS)  # 48

# ══════════════════════════════════════════════════════════════════
# LOAD MODEL & SCALER
# ══════════════════════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner=False)
def load_resources(model_choice: str):
    """Load model keras + scaler — cached agar tidak reload tiap interaksi."""
    import tensorflow as tf
    from skimage.feature import graycomatrix, graycoprops  # noqa

    model_file = (
        "best_s1_vgg16_glcm.keras"
        if model_choice == "VGG16 + GLCM (Skema 1)"
        else "best_s2_resnet_glcm.keras"
    )
    model_path  = os.path.join(MODEL_DIR, model_file)
    scaler_path = os.path.join(MODEL_DIR, "glcm_scaler.pkl")

    if not os.path.exists(model_path):
        return None, None, f"Model tidak ditemukan: {model_path}"
    if not os.path.exists(scaler_path):
        return None, None, f"Scaler tidak ditemukan: {scaler_path}"

    try:
        model  = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def preprocess_image(img_array: np.ndarray, backbone: str = "vgg16") -> np.ndarray:
    """CLAHE + denoise + resize + backbone normalisasi."""
    import tensorflow as tf

    gray     = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    resized  = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img_3ch  = np.stack([resized] * 3, axis=-1).astype(np.float32)

    if backbone == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input(img_3ch)
    else:
        return tf.keras.applications.resnet_v2.preprocess_input(img_3ch)


def extract_glcm(img_array: np.ndarray) -> np.ndarray:
    """Ekstraksi 48 fitur GLCM dari gambar."""
    from skimage.feature import graycomatrix, graycoprops

    gray     = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    resized  = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img_q    = (resized // 8).astype(np.uint8)
    glcm     = graycomatrix(
        img_q,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=32,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        feats.extend(graycoprops(glcm, prop).flatten().tolist())
    return np.array(feats, dtype=np.float32)


def predict(model, scaler, img_array: np.ndarray, backbone: str):
    """Jalankan inferensi dual-branch."""
    # CNN branch
    cnn_input = preprocess_image(img_array, backbone)[np.newaxis]  # (1,224,224,3)
    # GLCM branch
    glcm_raw    = extract_glcm(img_array)
    glcm_scaled = scaler.transform(glcm_raw.reshape(1, -1))        # (1,48)

    proba = model.predict(
        {"img_input": cnn_input, "glcm_input": glcm_scaled},
        verbose=0,
    )[0]
    return proba


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem">🔬</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#0f766e; font-weight:600;">OvarAI</div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">Sistem Deteksi Kista Ovarium</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Pilih Model**")
    model_choice = st.selectbox(
        "Model",
        ["VGG16 + GLCM (Skema 1)", "ResNet50V2 + GLCM (Skema 2)"],
        label_visibility="collapsed",
    )

    backbone = "vgg16" if "VGG16" in model_choice else "resnet"

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📊 Performa Model</h4>
        <div class="metric-pill"><span>Model</span><span class="val">Dual-Branch CNN</span></div>
        <div class="metric-pill"><span>Backbone</span><span class="val">VGG16 / ResNet50V2</span></div>
        <div class="metric-pill"><span>Fitur tekstur</span><span class="val">GLCM (48 fitur)</span></div>
        <div class="metric-pill"><span>Kelas</span><span class="val">5 kelas</span></div>
        <div class="metric-pill"><span>Input size</span><span class="val">224 × 224 px</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🏷️ Kelas yang Dideteksi</h4>
    </div>
    """, unsafe_allow_html=True)

    for cls, info in CLASS_INFO.items():
        st.markdown(
            f"<div class='chip {'ok' if info['severity']=='ok' else 'warn'}'>"
            f"{info['icon']} {info['label']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#94a3b8; text-align:center; line-height:1.6;">
        Dibuat untuk tujuan edukasi & penelitian.<br>
        Bukan pengganti diagnosis medis profesional.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN — HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">🔬 DUAL-BRANCH CNN + GLCM</div>
    <h1>OvarAI</h1>
    <p>Deteksi Kondisi Ovarium dari Citra USG menggunakan Deep Learning</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════
with st.spinner("Memuat model..."):
    model, scaler, err = load_resources(model_choice)

if err:
    st.error(f"❌ Gagal memuat model: {err}")
    st.markdown("""
    **Pastikan folder `models/` berisi:**
    - `best_s1_vgg16_glcm.keras`
    - `best_s2_resnet_glcm.keras`
    - `glcm_scaler.pkl`
    """)
    st.stop()

st.success(f"✅ Model **{model_choice}** berhasil dimuat", icon="🤖")


# ══════════════════════════════════════════════════════════════════
# UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
col_up, col_res = st.columns([1, 1.1], gap="large")

with col_up:
    st.markdown("### 📤 Upload Citra USG")
    st.markdown("""
    <div style="font-size:0.88rem; color:#64748b; margin-bottom:1rem;">
    Format: <b>JPG, JPEG, PNG</b> — ukuran disarankan minimal 224×224 px
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        img_pil   = Image.open(uploaded).convert("RGB")
        img_array = np.array(img_pil)

        st.image(img_pil, caption="Citra yang diunggah", use_container_width=True)

        st.markdown(f"""
        <div style="font-size:0.82rem; color:#64748b; margin-top:0.5rem;">
            📐 Resolusi: <b>{img_pil.width} × {img_pil.height} px</b> &nbsp;|&nbsp;
            📁 Ukuran: <b>{uploaded.size/1024:.1f} KB</b>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing preview
        with st.expander("👁️ Preview preprocessing"):
            gray_prev = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_prev = clahe_obj.apply(gray_prev)
            col_a, col_b = st.columns(2)
            col_a.image(gray_prev, caption="Grayscale", use_container_width=True, clamp=True)
            col_b.image(clahe_prev, caption="Setelah CLAHE", use_container_width=True, clamp=True)

with col_res:
    st.markdown("### 🧠 Hasil Analisis")

    if not uploaded:
        st.markdown("""
        <div style="
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            color: #94a3b8;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            <div style="font-size:1rem; font-weight:500;">Belum ada gambar</div>
            <div style="font-size:0.85rem; margin-top:0.4rem;">Upload citra USG untuk memulai analisis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        btn = st.button("🔬 Analisis Sekarang", type="primary", use_container_width=True)

        if btn:
            with st.spinner("Mengekstraksi fitur & menjalankan inferensi..."):
                try:
                    proba = predict(model, scaler, img_array, backbone)
                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")
                    st.stop()

            top_idx   = int(np.argmax(proba))
            top_cls   = CLASS_NAMES[top_idx]
            top_info  = CLASS_INFO[top_cls]
            top_conf  = float(proba[top_idx])

            # ── Hasil utama ──────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Diagnosis</div>
                <div class="diagnosis">{top_info['icon']} {top_info['label']}</div>
                <div class="confidence">Confidence: <b>{top_conf*100:.1f}%</b></div>
                <br>
                <div class="chip {'ok' if top_info['severity']=='ok' else 'warn'}">
                    {'✅ Normal / Jinak' if top_info['severity']=='ok' else '⚠️ Perlu Evaluasi'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Deskripsi & saran ─────────────────────────────
            st.markdown(f"""
            <div class="info-box">
                <strong>📋 Deskripsi:</strong><br>
                {top_info['desc']}
                <br><br>
                <strong>💡 Saran Tindak Lanjut:</strong><br>
                {top_info['action']}
            </div>
            """, unsafe_allow_html=True)

            # ── Probabilitas semua kelas ──────────────────────
            st.markdown("<br>**📊 Distribusi Probabilitas**", unsafe_allow_html=True)

            sorted_idx = np.argsort(proba)[::-1]
            bars_html = ""
            for rank, i in enumerate(sorted_idx):
                cls  = CLASS_NAMES[i]
                pct  = float(proba[i]) * 100
                lbl  = CLASS_INFO[cls]["label"]
                is_top = i == top_idx
                bar_class = "prob-bar top" if is_top else "prob-bar"
                bars_html += f"""
                <div class="prob-row">
                    <div class="cls-name">{'<b>' if is_top else ''}{lbl}{'</b>' if is_top else ''}</div>
                    <div class="prob-bar-wrap">
                        <div class="{bar_class}" style="width:{pct:.1f}%"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

            # ── GLCM info ─────────────────────────────────────
            with st.expander("🧮 Detail Fitur GLCM"):
                glcm_raw = extract_glcm(img_array)
                glcm_df_data = {
                    "Properti": [],
                    "Min": [],
                    "Max": [],
                    "Mean": [],
                }
                for prop_idx, prop in enumerate(GLCM_PROPS):
                    start = prop_idx * len(GLCM_DISTANCES) * len(GLCM_ANGLES)
                    end   = start + len(GLCM_DISTANCES) * len(GLCM_ANGLES)
                    chunk = glcm_raw[start:end]
                    glcm_df_data["Properti"].append(prop.capitalize())
                    glcm_df_data["Min"].append(f"{chunk.min():.4f}")
                    glcm_df_data["Max"].append(f"{chunk.max():.4f}")
                    glcm_df_data["Mean"].append(f"{chunk.mean():.4f}")

                import pandas as pd
                st.dataframe(pd.DataFrame(glcm_df_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Peringatan Medis:</strong> Hasil analisis ini bersifat <strong>bantuan skrining awal</strong> berbasis kecerdasan buatan
    dan <strong>bukan merupakan diagnosis medis</strong>. Selalu konsultasikan hasil dengan dokter spesialis
    kandungan (SpOG) atau radiolog berpengalaman sebelum mengambil keputusan klinis.
    Sistem ini dibangun untuk tujuan <strong>penelitian dan edukasi</strong>.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ABOUT (expander bawah)
# ══════════════════════════════════════════════════════════════════
with st.expander("ℹ️ Tentang Sistem & Metodologi"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Arsitektur Model**
        - Dual-Branch Neural Network
        - Branch A: CNN (VGG16 / ResNet50V2) dengan ImageNet weights
        - Branch B: GLCM texture features (48 dimensi)
        - Fusion layer → klasifikasi 5 kelas

        **Training Pipeline**
        1. Transfer Learning (freeze backbone)
        2. Hyperparameter Tuning (Keras Tuner)
        3. Fine-tuning bertahap (partial unfreeze)
        """)
    with c2:
        st.markdown("""
        **Preprocessing per Gambar**
        1. Konversi ke grayscale
        2. Denoising (fastNlMeansDenoising)
        3. CLAHE (contrast enhancement)
        4. Resize ke 224×224
        5. Backbone normalization

        **Dataset**
        - [Ovarian Ultrasound Image Dataset](https://www.kaggle.com/datasets/ucimachinelearning/ovarian-ultrasound-image-dataset)
        - 5 kelas: complex cyst, dominant follicle, healthy, poly cyst, simple cyst
        """)
