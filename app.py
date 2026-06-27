import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import os
import math
import base64
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Skin Lesion Reader",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
IMG_SIZE = 224
MODEL_PATH = "skin_cancer_final_model.h5"
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

TIER_COLORS = {
    'SAFE':    {"color": "#5F8262", "bg": "#E8EFE6", "label": "BENIGN · LOW RISK"},
    'CAUTION': {"color": "#C18A2E", "bg": "#F5EBD8", "label": "PRE-CANCEROUS · MONITOR"},
    'ALERT':   {"color": "#D85C3E", "bg": "#FBE7E1", "label": "MALIGNANT · URGENT"},
}

CLASS_INFO = {
    'akiec': {'full_name': 'Actinic Keratosis', 'tier': 'CAUTION',
              'description': 'A rough, sun-damaged patch. Can progress to squamous cell carcinoma if left untreated.',
              'advice': 'See a dermatologist for evaluation.'},
    'bcc':   {'full_name': 'Basal Cell Carcinoma', 'tier': 'ALERT',
              'description': 'The most common skin cancer. Often appears as a pearly or waxy bump.',
              'advice': 'Needs prompt evaluation and likely biopsy.'},
    'bkl':   {'full_name': 'Benign Keratosis', 'tier': 'SAFE',
              'description': 'A non-cancerous growth, often scaly or wart-like in texture.',
              'advice': 'Watch for changes in size, shape, or color.'},
    'df':    {'full_name': 'Dermatofibroma', 'tier': 'SAFE',
              'description': 'A firm, harmless skin nodule.',
              'advice': 'No treatment needed unless it becomes symptomatic.'},
    'mel':   {'full_name': 'Melanoma', 'tier': 'ALERT',
              'description': 'The most dangerous skin cancer. Can spread quickly if missed.',
              'advice': 'Seek a dermatologist or oncologist immediately.'},
    'nv':    {'full_name': 'Melanocytic Nevus (Mole)', 'tier': 'SAFE',
              'description': 'A common mole formed by pigment-producing cells.',
              'advice': 'Track using the ABCDE rule for moles.'},
    'vasc':  {'full_name': 'Vascular Lesion', 'tier': 'SAFE',
              'description': 'A lesion related to blood vessels, such as a hemangioma.',
              'advice': 'See a doctor if it grows or bleeds.'},
}

# ============================================================
# STYLE
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --porcelain: #FAF6EF;
  --paper: #FFFFFF;
  --ink: #1C2526;
  --ink-soft: #5B6566;
  --teal: #2D6A6A;
  --line: #DEDACE;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp { background-color: var(--porcelain); }
.block-container { max-width: 1080px; padding-top: 2.2rem; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 0.5rem;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 2.5rem;
  line-height: 1.08;
  color: var(--ink);
  margin: 0 0 0.7rem 0;
}
.hero-sub {
  font-size: 1.02rem;
  color: var(--ink-soft);
  max-width: 620px;
  margin: 0;
  line-height: 1.5;
}
.hairline { height: 1px; background: var(--line); margin: 1.5rem 0 1.7rem 0; }
.hairline-thin { height: 1px; background: var(--line); margin: 0.9rem 0; }

.notice-card {
  background: #F5EBD8;
  border-left: 3px solid #C18A2E;
  padding: 0.85rem 1.1rem;
  font-size: 0.88rem;
  color: var(--ink);
  border-radius: 4px;
  margin-bottom: 1.8rem;
  line-height: 1.5;
}
.notice-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: #C18A2E;
  margin-right: 0.5rem;
}

.lens-wrap { position: relative; width: 300px; height: 300px; margin: 0 auto; }
.lens-ring { position: absolute; top: 0; left: 0; }
.lens-photo {
  position: absolute; top: 24px; left: 24px;
  width: 252px; height: 252px;
  border-radius: 50%;
  background-size: cover;
  background-position: center;
  box-shadow: 0 0 0 1px var(--paper), 0 8px 20px rgba(28,37,38,0.18);
}
.lens-spec {
  text-align: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
  margin-top: 0.6rem;
}

.diag-card {
  background: var(--paper);
  border-left: 4px solid var(--ink);
  border-radius: 6px;
  padding: 1.5rem 1.7rem;
  box-shadow: 0 1px 3px rgba(28,37,38,0.07);
  animation: fadeInUp 0.5s ease;
  height: 100%;
}
.risk-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  padding: 0.28rem 0.65rem;
  border-radius: 3px;
  display: inline-block;
  margin-bottom: 0.8rem;
}
.diag-name {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.55rem;
  margin: 0 0 0.9rem 0;
  color: var(--ink);
}
.diag-confidence { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 1.1rem; }
.diag-confidence-num { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 600; color: var(--ink); }
.diag-confidence-unit { font-size: 1.1rem; }
.diag-confidence-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; color: var(--ink-soft); }
.diag-desc { color: var(--ink-soft); font-size: 0.94rem; line-height: 1.55; margin-bottom: 0.7rem; }
.diag-advice { font-size: 0.92rem; color: var(--ink); line-height: 1.5; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) { .diag-card { animation: none; } }

.readout-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  color: var(--ink-soft);
  margin: 0.4rem 0 0.9rem 0;
}
.bar-row { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.6rem; }
.bar-label { width: 190px; font-size: 0.86rem; color: var(--ink); flex-shrink: 0; }
.bar-track { flex: 1; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-pct { width: 56px; text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--ink-soft); flex-shrink: 0; }

.pair-row { display: flex; gap: 1rem; }
.pair-card { position: relative; flex: 1; border-radius: 6px; overflow: hidden; border: 1px solid var(--line); }
.pair-img { width: 100%; display: block; }
.pair-tag {
  position: absolute; bottom: 8px; left: 8px;
  background: rgba(28,37,38,0.78); color: #fff;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.08em;
  padding: 0.2rem 0.55rem; border-radius: 3px;
}

.empty-state { text-align: center; padding: 2.2rem 0 1rem 0; }
.empty-lens { width: 200px; height: 200px; border-radius: 50%; border: 2px dashed var(--line); margin: 0 auto 1.1rem auto; }
.empty-text { color: var(--ink-soft); font-family: 'IBM Plex Mono', monospace; font-size: 0.84rem; letter-spacing: 0.04em; }

.side-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; color: var(--ink-soft); margin: 0.5rem 0 0.6rem 0; }
.spec-row { display: flex; justify-content: space-between; font-size: 0.82rem; padding: 0.2rem 0; color: var(--ink); }
.catalog-row { display: flex; align-items: center; gap: 0.55rem; padding: 0.25rem 0; font-size: 0.82rem; color: var(--ink); }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

.error-card {
  background: #FBE7E1; border-left: 3px solid #D85C3E;
  padding: 1rem 1.2rem; border-radius: 4px; font-size: 0.9rem; line-height: 1.5;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def make_ring_svg(diameter=300, ticks=36, color="#2D6A6A"):
    r_outer = diameter / 2
    r_major = r_outer - 10
    r_minor = r_outer - 6
    cx = cy = r_outer
    parts = []
    for i in range(ticks):
        angle = (2 * math.pi / ticks) * i
        major = (i % 3 == 0)
        r_in = r_major if major else r_minor
        x1 = cx + r_in * math.cos(angle)
        y1 = cy + r_in * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle)
        y2 = cy + r_outer * math.sin(angle)
        w = 1.6 if major else 1.0
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-opacity="0.5"/>')
    circle = f'<circle cx="{cx}" cy="{cy}" r="{r_outer-1}" fill="none" stroke="{color}" stroke-width="1" stroke-opacity="0.3"/>'
    return f'<svg width="{diameter}" height="{diameter}">{circle}{"".join(parts)}</svg>'


def pil_to_base64_square(img, size=320):
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def array_to_base64(arr):
    img = Image.fromarray(arr.astype('uint8'))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for l in reversed(layer.layers):
                if isinstance(l, keras.layers.Conv2D):
                    return l.name
    return None


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(255 * img)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<p class="eyebrow">DERMOSCOPIC ANALYSIS &middot; NTCC PROJECT 2026</p>
<h1 class="hero-title">Skin Lesion Reader</h1>
<p class="hero-sub">Upload a dermoscopic photo of a skin lesion. The model classifies it across seven categories and shows exactly where it looked to make that call.</p>
<div class="hairline"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-card">
  <span class="notice-tag">NOTE</span>This is a research prototype, not a certified diagnostic tool. Always consult a dermatologist for an actual diagnosis.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="side-title">MODEL SPEC</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="spec-row"><span>Model</span><span>EfficientNetB3</span></div>
    <div class="spec-row"><span>Dataset</span><span>HAM10000</span></div>
    <div class="spec-row"><span>Images</span><span>10,015</span></div>
    <div class="spec-row"><span>Test accuracy</span><span>~85&ndash;88%</span></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hairline-thin"></div>', unsafe_allow_html=True)
    st.markdown('<p class="side-title">REFERENCE CATALOG</p>', unsafe_allow_html=True)

    tier_order = {'ALERT': 0, 'CAUTION': 1, 'SAFE': 2}
    sorted_classes = sorted(CLASS_NAMES, key=lambda c: tier_order[CLASS_INFO[c]['tier']])
    rows = ""
    for code in sorted_classes:
        info = CLASS_INFO[code]
        color = TIER_COLORS[info['tier']]['color']
        rows += f'<div class="catalog-row"><span class="dot" style="background:{color};"></span><span>{info["full_name"]}</span></div>'
    st.markdown(rows, unsafe_allow_html=True)

# ============================================================
# MAIN — UPLOAD + RESULTS
# ============================================================
if not os.path.exists(MODEL_PATH):
    st.markdown(f"""
    <div class="error-card">
      <strong>Model file not found.</strong><br/>
      Place <code>{MODEL_PATH}</code> in this same folder, next to <code>app.py</code>, then restart the app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Reading the image..."):
        model = load_model()
        img_array = preprocess_image(image)
        predictions = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(predictions[0]))
        pred_class = CLASS_NAMES[pred_index]
        confidence = float(predictions[0][pred_index] * 100)
        info = CLASS_INFO[pred_class]
        tier = TIER_COLORS[info['tier']]

    col_lens, col_diag = st.columns([0.85, 1.15], gap="large")

    with col_lens:
        photo_b64 = pil_to_base64_square(image, size=252)
        ring_svg = make_ring_svg(diameter=300, color="#2D6A6A")
        st.markdown(f"""
        <div class="lens-wrap">
          <div class="lens-ring">{ring_svg}</div>
          <div class="lens-photo" style="background-image:url('data:image/png;base64,{photo_b64}');"></div>
        </div>
        <p class="lens-spec">224 &times; 224 &middot; NORMALIZED INPUT</p>
        """, unsafe_allow_html=True)

    with col_diag:
        st.markdown(f"""
        <div class="diag-card" style="border-left-color:{tier['color']};">
          <span class="risk-badge" style="background:{tier['bg']}; color:{tier['color']};">{tier['label']}</span>
          <h2 class="diag-name">{info['full_name']}</h2>
          <div class="diag-confidence">
            <span class="diag-confidence-num">{confidence:.1f}<span class="diag-confidence-unit">%</span></span>
            <span class="diag-confidence-label">CONFIDENCE</span>
          </div>
          <p class="diag-desc">{info['description']}</p>
          <p class="diag-advice"><strong>Next step &mdash;</strong> {info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="readout-title">FULL READOUT &middot; ALL 7 CLASSES</p>', unsafe_allow_html=True)
    order = np.argsort(predictions[0])[::-1]
    bars = ""
    for idx in order:
        code = CLASS_NAMES[idx]
        c_info = CLASS_INFO[code]
        pct = float(predictions[0][idx] * 100)
        color = TIER_COLORS[c_info['tier']]['color']
        bars += f"""
        <div class="bar-row">
          <span class="bar-label">{c_info['full_name']}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>"""
    st.markdown(bars, unsafe_allow_html=True)

    st.markdown('<p class="readout-title">SPECIMEN &middot; ATTENTION MAP</p>', unsafe_allow_html=True)
    try:
        last_conv = find_last_conv_layer(model)
        heatmap = get_gradcam_heatmap(model, img_array, last_conv, pred_index)
        original_resized = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
        overlay = overlay_gradcam(original_resized, heatmap)

        orig_b64 = array_to_base64((original_resized * 255).astype('uint8'))
        heat_b64 = array_to_base64(overlay)

        st.markdown(f"""
        <div class="pair-row">
          <div class="pair-card">
            <img src="data:image/png;base64,{orig_b64}" class="pair-img"/>
            <span class="pair-tag">SPECIMEN</span>
          </div>
          <div class="pair-card">
            <img src="data:image/png;base64,{heat_b64}" class="pair-img"/>
            <span class="pair-tag">ATTENTION MAP</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="error-card">Could not generate the attention map for this image. ({e})</div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-lens"></div>
      <p class="empty-text">UPLOAD A DERMOSCOPIC IMAGE TO RUN THE READ</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hairline-thin"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#5B6566; font-size:0.8rem;">NTCC In-House Practical Training &middot; B.Tech CSE &middot; Semester 5 &middot; 2026</p>', unsafe_allow_html=True)
