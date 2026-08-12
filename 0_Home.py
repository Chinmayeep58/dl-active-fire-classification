import streamlit as st

st.set_page_config(
    page_title="Active-Fire Detection & Feedback Learning",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0D1B2A;
    color: #F0EDE8;
}

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2.5rem;
    text-align: center;
    position: relative;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    color: #E8622A;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #F0EDE8 30%, #E8622A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.2rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #9CA3AF;
    font-weight: 300;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Feature cards ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2.5rem 0;
}

.feature-card {
    background: #111F2E;
    border: 1px solid #1E3A2F;
    border-left: 3px solid #E8622A;
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
    font-size: 0.88rem;
    color: #C9C5BF;
    line-height: 1.6;
}

.feature-card .icon {
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #E8622A;
    text-transform: uppercase;
    margin: 3rem 0 1.2rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 600;
    color: #F0EDE8;
    margin-bottom: 1.5rem;
    letter-spacing: -0.01em;
}

/* ── Workflow blocks ── */
.workflow {
    margin-bottom: 2.5rem;
}

.workflow-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F0EDE8;
    margin-bottom: 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E3A2F;
}

.step {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.75rem;
    align-items: flex-start;
}

.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #E8622A;
    background: #1A120C;
    border: 1px solid #E8622A33;
    border-radius: 3px;
    padding: 0.2rem 0.45rem;
    white-space: nowrap;
    margin-top: 0.1rem;
    flex-shrink: 0;
}

.step-content {
    font-size: 0.88rem;
    color: #B0AAA3;
    line-height: 1.65;
}

.step-content strong {
    color: #D9D4CE;
    font-weight: 500;
}

.step-content code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: #1A2A1A;
    color: #6EE7B7;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
}

/* ── Workflow grid ── */
.workflow-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
}

/* ── Format table ── */
.format-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

.format-block {
    background: #111F2E;
    border: 1px solid #1E3A2F;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
}

.format-block h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #E8622A;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

.format-block ul {
    margin: 0;
    padding-left: 1.1rem;
    color: #9CA3AF;
    font-size: 0.85rem;
    line-height: 1.8;
}

.format-block ul li code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6EE7B7;
    background: #1A2A1A;
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
}

/* ── Divider ── */
.ember-divider {
    height: 1px;
    background: linear-gradient(to right, #E8622A22, #E8622A55, #E8622A22);
    margin: 2.5rem 0;
    border: none;
}

/* ── Hide Streamlit chrome ── */

.block-container {padding-top: 1rem; max-width: 1100px;}
</style>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Satellite · ML · Feedback Loop</div>
    <div class="hero-title">Active-Fire Detection &<br>Feedback Learning System</div>
    <div class="hero-sub">
        An end-to-end workflow for Active-Fire detection using satellite imagery
        and machine learning — from raw coordinates to validated predictions.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────
st.markdown("""
<div class="feature-grid">
    <div class="feature-card"><span class="icon">📂</span>Upload Active-Fire incident data via CSV or Shapefile</div>
    <div class="feature-card"><span class="icon">🛰</span>Auto-extract satellite image patches (GeoTIFFs)</div>
    <div class="feature-card"><span class="icon">🤖</span>Run deep learning model for Active-Fire prediction/classification</div>
    <div class="feature-card"><span class="icon">📊</span>Generate prediction results in CSV format</div>
    <div class="feature-card"><span class="icon">🔍</span>Review predictions through an interactive interface</div>
    <div class="feature-card"><span class="icon">🔄</span>Retrain the model using validated feedback data</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ember-divider"></div>', unsafe_allow_html=True)

# ── How to Use ────────────────────────────────────────
st.markdown('<div class="section-heading">— Documentation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How to Use the Application</div>', unsafe_allow_html=True)

# ── Data Extraction ──
st.markdown("""
<div class="workflow">
    <div class="workflow-title">📡 Data Extraction</div>
    <div class="step">
        <span class="step-num">01</span>
        <div class="step-content">
            <strong>Upload Input Data</strong> — Navigate to the <strong>ESRI Extraction or Sentinel-2 Extraction</strong> page.
            Upload either a CSV file containing latitude and longitude coordinates, or a Shapefile (.zip) containing Active-Fire locations.
            Make the dataset public on Kaggle, then enter the URL in the format <code>kaggle_username/dataset-slug</code>. 
            Click <strong>Download Dataset</strong>, and once downloaded click <strong>Start Extraction</strong>.
        </div>
    </div>
    <div class="step">
        <span class="step-num">02</span>
        <div class="step-content">
            <strong>Extract Satellite Imagery</strong> — The system retrieves satellite imagery for the uploaded locations.
            GeoTIFF image patches are generated and stored for further processing.
            Download the processed files by clicking <strong>Download Extracted Dataset</strong>.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Workflow grid ──
st.markdown('<div class="workflow-grid">', unsafe_allow_html=True)

st.markdown("""
<div class="workflow">
    <div class="workflow-title">🗺 Model Predictions & Map Visualization</div>
    <div class="step">
        <span class="step-num">01</span>
        <div class="step-content"><strong>Data Extraction</strong> — Follow and complete Step 1 from data extraction.</div>
    </div>
    <div class="step">
        <span class="step-num">02</span>
        <div class="step-content"><strong>Model Loading</strong> — Navigate to <strong>Model/Dataset loading and Labelling</strong>.
        Select binary/multiclass classification, enter public Kaggle model URLs, and click <strong>Load Models</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">03</span>
        <div class="step-content"><strong>Run Predictions</strong> — Navigate to <strong>Model Predictions and Map Plotting</strong>.
        Click <strong>Run Predictions</strong>. Download results via <strong>Download Predictions</strong>.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow">
    <div class="workflow-title">🏷 Review & Labelling</div>
    <div class="step">
        <span class="step-num">01</span>
        <div class="step-content"><strong>Model Loading</strong> — Navigate to <strong>Model/Dataset loading and Labelling</strong>,
        select classification type, enter model URLs, and click <strong>Load Models</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">02</span>
        <div class="step-content"><strong>Dataset Loading</strong> — Enter public Kaggle dataset URLs and click <strong>Load Datasets</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">03</span>
        <div class="step-content"><strong>Class Correction</strong> — Review ESRI and Sentinel-2 predictions with images.
        Correct classes using the provided buttons. Download via <strong>Download Labels CSV</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">04</span>
        <div class="step-content"><strong>Evaluation Metrics</strong> — Click <strong>View Evaluation Metrics</strong> to assess model performance
        and decide if feedback training is needed.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Feedback Training ──
st.markdown("""
<div class="workflow">
    <div class="workflow-title">🔁 Feedback Training</div>
    <div class="step">
        <span class="step-num">01</span>
        <div class="step-content"><strong>Model Loading</strong> — Navigate to <strong>Model/Dataset loading and Labelling</strong>,
        select classification type, enter model URLs, and click <strong>Load Models</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">02</span>
        <div class="step-content"><strong>Dataset Loading</strong> — Enter public Kaggle dataset URLs and click <strong>Load Datasets</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">03</span>
        <div class="step-content"><strong>Feedback Dataset Loading</strong> — Enter the public Kaggle URL of the feedback CSV and click <strong>Load Feedback CSV</strong>.</div>
    </div>
    <div class="step">
        <span class="step-num">04</span>
        <div class="step-content"><strong>Start Feedback Training</strong> — Click <strong>Start Feedback Training</strong> to retrain the models.
        Evaluation metrics are displayed on completion — compare performance before and after training.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ember-divider"></div>', unsafe_allow_html=True)

# ── Input Formats ──────────────────────────────────────
st.markdown('<div class="section-heading">— Reference</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Expected Input Formats</div>', unsafe_allow_html=True)

st.markdown("""
<div class="format-grid">
    <div class="format-block">
        <h4>CSV File</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin-bottom:0.6rem;">Required columns:</p>
        <ul>
            <li><code>latitude</code></li>
            <li><code>longitude</code></li>
        </ul>
    </div>
    <div class="format-block">
        <h4>Shapefile</h4>
        <p style="color:#9CA3AF; font-size:0.85rem; margin-bottom:0.6rem;">Upload as a <code>.zip</code> archive containing:</p>
        <ul>
            <li><code>.shp</code></li>
            <li><code>.shx</code></li>
            <li><code>.dbf</code></li>
            <li><code>.prj</code></li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)