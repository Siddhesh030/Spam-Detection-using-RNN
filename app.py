import os
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# APPLICATION CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AegisText AI | Neural Spam Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# GLOBAL STYLING & CYBER-SECURITY THEME
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #020617 100%);
        color: #f1f5f9;
    }

    /* Top Hero Banner */
    .hero-container {
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.35rem 0 0.15rem 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Verdict Badges */
    .verdict-box {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin: 1.25rem 0;
        animation: fadeIn 0.4s ease-in-out;
    }

    .verdict-spam {
        background: radial-gradient(circle at 50% 30%, rgba(239, 68, 68, 0.2) 0%, rgba(153, 27, 27, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
    }

    .verdict-ham {
        background: radial-gradient(circle at 50% 30%, rgba(16, 185, 129, 0.2) 0%, rgba(6, 78, 59, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }

    .verdict-chip {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Code / sequence chips */
    .token-chip {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #7dd3fc;
    }
    
    .token-oov {
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #64748b;
    }

    div[data-testid="stSidebar"] {
        background: #090d16;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# -----------------------------------------------------------------------------
# ROBUST ARTIFACT RESOLVER & CACHING
# -----------------------------------------------------------------------------
def locate_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner="Connecting to neural inference pipeline...")
def load_artifacts():
    model_path = locate_file(["spam_rnn.keras", "spam_rnn(2).keras"])
    tok_path = locate_file(["tokenizer.pkl", "tokenizer(4).pkl"])
    cfg_path = locate_file(["model_config.pkl", "model_config(1).pkl"])
    le_path = locate_file(["label_encoder.pkl", "label_encoder(1).pkl"])

    missing = []
    if not model_path: missing.append("spam_rnn.keras")
    if not tok_path: missing.append("tokenizer.pkl")
    if not cfg_path: missing.append("model_config.pkl")
    if not le_path: missing.append("label_encoder.pkl")

    if missing:
        return None, None, None, None, f"Missing critical artifacts: {', '.join(missing)}"

    try:
        model = load_model(model_path, compile=False)
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(cfg_path, "rb") as f:
            config = pickle.load(f)
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, config, label_encoder, None
    except Exception as err:
        return None, None, None, None, f"Artifact initialization error: {str(err)}"


model, tokenizer, model_config, label_encoder, error_msg = load_artifacts()

# -----------------------------------------------------------------------------
# INFERENCE LOGIC
# -----------------------------------------------------------------------------
def run_classification(text_input: str, custom_threshold: float = None):
    max_len = model_config.get("max_length", 100)
    padding_style = model_config.get("padding", "post")
    trunc_style = model_config.get("truncating", "post")
    default_thresh = float(model_config.get("threshold", 0.5))
    active_thresh = custom_threshold if custom_threshold is not None else default_thresh

    # Tokenizer sequence conversion
    raw_seq = tokenizer.texts_to_sequences([text_input])
    padded_seq = pad_sequences(
        raw_seq, maxlen=max_len, padding=padding_style, truncating=trunc_style
    )

    t_start = time.perf_counter()
    raw_pred = model.predict(padded_seq, verbose=0)
    latency_ms = (time.perf_counter() - t_start) * 1000

    shape = raw_pred.shape
    classes = list(label_encoder.classes_)

    if len(shape) == 2 and shape[1] == 1:
        # Binary output sigmoid
        spam_p = float(raw_pred[0][0])
        ham_p = 1.0 - spam_p
    elif len(shape) == 2 and shape[1] == 2:
        ham_idx = classes.index("ham") if "ham" in classes else 0
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_p = float(raw_pred[0][ham_idx])
        spam_p = float(raw_pred[0][spam_idx])
    else:
        raise ValueError(f"Unrecognized prediction tensor shape: {shape}")

    label = "Spam" if spam_p >= active_thresh else "Legitimate (Ham)"
    confidence = spam_p if label == "Spam" else ham_p

    return {
        "label": label,
        "spam_prob": spam_p,
        "ham_prob": ham_p,
        "confidence": confidence,
        "threshold": active_thresh,
        "latency_ms": latency_ms,
        "tokens": raw_seq[0],
        "padded_shape": padded_seq.shape,
    }

# -----------------------------------------------------------------------------
# TOP HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">AegisText AI // Deep Security</h1>
        <div class="hero-subtitle">
            Enterprise Sequential Deep Learning Architecture for Real-Time Text & Communications Verification
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if error_msg:
    st.error(f"⚠️ Security Pipeline Offline: {error_msg}")
    st.info("Ensure `spam_rnn.keras`, `tokenizer.pkl`, `model_config.pkl`, and `label_encoder.pkl` reside in the repository root.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Operations Hub")
    selected_module = st.radio(
        "Module Selection",
        [
            "Live Threat Scanner",
            "Batch Ingestion Auditor",
            "Deep Network Telemetry",
            "Model Registry & Specs",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Calibration Settings")
    default_th = float(model_config.get("threshold", 0.5))
    use_custom_th = st.toggle("Override Decision Boundary", value=False)
    
    if use_custom_th:
        active_threshold = st.slider(
            "Classification Cutoff ($\tau$)",
            min_value=0.05,
            max_value=0.95,
            value=default_th,
            step=0.01,
            help="Probabilities higher than this threshold trigger a SPAM classification.",
        )
    else:
        active_threshold = default_th
        st.caption(f"Fixed Model Threshold: `{default_th:.2f}`")

    st.markdown("---")
    st.markdown("### 📡 Session Live Metrics")
    total_audits = len(st.session_state.audit_log)
    detected_threats = sum(1 for x in st.session_state.audit_log if x["Verdict"] == "Spam")
    threat_rate = (detected_threats / total_audits * 100) if total_audits > 0 else 0.0

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; font-size: 0.85rem;">
            <div>Total Scanned: <b>{total_audits}</b></div>
            <div>Threats Flagged: <b style="color: #ef4444;">{detected_threats}</b></div>
            <div>Threat Ratio: <b>{threat_rate:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_audits > 0:
        if st.button("Purge Audit Memory", use_container_width=True):
            st.session_state.audit_log = []
            st.rerun()

# -----------------------------------------------------------------------------
# MODULE 1: LIVE THREAT SCANNER
# -----------------------------------------------------------------------------
if selected_module == "Live Threat Scanner":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Neural Backbone</div>
            <div class="kpi-value">{model_config.get('model_type', 'LSTM')}</div>
            <span style="color:#38bdf8; font-size:0.75rem;">Recurrent Cell Core</span></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Vocab Memory</div>
            <div class="kpi-value">{model_config.get('vocab_size', 8143):,}</div>
            <span style="color:#818cf8; font-size:0.75rem;">Indexed Word Features</span></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Max Token Buffer</div>
            <div class="kpi-value">{model_config.get('max_length', 100)}</div>
            <span style="color:#c084fc; font-size:0.75rem;">Padded Sequence Window</span></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Active Threshold</div>
            <div class="kpi-value">{active_threshold:.2f}</div>
            <span style="color:#34d399; font-size:0.75rem;">Sigmoid Cutoff Trigger</span></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_input, col_preset = st.columns([2.5, 1])

    with col_preset:
        st.markdown("#### Preset Intelligence")
        preset_choice = st.radio(
            "Select verification scenario:",
            [
                "Clear Terminal",
                "Benign: Project Update",
                "Phishing: Free Award Claim",
                "Fraud: Bank Account Alert",
                "Benign: Dinner Confirmation",
            ],
            index=0,
        )
        sample_map = {
            "Clear Terminal": "",
            "Benign: Project Update": "Hey team, the deliverables for the Q4 review are ready for review on GitHub.",
            "Phishing: Free Award Claim": "URGENT! You have won a 1,000 cash prize or a free holiday to Hawaii. Text CLAIM to 87077 now to secure your reward!",
            "Fraud: Bank Account Alert": "Alert from Bank Security: Your online access has been compromised. Verify your account at http://secure-login-update.com immediately.",
            "Benign: Dinner Confirmation": "Are we still meeting at 7:30 PM tonight for sushi? Let me know if you are running late.",
        }

    with col_input:
        st.markdown("#### Message Payload")
        message_body = st.text_area(
            "Input raw communication vector (SMS, email excerpt, or chat payload):",
            value=sample_map[preset_choice],
            height=130,
            placeholder="Type or paste incoming text stream for neural sequence evaluation...",
        )
        
        col_exec1, col_exec2 = st.columns([1, 4])
        with col_exec1:
            execute_audit = st.button("⚡ Run Threat Audit", type="primary", use_container_width=True)
        with col_exec2:
            if st.button("Reset Input"):
                st.rerun()

    if execute_audit:
        if not message_body.strip():
            st.warning("⚠️ Text stream is empty. Input content to evaluate.")
        else:
            with st.spinner("Tokenizing vector, evaluating hidden states..."):
                result = run_classification(message_body.strip(), active_threshold)
                
                # Append to session history
                st.session_state.audit_log.append(
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Message": message_body.strip(),
                        "Verdict": "Spam" if "Spam" in result["label"] else "Ham",
                        "Spam Probability (%)": round(result["spam_prob"] * 100, 2),
                        "Ham Probability (%)": round(result["ham_prob"] * 100, 2),
                        "Confidence (%)": round(result["confidence"] * 100, 2),
                        "Latency (ms)": round(result["latency_ms"], 2),
                    }
                )

            # Display Verdict Banner
            is_spam = "Spam" in result["label"]
            css_class = "verdict-spam" if is_spam else "verdict-ham"
            verdict_text = "🚨 THREAT DETECTED: SPAM / MALICIOUS" if is_spam else "✅ VERIFIED CLEAN: LEGITIMATE COMMUNICATION"

            st.markdown(
                f"""
                <div class="verdict-box {css_class}">
                    <span class="verdict-chip">{verdict_text}</span>
                    <span>Spam Probability: <b>{result['spam_prob']*100:.2f}%</b> &nbsp;|&nbsp; 
                          Legitimate Probability: <b>{result['ham_prob']*100:.2f}%</b> &nbsp;|&nbsp; 
                          Inference Latency: <b>{result['latency_ms']:.1f}ms</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Visual Telemetry Breakdown
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=["Ham (Legitimate)", "Spam (Malicious)"],
                            y=[result["ham_prob"] * 100, result["spam_prob"] * 100],
                            marker_color=["#10b981", "#ef4444"],
                            text=[f"{result['ham_prob']*100:.1f}%", f"{result['spam_prob']*100:.1f}%"],
                            textposition="auto",
                        )
                    ]
                )
                fig_bar.update_layout(
                    title="Class Probability Distribution",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font=dict(color="#94a3b8"),
                    height=280,
                    margin=dict(l=30, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["confidence"] * 100,
                        number={"suffix": "%", "font": {"color": "#f8fafc", "size": 32}},
                        title={"text": "Decision Confidence", "font": {"color": "#94a3b8"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": "#818cf8"},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                                {"range": [50, 80], "color": "rgba(245, 158, 11, 0.15)"},
                                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "#ef4444", "width": 3},
                                "thickness": 0.75,
                                "value": active_threshold * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=30, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Sequence Token Inspection
            with st.expander("🔍 Sequence Tokenization Inspection (Input to LSTM)"):
                words = message_body.strip().split()
                st.write("**Extracted Tokens from Vocabulary Registry:**")
                token_html = ""
                for w in words:
                    clean_w = w.lower().strip(".,!?:;\"'()[]{}")
                    tok_id = tokenizer.word_index.get(clean_w, None)
                    if tok_id is not None and tok_id < model_config.get("vocab_size", 8143):
                        token_html += f'<span class="token-chip">{clean_w} (ID: {tok_id})</span> '
                    else:
                        token_html += f'<span class="token-chip token-oov">{clean_w} (OOV)</span> '
                st.markdown(token_html, unsafe_allow_html=True)
                st.caption("Padded Matrix Vector Length: 100 features | Mask Zero: True")

# -----------------------------------------------------------------------------
# MODULE 2: BATCH INGESTION AUDITOR
# -----------------------------------------------------------------------------
elif selected_module == "Batch Ingestion Auditor":
    st.markdown("### 📁 Batch Stream Ingestion & Offline Auditing")
    st.caption("Upload a `.csv` file containing text logs to perform high-throughput neural classification.")

    uploaded_file = st.file_uploader("Select CSV Source (UTF-8)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded dataset preview ({len(df_upload)} records):")
            st.dataframe(df_upload.head(4), use_container_width=True)

            text_columns = [col for col in df_upload.columns if df_upload[col].dtype == "object"]
            if not text_columns:
                st.error("No compatible text columns detected in the uploaded file.")
            else:
                target_col = st.selectbox("Select target text column to audit:", text_columns)

                if st.button("🚀 Execute Batch Neural Scan", type="primary"):
                    progress_bar = st.progress(0, text="Initializing batch tensorization...")
                    results = []
                    
                    total = len(df_upload)
                    start_batch = time.perf_counter()

                    # Batch pre-processing for speed
                    texts = df_upload[target_col].astype(str).tolist()
                    max_len = model_config.get("max_length", 100)
                    pad_style = model_config.get("padding", "post")
                    trunc_style = model_config.get("truncating", "post")

                    sequences = tokenizer.texts_to_sequences(texts)
                    padded = pad_sequences(sequences, maxlen=max_len, padding=pad_style, truncating=trunc_style)

                    progress_bar.progress(40, text="Evaluating RNN sequences in tensor engine...")
                    raw_predictions = model.predict(padded, batch_size=64, verbose=0)
                    progress_bar.progress(85, text="Aggregating classification analytics...")

                    for i in range(total):
                        sp = float(raw_predictions[i][0])
                        verdict = "Spam" if sp >= active_threshold else "Ham"
                        results.append({
                            "Spam_Probability": round(sp, 4),
                            "Verdict": verdict
                        })

                    duration = time.perf_counter() - start_batch
                    progress_bar.progress(100, text=f"Audit complete in {duration:.2f}s ({total/duration:.1f} msgs/sec)")

                    df_results = df_upload.copy()
                    df_results["Aegis_Spam_Probability"] = [r["Spam_Probability"] for r in results]
                    df_results["Aegis_Verdict"] = [r["Verdict"] for r in results]

                    st.markdown("---")
                    st.markdown("#### Audit Discovery Summary")
                    
                    spam_count = sum(1 for r in results if r["Verdict"] == "Spam")
                    ham_count = total - spam_count
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Total Messages Audited", total)
                    b_col2.metric("Flagged as Spam", spam_count, delta=f"{spam_count/total*100:.1f}%")
                    b_col3.metric("Verified Clean", ham_count)

                    st.dataframe(df_results, use_container_width=True)

                    csv_export = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Annotated Audit CSV",
                        data=csv_export,
                        file_name=f"aegistext_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"Failed to process CSV: {str(e)}")

# -----------------------------------------------------------------------------
# MODULE 3: DEEP NETWORK TELEMETRY
# -----------------------------------------------------------------------------
elif selected_module == "Deep Network Telemetry":
    st.markdown("### 📊 Operational Telemetry & Threat Trends")

    if not st.session_state.audit_log:
        st.info("No runtime audits logged in this session yet. Run single messages or batch scans to observe telemetry.")
    else:
        df_log = pd.DataFrame(st.session_state.audit_log)

        t1, t2 = st.columns(2)
        with t1:
            class_counts = df_log["Verdict"].value_counts().reset_index()
            class_counts.columns = ["Verdict", "Count"]
            fig_pie = px.pie(
                class_counts,
                values="Count",
                names="Verdict",
                hole=0.55,
                color="Verdict",
                color_discrete_map={"Spam": "#ef4444", "Ham": "#10b981"},
                title="Active Session Classification Ratio",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with t2:
            fig_scat = px.scatter(
                df_log.reset_index(),
                x="index",
                y="Spam Probability (%)",
                color="Verdict",
                size="Confidence (%)",
                color_discrete_map={"Spam": "#ef4444", "Ham": "#10b981"},
                title="Temporal Stream Threat Dispersion",
                labels={"index": "Event Order"},
            )
            fig_scat.add_hline(y=active_threshold * 100, line_dash="dot", line_color="#f59e0b", annotation_text="Cutoff")
            fig_scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("#### Session Audit Journal")
        st.dataframe(df_log, use_container_width=True)

        col_d1, _ = st.columns([1, 4])
        with col_d1:
            csv_data = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Session Log",
                data=csv_data,
                file_name=f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# -----------------------------------------------------------------------------
# MODULE 4: MODEL REGISTRY & SPECS
# -----------------------------------------------------------------------------
elif selected_module == "Model Registry & Specs":
    st.markdown("### 🏛️ Model Registry & Architectural Specifications")

    col_spec1, col_spec2 = st.columns(2)

    with col_spec1:
        st.markdown("#### Topology Breakdown")
        st.markdown(
            """
            * **Input Layer**: Text sequence padded to fixed tensor dimension `(None, 100)`
            * **Embedding Layer**: 8,143 input vocabulary dimension $\\to$ 64-dimensional dense projection vector (`mask_zero=True`)
            * **LSTM Recurrent Cell**: 64 recurrent memory units with tanh activation & sigmoid gates
            * **Dropout Regularization**: 30% dropout probability for anti-overfitting
            * **Dense Feature Fusion**: 32 rectified linear units (ReLU)
            * **Final Classifier**: Single scalar neuron with Sigmoid activation $[0, 1]$
            """
        )

    with col_spec2:
        st.markdown("#### Parameter Configuration Snapshot")
        cfg_rows = [{"Hyperparameter": k, "Configured Value": str(v)} for k, v in model_config.items()]
        st.table(pd.DataFrame(cfg_rows))

    st.markdown("---")
    st.markdown("#### Class Label Indexing")
    le_data = [
        {"Target Index": i, "Raw Class Label": label, "Security State": "Clean (Ham)" if label == "ham" else "Threat (Spam)"}
        for i, label in enumerate(label_encoder.classes_)
    ]
    st.table(pd.DataFrame(le_data))
