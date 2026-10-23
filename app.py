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
    page_title="Spam Detection using RNN",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DYNAMIC MULTI-COLOR THEME DEFINITIONS
# -----------------------------------------------------------------------------
THEMES = {
    "Cyber Dark (Default)": {
        "bg_gradient": "radial-gradient(circle at 15% 15%, #0f172a 0%, #020617 100%)",
        "card_bg": "rgba(30, 41, 59, 0.55)",
        "card_border": "rgba(148, 163, 184, 0.15)",
        "primary": "#38bdf8",
        "primary_gradient": "linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%)",
        "accent": "#818cf8",
        "text": "#f1f5f9",
        "subtext": "#94a3b8",
        "sidebar_bg": "#090d16",
    },
    "Emerald Matrix": {
        "bg_gradient": "radial-gradient(circle at 15% 15%, #064e3b 0%, #022c22 100%)",
        "card_bg": "rgba(6, 78, 59, 0.45)",
        "card_border": "rgba(52, 211, 153, 0.2)",
        "primary": "#34d399",
        "primary_gradient": "linear-gradient(120deg, #34d399 0%, #10b981 50%, #059669 100%)",
        "accent": "#6ee7b7",
        "text": "#ecfdf5",
        "subtext": "#a7f3d0",
        "sidebar_bg": "#022019",
    },
    "Sunset Neon": {
        "bg_gradient": "radial-gradient(circle at 15% 15%, #431407 0%, #1c0a04 100%)",
        "card_bg": "rgba(67, 20, 7, 0.45)",
        "card_border": "rgba(251, 146, 60, 0.25)",
        "primary": "#fb923c",
        "primary_gradient": "linear-gradient(120deg, #f97316 0%, #fb923c 50%, #f43f5e 100%)",
        "accent": "#f43f5e",
        "text": "#fff7ed",
        "subtext": "#fed7aa",
        "sidebar_bg": "#140602",
    },
    "Royal Indigo": {
        "bg_gradient": "radial-gradient(circle at 15% 15%, #1e1b4b 0%, #0f0d2e 100%)",
        "card_bg": "rgba(49, 46, 129, 0.4)",
        "card_border": "rgba(165, 180, 252, 0.2)",
        "primary": "#818cf8",
        "primary_gradient": "linear-gradient(120deg, #818cf8 0%, #a855f7 50%, #ec4899 100%)",
        "accent": "#c084fc",
        "text": "#f5f3ff",
        "subtext": "#c7d2fe",
        "sidebar_bg": "#08061a",
    },
    "Slate Monolith": {
        "bg_gradient": "radial-gradient(circle at 15% 15%, #1e293b 0%, #0f172a 100%)",
        "card_bg": "rgba(51, 65, 85, 0.5)",
        "card_border": "rgba(203, 213, 225, 0.15)",
        "primary": "#cbd5e1",
        "primary_gradient": "linear-gradient(120deg, #94a3b8 0%, #cbd5e1 50%, #f8fafc 100%)",
        "accent": "#94a3b8",
        "text": "#f8fafc",
        "subtext": "#94a3b8",
        "sidebar_bg": "#0b1120",
    },
}

# -----------------------------------------------------------------------------
# SIDEBAR THEME SWITCHER & STATE
# -----------------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

with st.sidebar:
    st.markdown("### 🎨 Theme Customization")
    selected_theme_name = st.selectbox(
        "Select Color Palette",
        list(THEMES.keys()),
        index=0,
    )
    theme = THEMES[selected_theme_name]

# Apply chosen theme styles dynamically
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    .stApp {{
        background: {theme['bg_gradient']};
        color: {theme['text']};
    }}

    .hero-container {{
        padding: 1.75rem 2rem;
        background: {theme['card_bg']};
        border: 1px solid {theme['card_border']};
        border-radius: 18px;
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }}
    
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        background: {theme['primary_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }}

    .hero-subtitle {{
        color: {theme['subtext']};
        font-size: 0.95rem;
        margin-top: 0.35rem;
        font-weight: 400;
    }}

    .kpi-card {{
        background: {theme['card_bg']};
        border: 1px solid {theme['card_border']};
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        position: relative;
        overflow: hidden;
    }}
    
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 3px;
        background: {theme['primary_gradient']};
    }}

    .kpi-label {{
        color: {theme['subtext']};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }}

    .kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.35rem 0 0.15rem 0;
        font-family: 'JetBrains Mono', monospace;
    }}

    .verdict-box {{
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin: 1.25rem 0;
    }}

    .verdict-spam {{
        background: radial-gradient(circle at 50% 30%, rgba(239, 68, 68, 0.25) 0%, rgba(153, 27, 27, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }}

    .verdict-ham {{
        background: radial-gradient(circle at 50% 30%, rgba(16, 185, 129, 0.25) 0%, rgba(6, 78, 59, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
    }}

    .verdict-chip {{
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }}

    .token-chip {{
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid {theme['card_border']};
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: {theme['primary']};
    }}
    
    .token-oov {{
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #64748b;
    }}

    div[data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']};
        border-right: 1px solid {theme['card_border']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CACHED ARTIFACT LOADER
# -----------------------------------------------------------------------------
def locate_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource(show_spinner="Loading Recurrent Neural Network models and assets...")
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
        return None, None, None, None, f"Missing required artifacts: {', '.join(missing)}"

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
        return None, None, None, None, f"Failed to initialize artifacts: {str(err)}"

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
        spam_p = float(raw_pred[0][0])
        ham_p = 1.0 - spam_p
    elif len(shape) == 2 and shape[1] == 2:
        ham_idx = classes.index("ham") if "ham" in classes else 0
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_p = float(raw_pred[0][ham_idx])
        spam_p = float(raw_pred[0][spam_idx])
    else:
        raise ValueError(f"Unrecognized prediction shape: {shape}")

    label = "Spam" if spam_p >= active_thresh else "Not Spam"
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
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">Spam Detection using RNN</h1>
        <div class="hero-subtitle">
            Deep Learning Sequential Language Architecture for Real-Time Text Classification
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if error_msg:
    st.error(f"⚠️ Model Pipeline Offline: {error_msg}")
    st.info("Place your `spam_rnn.keras`, `tokenizer.pkl`, `model_config.pkl`, and `label_encoder.pkl` in the application root directory.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎛️ Navigation")
    selected_module = st.radio(
        "Go to",
        [
            "Message Classification",
            "Batch File Auditor",
            "Prediction Analytics",
            "Model Architecture & Specs",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Threshold Tuning")
    default_th = float(model_config.get("threshold", 0.5))
    use_custom_th = st.toggle("Custom Decision Boundary", value=False)
    
    if use_custom_th:
        active_threshold = st.slider(
            "Classification Cutoff (τ)",
            min_value=0.05,
            max_value=0.95,
            value=default_th,
            step=0.01,
            help="Probabilities higher than or equal to this threshold will be flagged as Spam.",
        )
    else:
        active_threshold = default_th
        st.caption(f"Default Model Threshold: `{default_th:.2f}`")

    st.markdown("---")
    st.markdown("### 📈 Live Session Totals")
    total_audits = len(st.session_state.audit_log)
    spam_count = sum(1 for x in st.session_state.audit_log if x["Verdict"] == "Spam")
    spam_pct = (spam_count / total_audits * 100) if total_audits > 0 else 0.0

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.4); padding: 12px; border-radius: 8px; font-size: 0.85rem; border: 1px solid rgba(148, 163, 184, 0.15);">
            <div>Processed: <b>{total_audits}</b></div>
            <div>Spam Detected: <b style="color: #ef4444;">{spam_count}</b></div>
            <div>Spam Rate: <b>{spam_pct:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_audits > 0:
        if st.button("Clear Prediction Logs", use_container_width=True):
            st.session_state.audit_log = []
            st.rerun()

# -----------------------------------------------------------------------------
# MODULE 1: MESSAGE CLASSIFICATION
# -----------------------------------------------------------------------------
if selected_module == "Message Classification":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Neural Architecture</div>
            <div class="kpi-value">{model_config.get('model_type', 'LSTM')}</div>
            <span style="color:{theme['primary']}; font-size:0.75rem;">Recurrent Cell Core</span></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Vocabulary Size</div>
            <div class="kpi-value">{model_config.get('vocab_size', 8143):,}</div>
            <span style="color:{theme['accent']}; font-size:0.75rem;">Indexed Tokens</span></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Sequence Limit</div>
            <div class="kpi-value">{model_config.get('max_length', 100)}</div>
            <span style="color:{theme['primary']}; font-size:0.75rem;">Tokens Max Length</span></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Decision Boundary</div>
            <div class="kpi-value">{active_threshold:.2f}</div>
            <span style="color:{theme['accent']}; font-size:0.75rem;">Sigmoid Cutoff Trigger</span></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_input, col_preset = st.columns([2.6, 1.2])

    with col_preset:
        st.markdown("#### Sample Scenarios")
        preset_choice = st.radio(
            "Quick fill message:",
            [
                "Custom Empty",
                "Ham: Meeting Inquiry",
                "Spam: Free Prize Notification",
                "Spam: Urgent Bank Action",
                "Ham: Running Late",
            ],
            index=0,
        )
        sample_map = {
            "Custom Empty": "",
            "Ham: Meeting Inquiry": "Hey, are we still meeting tomorrow afternoon at 4 PM?",
            "Spam: Free Prize Notification": "Congratulations! You have won a 1000 cash prize or a free holiday to Hawaii. Text CLAIM to 87077 to secure your reward!",
            "Spam: Urgent Bank Action": "URGENT: Your bank account has been locked due to suspicious activity. Verify credentials at http://update-secure-login.com to restore access.",
            "Ham: Running Late": "I'm stuck in traffic near downtown, will be around 10 minutes late.",
        }

    with col_input:
        st.markdown("#### Message Payload")
        message_body = st.text_area(
            "Enter text or SMS content to analyze:",
            value=sample_map[preset_choice],
            height=140,
            placeholder="Type or paste the message you want to verify...",
        )
        
        col_exec1, col_exec2 = st.columns([1.2, 4])
        with col_exec1:
            execute_audit = st.button("Classify Message", type="primary", use_container_width=True)
        with col_exec2:
            if st.button("Reset Input"):
                st.rerun()

    if execute_audit:
        if not message_body.strip():
            st.warning("Please type or select a message before evaluating.")
        else:
            with st.spinner("Processing text and executing RNN inference..."):
                result = run_classification(message_body.strip(), active_threshold)
                
                st.session_state.audit_log.append(
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Message": message_body.strip(),
                        "Verdict": result["label"],
                        "Spam Probability (%)": round(result["spam_prob"] * 100, 2),
                        "Ham Probability (%)": round(result["ham_prob"] * 100, 2),
                        "Confidence (%)": round(result["confidence"] * 100, 2),
                        "Latency (ms)": round(result["latency_ms"], 2),
                    }
                )

            is_spam = (result["label"] == "Spam")
            css_class = "verdict-spam" if is_spam else "verdict-ham"
            verdict_text = "🚨 CLASSIFIED AS SPAM" if is_spam else "✅ CLASSIFIED AS NOT SPAM (HAM)"

            st.markdown(
                f"""
                <div class="verdict-box {css_class}">
                    <span class="verdict-chip">{verdict_text}</span>
                    <span>Spam Probability: <b>{result['spam_prob']*100:.2f}%</b> &nbsp;|&nbsp; 
                          Not Spam Probability: <b>{result['ham_prob']*100:.2f}%</b> &nbsp;|&nbsp; 
                          Inference Time: <b>{result['latency_ms']:.1f}ms</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=["Not Spam (Ham)", "Spam"],
                            y=[result["ham_prob"] * 100, result["spam_prob"] * 100],
                            marker_color=["#10b981", "#ef4444"],
                            text=[f"{result['ham_prob']*100:.1f}%", f"{result['spam_prob']*100:.1f}%"],
                            textposition="auto",
                        )
                    ]
                )
                fig_bar.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font=dict(color=theme["subtext"]),
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
                        title={"text": "Classification Confidence", "font": {"color": theme["subtext"]}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": theme["subtext"]},
                            "bar": {"color": theme["primary"]},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.2)"},
                                {"range": [50, 75], "color": "rgba(245, 158, 11, 0.2)"},
                                {"range": [75, 100], "color": "rgba(16, 185, 129, 0.2)"},
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

            with st.expander("🔍 Token Breakdown (Mapped Indices)"):
                words = message_body.strip().split()
                token_html = ""
                for w in words:
                    clean_w = w.lower().strip(".,!?:;\"'()[]{}")
                    tok_id = tokenizer.word_index.get(clean_w, None)
                    if tok_id is not None and tok_id < model_config.get("vocab_size", 8143):
                        token_html += f'<span class="token-chip">{clean_w} [{tok_id}]</span> '
                    else:
                        token_html += f'<span class="token-chip token-oov">{clean_w} [OOV]</span> '
                st.markdown(token_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODULE 2: BATCH FILE AUDITOR
# -----------------------------------------------------------------------------
elif selected_module == "Batch File Auditor":
    st.markdown("### 📁 Batch Data Processing")
    st.caption("Upload a CSV file containing message rows to classify bulk datasets.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded dataset: `{len(df_upload)}` rows")
            st.dataframe(df_upload.head(4), use_container_width=True)

            text_columns = [col for col in df_upload.columns if df_upload[col].dtype == "object"]
            if not text_columns:
                st.error("No valid text columns found in the uploaded file.")
            else:
                target_col = st.selectbox("Select the column with text/messages:", text_columns)

                if st.button("Run Batch Classification", type="primary"):
                    progress_bar = st.progress(0, text="Preparing sequences...")
                    
                    texts = df_upload[target_col].astype(str).tolist()
                    max_len = model_config.get("max_length", 100)
                    pad_style = model_config.get("padding", "post")
                    trunc_style = model_config.get("truncating", "post")

                    sequences = tokenizer.texts_to_sequences(texts)
                    padded = pad_sequences(sequences, maxlen=max_len, padding=pad_style, truncating=trunc_style)

                    progress_bar.progress(40, text="Executing neural batch predictions...")
                    t_start = time.perf_counter()
                    raw_predictions = model.predict(padded, batch_size=64, verbose=0)
                    duration = time.perf_counter() - t_start
                    
                    progress_bar.progress(90, text="Compiling classification report...")

                    results = []
                    for i in range(len(df_upload)):
                        sp = float(raw_predictions[i][0])
                        verdict = "Spam" if sp >= active_threshold else "Not Spam"
                        results.append({
                            "Spam_Probability": round(sp, 4),
                            "Verdict": verdict
                        })

                    progress_bar.progress(100, text=f"Finished in {duration:.2f}s ({len(df_upload)/duration:.1f} rows/sec)")

                    df_results = df_upload.copy()
                    df_results["Spam_Probability"] = [r["Spam_Probability"] for r in results]
                    df_results["Predicted_Class"] = [r["Verdict"] for r in results]

                    st.markdown("---")
                    st.markdown("#### Batch Summary")
                    
                    spam_n = sum(1 for r in results if r["Verdict"] == "Spam")
                    ham_n = len(df_upload) - spam_n
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Rows Processed", len(df_upload))
                    b_col2.metric("Spam Count", spam_n, delta=f"{spam_n/len(df_upload)*100:.1f}%")
                    b_col3.metric("Not Spam Count", ham_n)

                    st.dataframe(df_results, use_container_width=True)

                    csv_export = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_export,
                        file_name=f"spam_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# -----------------------------------------------------------------------------
# MODULE 3: PREDICTION ANALYTICS
# -----------------------------------------------------------------------------
elif selected_module == "Prediction Analytics":
    st.markdown("### 📊 Session Prediction Analytics")

    if not st.session_state.audit_log:
        st.info("No prediction data logged yet in this session. Run message classifications to populate statistics.")
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
                hole=0.5,
                color="Verdict",
                color_discrete_map={"Spam": "#ef4444", "Not Spam": "#10b981"},
                title="Classification Proportions",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme["subtext"]),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with t2:
            fig_scat = px.scatter(
                df_log.reset_index(),
                x="index",
                y="Spam Probability (%)",
                color="Verdict",
                size="Confidence (%)",
                color_discrete_map={"Spam": "#ef4444", "Not Spam": "#10b981"},
                title="Prediction Distribution Timeline",
                labels={"index": "Message Index"},
            )
            fig_scat.add_hline(y=active_threshold * 100, line_dash="dot", line_color="#f59e0b", annotation_text="Threshold")
            fig_scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                font=dict(color=theme["subtext"]),
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("#### Detailed Prediction Log")
        st.dataframe(df_log, use_container_width=True)

        col_d1, _ = st.columns([1.5, 4])
        with col_d1:
            csv_data = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Prediction History",
                data=csv_data,
                file_name=f"spam_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# -----------------------------------------------------------------------------
# MODULE 4: MODEL ARCHITECTURE & SPECS
# -----------------------------------------------------------------------------
elif selected_module == "Model Architecture & Specs":
    st.markdown("### 🏛️ Recurrent Neural Network Specifications")

    col_spec1, col_spec2 = st.columns(2)

    with col_spec1:
        st.markdown("#### Layer Architecture")
        st.markdown(
            """
            * **Input Sequence**: Padded integer tokens vector `(None, 100)`
            * **Embedding Layer**: Vocabulary size `8,143` mapped to a `64`-dimensional dense space (`mask_zero=True`)
            * **Recurrent Layer (LSTM)**: `64` memory units with standard hyperbolic tangent and sigmoid gates
            * **Dropout Layer 1**: `0.3` rate to prevent overfitting on recurrent states
            * **Dense Layer**: `32` units with ReLU activation function
            * **Dropout Layer 2**: `0.3` rate
            * **Output Layer**: Single unit scalar with Sigmoid activation $[0.0, 1.0]$
            """
        )

    with col_spec2:
        st.markdown("#### Hyperparameters & Config")
        cfg_rows = [{"Configuration Property": k, "Value": str(v)} for k, v in model_config.items()]
        st.table(pd.DataFrame(cfg_rows))

    st.markdown("---")
    st.markdown("#### Label Mapping")
    le_data = [
        {"Index": i, "Encoded Class": label, "Interpretation": "Non-Spam Message (Ham)" if label == "ham" else "Spam Message"}
        for i, label in enumerate(label_encoder.classes_)
    ]
    st.table(pd.DataFrame(le_data))
