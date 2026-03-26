import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os
import random

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartLink — Elderly Health AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root variables */
:root {
    --mint:    #d8f3dc;
    --teal:    #1b5e38;
    --teal-md: #2d8653;
    --teal-lt: #52b788;
    --cream:   #f8fdf9;
    --red:     #e63946;
    --amber:   #f4a261;
    --low:     #52b788;
    --card-bg: #ffffff;
    --shadow:  0 2px 20px rgba(27,94,56,.09);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'DM Serif Display', serif; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1b5e38 0%, #2d8653 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }
[data-testid="stSidebar"] input { color: #1b5e38 !important; }

/* Cards */
.card {
    background: var(--card-bg);
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    border: 1px solid rgba(82,183,136,.15);
}

/* Risk badges */
.risk-low    { background:#d8f3dc; color:#1b5e38; padding:6px 18px; border-radius:30px; font-weight:600; font-size:.9rem; display:inline-block; }
.risk-medium { background:#fff3cd; color:#856404; padding:6px 18px; border-radius:30px; font-weight:600; font-size:.9rem; display:inline-block; }
.risk-high   { background:#f8d7da; color:#842029; padding:6px 18px; border-radius:30px; font-weight:600; font-size:.9rem; display:inline-block; }

/* Metric tiles */
.metric-tile {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: var(--shadow);
    border-left: 5px solid var(--teal-lt);
    margin-bottom: .8rem;
}
.metric-tile .val { font-size:2rem; font-weight:700; color:var(--teal); }
.metric-tile .lbl { font-size:.82rem; color:#6c757d; font-weight:500; text-transform:uppercase; letter-spacing:.05em; }

/* Alert box */
.alert-box {
    background: linear-gradient(135deg, #f8d7da, #fce4e4);
    border: 2px solid #e63946;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(230,57,70,.4); }
    50%      { box-shadow: 0 0 0 10px rgba(230,57,70,0); }
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1b5e38, #2d8653) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .6rem 2rem !important;
    font-weight: 600 !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(27,94,56,.35) !important;
}

/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    border-radius: 10px !important;
    border: 1.5px solid #c3e6cb !important;
}

/* Header bar */
.topbar {
    display:flex; align-items:center; gap:12px;
    border-bottom: 2px solid var(--mint);
    padding-bottom: 1rem; margin-bottom: 1.5rem;
}
.topbar h1 { margin:0; color:var(--teal); font-size:2.2rem; }
.topbar .sub { color:#6c757d; font-size:.95rem; margin-top:.2rem; }

/* Section titles */
.sec-title {
    font-family: 'DM Serif Display', serif;
    color: var(--teal); font-size:1.35rem;
    margin-bottom:.8rem; border-left:4px solid var(--teal-lt);
    padding-left:.75rem;
}

/* Face analysis */
.face-card {
    background: linear-gradient(135deg,#1b5e38,#2d8653);
    color: white; border-radius:18px; padding:1.5rem;
    text-align:center;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--mint);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: var(--teal);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--teal) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── ML Model (Rule + weighted score — no sklearn dependency needed) ────────────
def predict_risk(bp_sys, bp_dia, sugar, heart_rate, age, has_diabetes, has_hypertension):
    """
    Weighted scoring model that mimics a trained classifier.
    Returns: (risk_level, risk_score, explanation)
    """
    score = 0
    flags = []

    # Blood pressure
    if bp_sys >= 180 or bp_dia >= 120:
        score += 40; flags.append("🚨 Hypertensive crisis (BP ≥ 180/120)")
    elif bp_sys >= 140 or bp_dia >= 90:
        score += 25; flags.append("⚠️ Stage 2 hypertension")
    elif bp_sys >= 130 or bp_dia >= 80:
        score += 12; flags.append("⚠️ Stage 1 hypertension")
    elif bp_sys < 90:
        score += 20; flags.append("⚠️ Hypotension (BP too low)")

    # Blood sugar
    if sugar > 400:
        score += 35; flags.append("🚨 Dangerously high blood sugar")
    elif sugar > 250:
        score += 25; flags.append("⚠️ Very high blood sugar (hyperglycemia)")
    elif sugar > 180:
        score += 15; flags.append("⚠️ Elevated blood sugar")
    elif sugar < 70:
        score += 30; flags.append("🚨 Hypoglycemia (sugar too low)")
    elif sugar < 54:
        score += 40; flags.append("🚨 Severe hypoglycemia")

    # Heart rate
    if heart_rate > 150:
        score += 35; flags.append("🚨 Severe tachycardia (HR > 150)")
    elif heart_rate > 100:
        score += 18; flags.append("⚠️ Tachycardia (HR > 100)")
    elif heart_rate < 40:
        score += 35; flags.append("🚨 Severe bradycardia (HR < 40)")
    elif heart_rate < 60:
        score += 10; flags.append("⚠️ Bradycardia (HR < 60)")

    # Age factor
    if age >= 80:   score += 10
    elif age >= 70: score += 6
    elif age >= 60: score += 3

    # Pre-existing
    if has_diabetes:    score += 8; flags.append("📋 Diabetic patient (elevated base risk)")
    if has_hypertension:score += 8; flags.append("📋 Hypertensive patient (elevated base risk)")

    # Risk classification
    score = min(score, 100)
    if score >= 55:
        level = "HIGH"
    elif score >= 28:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, score, flags

# ─── SMS via Twilio ─────────────────────────────────────────────────────────────
def send_sms_alert(patient_name, risk_level, score, bp_sys, bp_dia, sugar, hr,
                   account_sid, auth_token, from_number, to_number):
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        body = (
            f"🚨 HEARTLINK HEALTH ALERT 🚨\n"
            f"Patient: {patient_name}\n"
            f"Risk Level: {risk_level} (Score: {score}/100)\n"
            f"BP: {bp_sys}/{bp_dia} mmHg | Sugar: {sugar} mg/dL | HR: {hr} bpm\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"⚠️ Please contact your patient immediately!"
        )
        msg = client.messages.create(body=body, from_=from_number, to=to_number)
        return True, msg.sid
    except ImportError:
        return False, "Twilio not installed. Run: pip install twilio"
    except Exception as e:
        return False, str(e)

# ─── Fake history generator ────────────────────────────────────────────────────
def generate_history(n=14, base_bp=135, base_sugar=140, base_hr=78):
    records = []
    for i in range(n, 0, -1):
        d = datetime.now() - timedelta(days=i)
        records.append({
            "date":       d.strftime("%b %d"),
            "bp_sys":     int(base_bp + random.gauss(0, 8)),
            "bp_dia":     int((base_bp * 0.65) + random.gauss(0, 5)),
            "sugar":      int(base_sugar + random.gauss(0, 20)),
            "heart_rate": int(base_hr + random.gauss(0, 7)),
        })
    return pd.DataFrame(records)

# ─── Session state defaults ────────────────────────────────────────────────────
for k, v in {
    "history": generate_history(),
    "last_risk": None,
    "last_score": 0,
    "sms_sent": False,
    "patient_name": "Nimal Perera",
    "patient_age": 72,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 HeartLink")
    st.markdown("*AI Elderly Health Monitor**")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📥 Enter Health Data",
        "📊 Patient History",
        "😟 Facial Analysis",
        "⚙️ SMS Settings",
    ])

    st.markdown("---")
    st.markdown("### 👤 Current Patient")
    st.session_state.patient_name = st.text_input("Name", st.session_state.patient_name)
    st.session_state.patient_age  = st.number_input("Age", 50, 110, st.session_state.patient_age)

    st.markdown("---")
    if st.session_state.last_risk:
        risk = st.session_state.last_risk
        color = {"LOW":"#52b788","MEDIUM":"#f4a261","HIGH":"#e63946"}.get(risk,"#888")
        st.markdown(f"**Last Prediction:** <span style='color:{color};font-weight:700'>{risk}</span>", unsafe_allow_html=True)
        st.progress(st.session_state.last_score / 100)
        st.caption(f"Risk Score: {st.session_state.last_score}/100")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="topbar">
        <div>
            <h1>🫀 HeartLink Health AI</h1>
            <div class="sub">AI-Based Smart Elderly Health Risk Prediction & Alert System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.history
    last = df.iloc[-1]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Blood Pressure</div>
            <div class="val">{last.bp_sys}/{last.bp_dia}</div>
            <div style="color:#6c757d;font-size:.8rem">mmHg</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Blood Sugar</div>
            <div class="val">{last.sugar}</div>
            <div style="color:#6c757d;font-size:.8rem">mg/dL</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Heart Rate</div>
            <div class="val">{last.heart_rate}</div>
            <div style="color:#6c757d;font-size:.8rem">bpm</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        risk = st.session_state.last_risk or "—"
        badge_class = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}.get(risk, "risk-low")
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Current Risk</div>
            <div style="margin-top:.5rem"><span class="{badge_class}">{risk}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<div class="sec-title">📈 Vital Trends (14 Days)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, y=df.bp_sys, name="BP Systolic",
            line=dict(color="#1b5e38", width=2.5), mode="lines+markers",
            marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=df.date, y=df.bp_dia, name="BP Diastolic",
            line=dict(color="#52b788", width=2, dash="dot"), mode="lines+markers",
            marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df.date, y=df.sugar, name="Blood Sugar",
            line=dict(color="#f4a261", width=2.5), mode="lines+markers",
            marker=dict(size=6), yaxis="y2"))
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font_family="DM Sans", height=300, margin=dict(l=0,r=0,t=20,b=0),
            yaxis=dict(title="BP (mmHg)", gridcolor="#f0f0f0"),
            yaxis2=dict(title="Sugar (mg/dL)", overlaying="y", side="right",gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-title">💓 Heart Rate Trend</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df.date, y=df.heart_rate, fill="tozeroy",
            fillcolor="rgba(82,183,136,.15)",
            line=dict(color="#2d8653", width=2.5),
            mode="lines+markers", marker=dict(size=6)
        ))
        fig2.add_hline(y=100, line_dash="dash", line_color="#e63946", annotation_text="Danger (100)")
        fig2.add_hline(y=60,  line_dash="dash", line_color="#f4a261", annotation_text="Low (60)")
        fig2.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font_family="DM Sans", height=300, margin=dict(l=0,r=0,t=20,b=0),
            yaxis=dict(gridcolor="#f0f0f0", range=[40,130]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Risk gauge
    if st.session_state.last_risk:
        st.markdown('<div class="sec-title">🎯 AI Risk Gauge</div>', unsafe_allow_html=True)
        score = st.session_state.last_score
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 30},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#1b5e38"},
                "steps":[
                    {"range":[0,27],  "color":"#d8f3dc"},
                    {"range":[27,54], "color":"#fff3cd"},
                    {"range":[54,100],"color":"#f8d7da"},
                ],
                "threshold":{"line":{"color":"#e63946","width":4},"thickness":.75,"value":55}
            },
            title={"text": "Health Risk Score", "font": {"family":"DM Serif Display"}}
        ))
        gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=0),
                            paper_bgcolor="white", font_family="DM Sans")
        st.plotly_chart(gauge, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ENTER HEALTH DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Enter Health Data":
    st.markdown('<div class="topbar"><div><h1>📥 Enter Health Data</h1><div class="sub">Log vitals for AI risk assessment</div></div></div>', unsafe_allow_html=True)

    with st.form("vitals_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🩺 Blood Pressure")
            bp_sys = st.number_input("Systolic (mmHg)", 60, 250, 135, help="Top number")
            bp_dia = st.number_input("Diastolic (mmHg)", 40, 150, 85, help="Bottom number")

            st.markdown("#### 🍬 Blood Sugar")
            sugar = st.number_input("Glucose Level (mg/dL)", 30, 600, 145)

        with col2:
            st.markdown("#### 💓 Heart Rate")
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 220, 80)

            st.markdown("#### 📋 Medical History")
            has_diabetes = st.checkbox("Has Diabetes")
            has_hypertension = st.checkbox("Has Hypertension")

            st.markdown("#### 📝 Notes")
            notes = st.text_area("Symptoms / Notes (optional)", height=80)

        submitted = st.form_submit_button("🤖 Run AI Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Running AI analysis..."):
            time.sleep(1.2)  # dramatic pause for demo
            risk, score, flags = predict_risk(
                bp_sys, bp_dia, sugar, heart_rate,
                st.session_state.patient_age, has_diabetes, has_hypertension
            )

        st.session_state.last_risk = risk
        st.session_state.last_score = score

        # Add to history
        new_row = pd.DataFrame([{
            "date": datetime.now().strftime("%b %d"),
            "bp_sys": bp_sys, "bp_dia": bp_dia,
            "sugar": sugar, "heart_rate": heart_rate
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(30)

        # Result display
        badge_class = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}[risk]
        badge_emoji = {"LOW":"✅","MEDIUM":"⚠️","HIGH":"🚨"}[risk]
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2rem;">
            <div style="font-size:3rem">{badge_emoji}</div>
            <h2 style="font-family:'DM Serif Display',serif;color:#1b5e38;margin:.5rem 0">
                {st.session_state.patient_name}
            </h2>
            <span class="{badge_class}" style="font-size:1.1rem">{risk} RISK</span>
            <div style="margin-top:1rem;font-size:1.5rem;font-weight:700;color:#1b5e38">
                Risk Score: {score}/100
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Flags
        if flags:
            st.markdown('<div class="sec-title">🔍 Clinical Findings</div>', unsafe_allow_html=True)
            for f in flags:
                st.markdown(f"- {f}")

        # HIGH RISK — trigger SMS prompt
        if risk == "HIGH":
            st.markdown("""
            <div class="alert-box">
                <h3 style="color:#842029;margin:0">🚨 HIGH RISK DETECTED</h3>
                <p style="color:#842029;margin:.5rem 0 0">
                    AI has flagged this as a medical emergency. Send SMS alert immediately.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📱 Send Emergency SMS Alert Now"):
                cfg = st.session_state.get("twilio_cfg", {})
                if all(cfg.get(k) for k in ["sid","token","from","to"]):
                    ok, info = send_sms_alert(
                        st.session_state.patient_name, risk, score,
                        bp_sys, bp_dia, sugar, heart_rate,
                        cfg["sid"], cfg["token"], cfg["from"], cfg["to"]
                    )
                    if ok:
                        st.success(f"✅ SMS sent! Message SID: {info}")
                        st.session_state.sms_sent = True
                    else:
                        st.error(f"❌ SMS failed: {info}")
                else:
                    st.warning("⚙️ Please configure Twilio credentials in **SMS Settings** first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Patient History":
    st.markdown('<div class="topbar"><div><h1>📊 Patient History</h1><div class="sub">14-day health trend analysis</div></div></div>', unsafe_allow_html=True)

    df = st.session_state.history.copy()

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_bp = f"{int(df.bp_sys.mean())}/{int(df.bp_dia.mean())}"
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Avg Blood Pressure</div>
            <div class="val">{avg_bp}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Avg Blood Sugar</div>
            <div class="val">{int(df.sugar.mean())} mg/dL</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-tile">
            <div class="lbl">Avg Heart Rate</div>
            <div class="val">{int(df.heart_rate.mean())} bpm</div>
        </div>""", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🩸 Blood Pressure Detail", "📋 Raw Data"])

    with tab1:
        fig = px.line(df, x="date", y=["bp_sys","bp_dia","sugar","heart_rate"],
                      color_discrete_map={
                          "bp_sys":"#1b5e38","bp_dia":"#52b788",
                          "sugar":"#f4a261","heart_rate":"#e63946"
                      },
                      labels={"value":"Reading","date":"Date","variable":"Metric"},
                      title="All Vitals Over Time")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          font_family="DM Sans", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Systolic",  x=df.date, y=df.bp_sys, marker_color="#1b5e38"))
        fig2.add_trace(go.Bar(name="Diastolic", x=df.date, y=df.bp_dia, marker_color="#52b788"))
        fig2.add_hline(y=140, line_dash="dash", line_color="#e63946", annotation_text="Hypertension threshold")
        fig2.update_layout(barmode="group", paper_bgcolor="white", plot_bgcolor="white",
                           font_family="DM Sans", height=350,
                           yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(df.set_index("date").style.highlight_between(
            subset=["bp_sys"], left=140, right=300, color="#f8d7da"
        ).highlight_between(
            subset=["sugar"], left=180, right=600, color="#fff3cd"
        ), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FACIAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "😟 Facial Analysis":
    st.markdown('<div class="topbar"><div><h1>😟 Facial Discomfort Analysis</h1><div class="sub">AI detects pain or distress from facial expressions</div></div></div>', unsafe_allow_html=True)

    st.info("📷 This module uses your webcam + DeepFace AI to detect emotions. In a live deployment, this runs continuously on a bedside tablet or smart camera.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="sec-title">📸 Live Camera Feed</div>', unsafe_allow_html=True)

        use_camera = st.toggle("Enable Webcam Analysis", value=False)

        if use_camera:
            img_file = st.camera_input("Point camera at patient's face")

            if img_file is not None:
                import PIL.Image, io
                img = PIL.Image.open(img_file)

                # Save temp
                tmp_path = "/tmp/rysera_face.jpg"
                img.save(tmp_path)

                with st.spinner("🤖 Analysing facial expression..."):
                    try:
                        from deepface import DeepFace
                        result = DeepFace.analyze(tmp_path, actions=["emotion"], enforce_detection=False)
                        emotions = result[0]["emotion"]
                        dominant = result[0]["dominant_emotion"]

                        # Discomfort = fear + sad + angry + disgust
                        discomfort_score = (
                            emotions.get("fear",0) +
                            emotions.get("sad",0) +
                            emotions.get("angry",0) +
                            emotions.get("disgust",0)
                        ) / 100.0

                        st.markdown(f"""
                        <div class="card">
                            <div class="sec-title">🧠 Emotion Analysis Result</div>
                            <p><b>Dominant Emotion:</b> {dominant.upper()}</p>
                            <p><b>Discomfort Index:</b> {discomfort_score*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Emotion bars
                        em_df = pd.DataFrame(list(emotions.items()), columns=["Emotion","Score"])
                        fig = px.bar(em_df, x="Emotion", y="Score",
                                     color="Score",
                                     color_continuous_scale=["#d8f3dc","#f4a261","#e63946"])
                        fig.update_layout(paper_bgcolor="white",plot_bgcolor="white",
                                          font_family="DM Sans",height=280,
                                          showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        if discomfort_score > 0.5:
                            st.markdown("""
                            <div class="alert-box">
                                <b>🚨 High discomfort detected!</b><br>
                                Patient may be in pain or distress. Consider triggering a health check.
                            </div>
                            """, unsafe_allow_html=True)

                    except ImportError:
                        # Simulate for demo
                        st.warning("DeepFace not installed — showing simulated demo output.")
                        emotions_demo = {
                            "happy":5.2,"sad":18.3,"angry":12.1,
                            "fear":22.4,"surprise":8.0,"disgust":9.0,"neutral":25.0
                        }
                        dominant_demo = "neutral"
                        discomfort_demo = (22.4+18.3+12.1+9.0)/100

                        st.markdown(f"""
                        <div class="card">
                            <div class="sec-title">🧠 Emotion Analysis (Simulated)</div>
                            <p><b>Dominant Emotion:</b> {dominant_demo.upper()}</p>
                            <p><b>Discomfort Index:</b> {discomfort_demo*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        em_df = pd.DataFrame(list(emotions_demo.items()), columns=["Emotion","Score"])
                        fig = px.bar(em_df, x="Emotion", y="Score",
                                     color="Score",
                                     color_continuous_scale=["#d8f3dc","#f4a261","#e63946"])
                        fig.update_layout(paper_bgcolor="white",plot_bgcolor="white",
                                          font_family="DM Sans",height=280,showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        else:
            st.markdown("""
            <div class="face-card">
                <div style="font-size:4rem">📷</div>
                <h3 style="color:white">Camera is Off</h3>
                <p style="color:rgba(255,255,255,.8)">Toggle the switch above to enable facial analysis</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="sec-title">ℹ️ How It Works</div>
            <p style="font-size:.88rem;color:#555">
                <b>1. Capture</b><br>Webcam takes a photo of the patient's face<br><br>
                <b>2. Analyse</b><br>DeepFace AI detects 7 emotions: happy, sad, angry, fear, surprise, disgust, neutral<br><br>
                <b>3. Score</b><br>A <i>discomfort index</i> is calculated from negative emotions<br><br>
                <b>4. Alert</b><br>If discomfort &gt; 50%, an alert is suggested to the caregiver<br><br>
            </p>
        </div>
        <div class="card">
            <div class="sec-title">😣 Discomfort Emotions</div>
            <p style="font-size:.88rem;color:#555">
                🔴 Fear &nbsp;&nbsp; 🟠 Sadness<br>
                🟠 Anger &nbsp; 🟡 Disgust<br><br>
                <b>Safe Emotions:</b><br>
                🟢 Happy &nbsp; 🟢 Neutral
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SMS SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ SMS Settings":
    st.markdown('<div class="topbar"><div><h1>⚙️ SMS Alert Settings</h1><div class="sub">Configure Twilio for live emergency alerts</div></div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="sec-title">📱 Twilio Configuration</div>
        <p style="color:#555;font-size:.9rem">
            Get your credentials at <b>console.twilio.com</b> — free trial gives you $15 credit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cfg = st.session_state.get("twilio_cfg", {})

    with st.form("twilio_form"):
        sid   = st.text_input("Account SID",   value=cfg.get("sid",""),   type="password", placeholder="ACxxxxxxxxxxxxxxxx")
        token = st.text_input("Auth Token",     value=cfg.get("token",""), type="password", placeholder="your_auth_token")
        frm   = st.text_input("From Number",   value=cfg.get("from",""),  placeholder="+1415XXXXXXX (your Twilio number)")
        to    = st.text_input("To Number",     value=cfg.get("to",""),    placeholder="+94XXXXXXXXX (recipient)")

        c1, c2 = st.columns(2)
        with c1:
            save = st.form_submit_button("💾 Save Settings", use_container_width=True)
        with c2:
            test = st.form_submit_button("📤 Send Test SMS",  use_container_width=True)

    if save or test:
        st.session_state.twilio_cfg = {"sid":sid,"token":token,"from":frm,"to":to}
        if save:
            st.success("✅ Twilio settings saved!")

        if test:
            if all([sid, token, frm, to]):
                with st.spinner("Sending test SMS..."):
                    ok, info = send_sms_alert(
                        st.session_state.patient_name, "TEST", 0,
                        120, 80, 100, 72,
                        sid, token, frm, to
                    )
                if ok:
                    st.success(f"✅ Test SMS sent! SID: {info}")
                else:
                    st.error(f"❌ Failed: {info}")
            else:
                st.warning("Please fill in all fields first.")

    st.markdown("""
    <div class="card" style="margin-top:1.5rem">
        <div class="sec-title">📋 Alert Recipients</div>
        <p style="color:#555;font-size:.9rem">In a production system, alerts go to:</p>
        <ul style="color:#555;font-size:.9rem">
            <li>👨‍👩‍👧 Children / family members overseas</li>
            <li>👨‍⚕️ Family doctor or GP</li>
            <li>🏥 Nearest hospital emergency contact</li>
        </ul>
        <p style="color:#555;font-size:.9rem">
            Twilio supports international SMS — family abroad will receive alerts instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)
