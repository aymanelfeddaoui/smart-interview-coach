import streamlit as st
import numpy as np
import pickle
import time
import cv2
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from torchvision import transforms

# ─── Imports IA ───────────────────────────────────────────────────────────────
from src.nlp.speech_to_text import transcrire_audio
from src.nlp.analyze_text import evaluer_pertinence
from src.rl_agent.q_learning_agent import discretiser_etat
from src.vision.emotion_model import EmotionCNN

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(page_title="Smart Interview Coach", page_icon="🤖", layout="wide")

# ─── CSS SPECTACULAIRE ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: #04060f !important;
  color: #f0f0ff !important;
  font-family: 'Syne', sans-serif !important;
  overflow-x: hidden;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 60% at 20% -10%, rgba(99,60,255,0.35) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 90% 110%, rgba(255,60,160,0.25) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 60% 50%, rgba(0,200,255,0.08) 0%, transparent 60%),
    #04060f !important;
}

/* ── HIDE STREAMLIT UI ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* ── MAIN CONTENT AREA ── */
.block-container {
  padding: 2rem 3rem !important;
  max-width: 1200px !important;
}

/* ── HERO TITLE ── */
.hero-wrap {
  text-align: center;
  padding: 3.5rem 0 1rem;
  position: relative;
}
.hero-badge {
  display: inline-block;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  color: #a78bfa;
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.35);
  border-radius: 50px;
  padding: 0.35rem 1.1rem;
  margin-bottom: 1.2rem;
  text-transform: uppercase;
  animation: fadeSlideDown 0.7s ease both;
}
.hero-title {
  font-size: clamp(2.8rem, 6vw, 5rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.03em;
  background: linear-gradient(120deg, #ffffff 0%, #c4b5fd 40%, #f472b6 70%, #fb923c 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeSlideDown 0.8s ease 0.1s both;
}
.hero-sub {
  font-family: 'DM Mono', monospace;
  font-size: 0.9rem;
  color: rgba(200,200,240,0.5);
  margin-top: 0.8rem;
  letter-spacing: 0.05em;
  animation: fadeSlideDown 0.8s ease 0.2s both;
}

/* ── GLASSMORPHISM CARDS ── */
.card {
  background: rgba(15, 18, 40, 0.75);
  border: 1px solid rgba(139, 92, 246, 0.22);
  border-radius: 1.5rem;
  padding: 2rem 2.2rem;
  margin-bottom: 1.2rem;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, transform 0.3s;
  animation: fadeSlideUp 0.7s ease both;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(167,139,250,0.6), rgba(244,114,182,0.4), transparent);
}
.card:hover {
  border-color: rgba(139,92,246,0.5);
  transform: translateY(-2px);
}

/* ── ACCENT CARD (colored left border) ── */
.card-accent-purple { border-left: 3px solid #8b5cf6; }
.card-accent-pink   { border-left: 3px solid #f472b6; }
.card-accent-cyan   { border-left: 3px solid #22d3ee; }
.card-accent-orange { border-left: 3px solid #fb923c; }

/* ── NEON GLOW PULSE DOT ── */
.pulse-dot {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 50%;
  background: #f472b6;
  box-shadow: 0 0 0 0 rgba(244,114,182,0.7);
  animation: pulseGlow 1.4s ease-in-out infinite;
  margin-right: 0.5rem;
  vertical-align: middle;
}

/* ── GUIDE STEPS ── */
.step-list { list-style: none; padding: 0; }
.step-item {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 0.7rem 0;
  border-bottom: 1px solid rgba(99,102,241,0.1);
  font-size: 0.95rem;
  color: rgba(220,220,255,0.85);
  animation: fadeSlideUp 0.6s ease both;
}
.step-item:last-child { border-bottom: none; }
.step-num {
  min-width: 28px; height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 700;
  color: white; flex-shrink: 0;
  margin-top: 1px;
}

/* ── SECTION HEADERS ── */
.section-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #8b5cf6;
  margin-bottom: 0.8rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(139,92,246,0.4), transparent);
}

/* ── METRIC CHIPS ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
.metric-chip {
  flex: 1; min-width: 120px;
  background: rgba(99,102,241,0.1);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 1rem;
  padding: 1rem 1.2rem;
  text-align: center;
}
.metric-chip .chip-val {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
}
.metric-chip .chip-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: rgba(200,200,240,0.55);
  margin-top: 0.3rem;
}
.chip-purple .chip-val { color: #a78bfa; }
.chip-pink   .chip-val { color: #f472b6; }
.chip-cyan   .chip-val { color: #22d3ee; }

/* ── QUESTION DISPLAY ── */
.question-box {
  background: linear-gradient(135deg, rgba(99,60,255,0.12), rgba(244,114,182,0.08));
  border: 1px solid rgba(139,92,246,0.35);
  border-radius: 1.25rem;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.question-box::after {
  content: '"';
  position: absolute;
  right: 1.5rem; bottom: -1rem;
  font-size: 8rem;
  font-weight: 900;
  color: rgba(139,92,246,0.07);
  line-height: 1;
}
.question-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  color: #a78bfa;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}
.question-text {
  font-size: 1.35rem;
  font-weight: 700;
  color: #f0f0ff;
  line-height: 1.5;
  letter-spacing: -0.01em;
}

/* ── STATUS BANNER ── */
.status-banner {
  background: linear-gradient(135deg, rgba(244,114,182,0.15), rgba(251,146,60,0.12));
  border: 1px solid rgba(244,114,182,0.35);
  border-radius: 1rem;
  padding: 1rem 1.5rem;
  display: flex; align-items: center; gap: 0.8rem;
  font-weight: 600; font-size: 0.95rem;
  color: #fda4af;
  margin-bottom: 1rem;
}

/* ── DECISION BOX ── */
.decision-box {
  background: linear-gradient(135deg, rgba(34,211,238,0.1), rgba(99,102,241,0.12));
  border: 1px solid rgba(34,211,238,0.3);
  border-radius: 1.2rem;
  padding: 1.5rem 2rem;
  margin-top: 1.2rem;
  display: flex; align-items: center; gap: 1rem;
}
.decision-icon { font-size: 2rem; }
.decision-text { font-size: 0.95rem; color: rgba(200,240,255,0.9); line-height: 1.5; }
.decision-text strong { color: #22d3ee; font-weight: 700; }

/* ── TRANSCRIPT CARD ── */
.transcript-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.88rem;
  color: rgba(200,210,255,0.8);
  background: rgba(10,12,30,0.6);
  border-radius: 0.75rem;
  padding: 1.2rem 1.5rem;
  line-height: 1.7;
  border-left: 3px solid #8b5cf6;
  margin-top: 0.5rem;
}

/* ── HISTORY TAG ── */
.history-item {
  display: flex; align-items: center; gap: 0.7rem;
  padding: 0.5rem 0;
  font-size: 0.85rem;
  color: rgba(180,180,220,0.65);
  border-bottom: 1px solid rgba(99,102,241,0.08);
  font-family: 'DM Mono', monospace;
}
.history-item:last-child { border-bottom: none; }
.htag {
  font-size: 0.65rem; padding: 0.15rem 0.5rem;
  border-radius: 50px; font-weight: 600;
  letter-spacing: 0.05em;
}
.htag-perf { background: rgba(139,92,246,0.2); color: #c4b5fd; }
.htag-stress { background: rgba(244,114,182,0.15); color: #f9a8d4; }

/* ── PRIMARY BUTTON ── */
div.stButton > button {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
  color: white !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 0.9rem !important;
  padding: 0.85rem 2rem !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: 0 4px 30px rgba(99,102,241,0.35) !important;
}
div.stButton > button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, #7c3aed, #ec4899);
  opacity: 0;
  transition: opacity 0.3s;
}
div.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 40px rgba(139,92,246,0.55) !important;
}

/* ── STREAMLIT NATIVE OVERRIDES ── */
[data-testid="stMetric"] {
  background: rgba(15,18,40,0.7) !important;
  border: 1px solid rgba(99,102,241,0.2) !important;
  border-radius: 1rem !important;
  padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: rgba(180,180,220,0.6) !important; font-family: 'DM Mono', monospace !important; }

[data-testid="stSuccess"] {
  background: rgba(34,197,94,0.1) !important;
  border: 1px solid rgba(34,197,94,0.3) !important;
  border-radius: 1rem !important;
  color: #86efac !important;
}
[data-testid="stInfo"] {
  background: rgba(34,211,238,0.08) !important;
  border: 1px solid rgba(34,211,238,0.25) !important;
  border-radius: 1rem !important;
  color: #67e8f9 !important;
}
[data-testid="stWarning"] {
  background: rgba(251,146,60,0.1) !important;
  border: 1px solid rgba(251,146,60,0.3) !important;
  border-radius: 1rem !important;
  color: #fdba74 !important;
}

/* ── DIVIDER ── */
.fancy-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), rgba(244,114,182,0.4), transparent);
  margin: 1.5rem 0;
}

/* ── ANIMATIONS ── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
  0%   { box-shadow: 0 0 0 0 rgba(244,114,182,0.7); }
  70%  { box-shadow: 0 0 0 10px rgba(244,114,182,0); }
  100% { box-shadow: 0 0 0 0 rgba(244,114,182,0); }
}
@keyframes scanline {
  0%   { background-position: 0% 0%; }
  100% { background-position: 0% 100%; }
}
</style>
""", unsafe_allow_html=True)

# ─── Constantes et Modèles ────────────────────────────────────────────────────
ACTION_MAP = {
    0: "🧊 Question brise-glace",
    1: "🤝 Question comportementale",
    2: "⚙️ Question technique"
}
DUREE_ENREGISTREMENT = 30

@st.cache_resource
def charger_modele_vision():
    model = EmotionCNN(num_classes=7)
    model.load_state_dict(torch.load('models/emotion_model_best.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# ─── State Machine ─────────────────────────────────────────────────────────────
if "etape" not in st.session_state:
    st.session_state.etape = "accueil"
if "historique" not in st.session_state:
    st.session_state.historique = []
if "question_actuelle" not in st.session_state:
    st.session_state.question_actuelle = "Parlez-moi de votre expérience avec le langage Python."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.etape == "accueil":

    # ── HERO ──
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-badge">✦ Powered by Reinforcement Learning & Computer Vision</div>
      <h1 class="hero-title">Smart Interview<br>Coach</h1>
      <p class="hero-sub">Analyse temps réel • Émotions • NLP • Stratégie adaptative</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_guide, col_action = st.columns([1.15, 0.85], gap="large")

    # ── GUIDE ──
    with col_guide:
        st.markdown("""
        <div class="card card-accent-purple">
          <div class="section-label">📖 Guide d'utilisation</div>
          <ul class="step-list">
            <li class="step-item" style="animation-delay:0.05s">
              <span class="step-num">1</span>
              <span>Cliquez sur <strong>Démarrer</strong> pour lancer votre session d'entretien.</span>
            </li>
            <li class="step-item" style="animation-delay:0.1s">
              <span class="step-num">2</span>
              <span>La question IA s'affiche instantanément sur votre écran.</span>
            </li>
            <li class="step-item" style="animation-delay:0.15s">
              <span class="step-num">3</span>
              <span><strong>Caméra & micro</strong> s'activent automatiquement en temps réel.</span>
            </li>
            <li class="step-item" style="animation-delay:0.2s">
              <span class="step-num">4</span>
              <span>Regardez la caméra et répondez naturellement à haute voix.</span>
            </li>
            <li class="step-item" style="animation-delay:0.25s">
              <span class="step-num">5</span>
              <span>L'IA analyse <strong>stress</strong>, <strong>pertinence</strong> et <strong>adapte</strong> la suite stratégiquement.</span>
            </li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        # Historique si disponible
        if st.session_state.historique:
            st.markdown("""
            <div class="card card-accent-cyan" style="animation-delay:0.2s">
              <div class="section-label">🕒 Historique de session</div>
            """, unsafe_allow_html=True)
            for i, tour in enumerate(st.session_state.historique[-3:], 1):
                st.markdown(f"""
                <div class="history-item">
                  <span>Tour {i}</span>
                  <span class="htag htag-perf">🎯 {tour['perf']:.0f}%</span>
                  <span class="htag htag-stress">💓 {tour['stress']:.0f}%</span>
                  <span style="margin-left:auto;opacity:0.5;font-size:0.75rem">{tour['action_ia']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── ACTION ──
    with col_action:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 2.5rem 2rem; animation-delay:0.15s">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🤖</div>
          <div style="font-size:1.2rem; font-weight:700; color:#c4b5fd; margin-bottom:0.5rem;">
            Prêt à vous challenger ?
          </div>
          <div style="font-family:'DM Mono',monospace; font-size:0.8rem; color:rgba(180,180,220,0.5); margin-bottom:2rem; line-height:1.6;">
            30 secondes · Analyse IA complète<br>Feedback instantané
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Démarrer l'entretien", type="primary"):
            st.session_state.etape = "action"
            st.rerun()

        st.markdown("""
        <div style="display:flex; gap:0.8rem; margin-top:1rem;">
          <div style="flex:1; background:rgba(99,60,255,0.1); border:1px solid rgba(99,102,241,0.2);
               border-radius:0.9rem; padding:1rem; text-align:center;">
            <div style="font-size:1.5rem; font-weight:800; color:#a78bfa;">RL</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:rgba(180,180,220,0.5); margin-top:0.2rem;">Q-Learning</div>
          </div>
          <div style="flex:1; background:rgba(244,114,182,0.08); border:1px solid rgba(244,114,182,0.2);
               border-radius:0.9rem; padding:1rem; text-align:center;">
            <div style="font-size:1.5rem; font-weight:800; color:#f472b6;">CV</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:rgba(180,180,220,0.5); margin-top:0.2rem;">EmotionCNN</div>
          </div>
          <div style="flex:1; background:rgba(34,211,238,0.07); border:1px solid rgba(34,211,238,0.18);
               border-radius:0.9rem; padding:1rem; text-align:center;">
            <div style="font-size:1.5rem; font-weight:800; color:#22d3ee;">NLP</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:rgba(180,180,220,0.5); margin-top:0.2rem;">Whisper</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : ACTION (Vidéo & Audio)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.etape == "action":

    # En-tête de session
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
      <div style="font-size:0.8rem; font-family:'DM Mono',monospace; color:rgba(180,180,220,0.4);">SESSION ACTIVE</div>
      <div style="flex:1; height:1px; background:linear-gradient(90deg,rgba(139,92,246,0.4),transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Question box
    st.markdown(f"""
    <div class="question-box">
      <div class="question-label">🤖 Coach IA — Question</div>
      <div class="question-text">{st.session_state.question_actuelle}</div>
    </div>
    """, unsafe_allow_html=True)

    col_video, col_info = st.columns([1.6, 1], gap="large")

    with col_video:
        video_placeholder = st.empty()

    with col_info:
        st.markdown("""
        <div class="card card-accent-pink">
          <div class="section-label">📡 Analyse temps réel</div>
          <div style="display:flex; flex-direction:column; gap:0.8rem; margin-top:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.7rem; font-size:0.88rem; color:rgba(200,200,240,0.7);">
              <span style="color:#f472b6">●</span> Détection émotionnelle (CNN)
            </div>
            <div style="display:flex; align-items:center; gap:0.7rem; font-size:0.88rem; color:rgba(200,200,240,0.7);">
              <span style="color:#a78bfa">●</span> Score de stress cumulatif
            </div>
            <div style="display:flex; align-items:center; gap:0.7rem; font-size:0.88rem; color:rgba(200,200,240,0.7);">
              <span style="color:#22d3ee">●</span> Transcription Whisper
            </div>
            <div style="display:flex; align-items:center; gap:0.7rem; font-size:0.88rem; color:rgba(200,200,240,0.7);">
              <span style="color:#fb923c">●</span> Q-Learning stratégique
            </div>
          </div>
        </div>
        <div class="card card-accent-orange" style="animation-delay:0.1s">
          <div class="section-label">💡 Conseils live</div>
          <ul style="list-style:none; padding:0; font-size:0.85rem; color:rgba(200,200,240,0.7); line-height:2;">
            <li>👁️ Regardez la caméra</li>
            <li>🗣️ Parlez clairement</li>
            <li>😌 Respirez calmement</li>
            <li>⏱️ Gérez votre temps</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        status_placeholder = st.empty()

    status_placeholder.markdown("""
    <div class="status-banner">
      <span class="pulse-dot"></span>
      Enregistrement en cours — Répondez à la question !
    </div>
    """, unsafe_allow_html=True)

    # ── LOGIQUE ENREGISTREMENT (INCHANGÉE) ──
    fs = 16000
    enregistrement_audio = sd.rec(int(DUREE_ENREGISTREMENT * fs), samplerate=fs, channels=1, dtype='int16')

    modele_vision = charger_modele_vision()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions = ['colere', 'degout', 'peur', 'joie', 'neutre', 'tristesse', 'surprise']
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((48, 48)),
        transforms.Grayscale(1), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    scores_stress = []

    while time.time() - start_time < DUREE_ENREGISTREMENT:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            tensor_visage = transform(roi_gray).unsqueeze(0)
            with torch.no_grad():
                outputs = modele_vision(tensor_visage)
                _, predicted = torch.max(outputs, 1)
                emotion_predite = emotions[predicted.item()]

            if emotion_predite in ['peur', 'colere', 'tristesse']: scores_stress.append(0.8)
            elif emotion_predite in ['joie', 'neutre']:             scores_stress.append(0.2)
            else:                                                    scores_stress.append(0.5)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (139, 92, 246), 2)
            cv2.putText(frame, emotion_predite, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (244, 114, 182), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    sd.wait()

    chemin_audio = "data/live_reponse.wav"
    write(chemin_audio, fs, enregistrement_audio)

    video_placeholder.empty()
    status_placeholder.empty()
    st.info("⚙️ Analyse des cerveaux IA en cours… Patience !")

    # ── PIPELINE IA (INCHANGÉ) ──
    stress_final = sum(scores_stress) / len(scores_stress) if scores_stress else 0.5
    texte_transcrit = transcrire_audio(chemin_audio)
    perf_score_100 = evaluer_pertinence(st.session_state.question_actuelle, texte_transcrit)
    perf_score = perf_score_100 / 100.0

    etat = discretiser_etat((stress_final, perf_score))
    with open("models/q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    action = int(np.argmax(q_table[etat[0], etat[1]]))

    st.session_state.historique.append({
        "question": st.session_state.question_actuelle,
        "reponse": texte_transcrit,
        "perf": perf_score_100,
        "stress": stress_final * 100,
        "action_ia": ACTION_MAP[action]
    })

    st.session_state.question_actuelle = ACTION_MAP[action]
    st.session_state.etape = "resultat"
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.etape == "resultat":
    dernier_tour = st.session_state.historique[-1]

    # ── HEADER ──
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
      <div style="font-size:3rem;">✅</div>
      <div style="font-size:1.6rem; font-weight:800; color:#86efac; margin-top:0.3rem;">Analyse terminée !</div>
      <div style="font-family:'DM Mono',monospace; font-size:0.78rem; color:rgba(180,180,220,0.45); margin-top:0.3rem;">
        TOUR #{len(st.session_state.historique)} COMPLÉTÉ
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 0.7], gap="large")

    with col1:
        # Transcription
        st.markdown(f"""
        <div class="card card-accent-purple">
          <div class="section-label">🎙️ Ce que l'IA a entendu</div>
          <div class="transcript-text">
            « {dernier_tour['reponse']} »
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Question posée
        st.markdown(f"""
        <div class="card card-accent-cyan" style="animation-delay:0.1s">
          <div class="section-label">❓ Question posée</div>
          <div style="font-size:0.95rem; color:rgba(200,210,255,0.8); line-height:1.6; font-style:italic;">
            {dernier_tour['question']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Décision IA
        st.markdown(f"""
        <div class="decision-box">
          <div class="decision-icon">🧠</div>
          <div class="decision-text">
            <strong>Décision Stratégique :</strong> Pour gérer votre niveau de stress et de performance,
            la prochaine question sera une : <strong>{dernier_tour['action_ia']}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Métriques
        st.markdown("""
        <div class="card card-accent-pink" style="animation-delay:0.05s">
          <div class="section-label">📊 Évaluation</div>
        """, unsafe_allow_html=True)

        perf = dernier_tour['perf']
        stress = dernier_tour['stress']

        perf_color = "#86efac" if perf >= 70 else "#fbbf24" if perf >= 40 else "#f87171"
        stress_color = "#86efac" if stress <= 35 else "#fbbf24" if stress <= 65 else "#f87171"

        st.markdown(f"""
          <div style="display:flex; flex-direction:column; gap:1.2rem; margin-top:0.5rem;">
            <div style="background:rgba(10,12,30,0.6); border-radius:1rem; padding:1.3rem; text-align:center; border:1px solid rgba(139,92,246,0.2);">
              <div style="font-size:2.8rem; font-weight:800; color:{perf_color}; line-height:1;">{perf:.0f}<span style="font-size:1.2rem; opacity:0.6">%</span></div>
              <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:rgba(180,180,220,0.5); letter-spacing:0.12em; margin-top:0.4rem;">🎯 PERTINENCE</div>
            </div>
            <div style="background:rgba(10,12,30,0.6); border-radius:1rem; padding:1.3rem; text-align:center; border:1px solid rgba(244,114,182,0.2);">
              <div style="font-size:2.8rem; font-weight:800; color:{stress_color}; line-height:1;">{stress:.0f}<span style="font-size:1.2rem; opacity:0.6">%</span></div>
              <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:rgba(180,180,220,0.5); letter-spacing:0.12em; margin-top:0.4rem;">💓 STRESS DÉTECTÉ</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Tours restants / historique count
        st.markdown(f"""
        <div class="card" style="text-align:center; animation-delay:0.1s">
          <div style="font-size:2rem; font-weight:800; color:#a78bfa;">{len(st.session_state.historique)}</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:rgba(180,180,220,0.45); letter-spacing:0.1em;">TOURS COMPLÉTÉS</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
    with col_btn2:
        if st.button("Passer à la question suivante ➡️", type="primary"):
            st.session_state.etape = "action"
            st.rerun()