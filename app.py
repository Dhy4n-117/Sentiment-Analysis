import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from typing import Dict, List

# Import all our logic functions from the new file
from analysis_logic import (
    load_models,
    analyze_text_comprehensive,
    extract_aspects,
    download_youtube_video,
    chatbot_response,
    recognize_speech,  # <-- CORRECTLY IMPORTED
    extract_audio_from_video,
    transcribe_audio,
    scrape_webpage_text,
    analyze_speech_emotion,
    fetch_youtube_metadata,
    download_youtube_audio_simple,
    convert_to_wav_16k_mono
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Advanced Sentiment Analysis",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    @import url('https_//fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-right: 1px solid #334155; }

    [data-testid="stSidebar"] > div:first-child > div:first-child {
        padding-top: 1rem; 
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child > div:nth-child(1) h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700; 
        font-size: 1.75rem; 
        letter-spacing: -0.5px;
        color: #f1f5f9; 
        padding-top: 0;
        margin-top: 0;
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child > div:nth-child(1) p {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        color: #94a3b8; 
        margin-top: -10px; 
        margin-bottom: 20px; 
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 3.0rem; 
        font-weight: 700;
        color: #A79BFF; /* Main title purple */
        margin-bottom: 1rem;
        text-align: center; /* Centered */
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center; /* Centered */
        margin-bottom: 2rem;
        margin-top: -1rem; 
    }

    .feature-badges { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin: 2rem 0; }
    .badge { 
        background: rgba(167, 155, 255, 0.1); /* New purple alpha */
        border: 1px solid #A79BFF; /* New purple border */
        color: #f1f5f9; 
        padding: 0.5rem 1rem; 
        border-radius: 50px; /* Pill shape */
        font-size: 0.9rem; 
        font-weight: 500; 
    }

    .sentiment-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 2rem; text-align: center; color: white; margin: 1rem 0; }
    .sentiment-score { font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; }
    .sentiment-label { font-size: 1.3rem; text-transform: uppercase; letter-spacing: 2px; }

    .stButton>button {
        background: linear-gradient(90deg, #004aad, #cb6ce6); /* Your new gradient */
        color: #f1f5f9; /* White text */
        border: none; /* Removed border */
        border-radius: 50px; /* Pill shape */
        padding: 0.75rem 1.5rem; 
        font-weight: 600; 
        transition: all 0.3s ease; 
        width: 100%; 
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 5px 15px rgba(203, 108, 230, 0.4); /* Shadow from gradient */
        filter: brightness(1.2); /* Brighten on hover */
    }

    /* --- NEW CARD CONTAINER STYLE --- */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(167, 155, 255, 0.05); /* Faint purple background */
        border-color: #A79BFF; /* Purple border */
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    /* --- END NEW CARD STYLE --- */

    /* Sidebar radio button styles */
    [data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label {
        padding: 10px 12px;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
        margin-bottom: 5px;
        border: 1px solid transparent;
        cursor: pointer;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: rgba(167, 155, 255, 0.25); /* Made hover effect stronger */
        border-color: #A79BFF; /* New purple border */
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background-color: rgba(167, 155, 255, 0.15); /* New purple selected */
        border-color: #A79BFF; /* New purple border */
        font-weight: 600;
        color: #f1f5f9;
    }

    .emotion-card { background: rgba(15, 23, 42, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #334155; margin: 0.5rem; }
    .emotion-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .emotion-name { font-weight: 600; color: #f1f5f9; margin-bottom: 0.5rem; }
    .emotion-value { font-size: 1.5rem; font-weight: 700; color: #06b6d4; }
    .chat-message { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .user-message { background: rgba(99, 102, 241, 0.2); margin-left: 20%; }
    .bot-message { background: rgba(15, 23, 42, 0.5); margin-right: 20%; }
    .aspect-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid; }
    .aspect-positive { border-color: #10b981; }
    .aspect-negative { border-color: #ef4444; }
    .aspect-neutral { border-color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Models
# ----------------------------
bert_analyzer, vader_analyzer, nlp_model, emotion_pipeline, emotion_pipeline_all, whisper_model, sarcasm_pipeline, ser_pipeline = load_models()
if not bert_analyzer or not whisper_model or not sarcasm_pipeline or not ser_pipeline:
    st.error("Fatal Error: Could not load AI models. The app cannot start.")
    st.stop()

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"  # Used for navigation


# ----------------------------
# Navigation Callback
# ----------------------------
def set_page(page_name):
    st.session_state.page = page_name


# ----------------------------
# Display Functions (UI Logic)
# ----------------------------
def create_sentiment_gauge(score, sentiment_label):
    """
    Creates a Plotly gauge chart for the sentiment score.
    Score is assumed to be from 0 to 1.
    """

    if sentiment_label == 'POSITIVE':
        gauge_color = "#4CAF50"  # Green
        title_color = "#4CAF50"
    elif sentiment_label == 'NEGATIVE':
        gauge_color = "#F44336"  # Red
        title_color = "#F44336"
    else:  # NEUTRAL
        gauge_color = "#FBBC05"  # Yellow
        title_color = "#FBBC05"

    title_text = f"Overall Sentiment: <span style='color:{title_color}; font-weight:bold;'>{sentiment_label}</span>"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'valueformat': '.2f', 'font': {'size': 30}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title_text, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, 0.4], 'color': '#FFCDD2'},  # Light Red
                {'range': [0.4, 0.6], 'color': '#FFF9C4'},  # Light Yellow
                {'range': [0.6, 1], 'color': '#C8E6C9'}  # Light Green
            ],
            'threshold': {
                'line': {'color': "gray", 'width': 4},
                'thickness': 0.75,
                'value': 0.5  # Neutral line
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def get_multi_modal_insight(text_emotion: str, speech_emotion: str) -> str:
    """
    Compares the top text emotion and top speech emotion to provide a multi-modal insight.
    """
    text_emotion = text_emotion.lower()
    speech_emotion = speech_emotion.lower()

    positive_emotions = ['joy', 'love', 'admiration', 'amusement', 'approval', 'caring', 'excitement', 'gratitude',
                         'optimism', 'relief']
    negative_emotions = ['sadness', 'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief',
                         'nervousness', 'remorse']

    speech_map = {
        'hap': 'joy',
        'sad': 'sadness',
        'ang': 'anger',
        'neu': 'neutral',
        'dis': 'disgust',
        'fea': 'fear',
        'sur': 'surprise'
    }
    speech_emotion = speech_map.get(speech_emotion, 'neutral')

    if text_emotion in positive_emotions and speech_emotion in ['anger', 'sadness', 'disgust']:
        return f"🚨 **High Conflict Detected:** The words are **{text_emotion.title()}**, but the tone of voice is **{speech_emotion.title()}**. This is a strong indicator of sarcasm or irony."

    if text_emotion in negative_emotions and speech_emotion == 'joy':
        return f"🚨 **High Conflict Detected:** The words are **{text_emotion.title()}**, but the tone of voice is **Joyful**. This could be sarcastic humor or schadenfreude."

    if (text_emotion in positive_emotions and speech_emotion == 'joy') or \
            (text_emotion in negative_emotions and speech_emotion in ['sadness', 'anger']):
        return f"✅ **High Agreement:** Both the text (**{text_emotion.title()}**) and the tone (**{speech_emotion.title()}**) are aligned, indicating a genuine expression."

    if speech_emotion == 'neutral' and text_emotion not in ['neutral', 'realization', 'confusion']:
        return f"ℹ️ **Informative Tone:** The text expresses **{text_emotion.title()}**, but the tone of voice is **Neutral**. This often happens when someone is stating a fact or reading a review."

    if text_emotion == 'neutral' and speech_emotion != 'neutral':
        return f"ℹ️ **Tonal Emphasis:** The words are neutral, but the tone of voice is **{speech_emotion.title()}**. The *how* it's being said is more important than *what* is being said."

    return f"**Text Emotion:** {text_emotion.title()}, **Speech Emotion:** {speech_emotion.title()}. The emotional read is mixed."


def show_new_results(result: Dict, aspects: List[Dict] = None, speech_emotion_results: Dict = None):
    """
    Display analysis results in a clean one-page layout.
    """

    # --- PATCH: defensive validation to avoid None / malformed result access ---
    if result is None:
        st.error("No analysis result to display.")
        return

    if not isinstance(result, dict):
        st.error(f"Invalid analysis result format: expected dict but got {type(result).__name__}")
        return

    if 'text' not in result:
        st.error("Invalid analysis result: missing key 'text'. Cannot display results.")
        return

    # Additional required keys check to avoid later KeyErrors
    required_keys = [
        'combined_score', 'final_sentiment', 'emotions', 'word_count', 'confidence',
        'bert_label', 'bert_score', 'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu'
    ]
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        st.error(f"Invalid analysis result: missing keys {missing_keys}. Cannot display results.")
        # Print the result for debugging in server logs
        print("DEBUG: Malformed analysis result returned:", repr(result))
        return
    # --- end PATCH ---

    if aspects is None:
        with st.spinner("Extracting aspects..."):
            aspects = extract_aspects(
                result['text'],
                nlp_model,
                bert_analyzer,
                vader_analyzer,
                emotion_pipeline_all,
                sarcasm_pipeline
            )

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    # --- CARD 1: SENTIMENT GAUGE ---
    with st.container(border=True):
        st.subheader("📈 Hybrid Sentiment Score")
        try:
            fig_gauge = create_sentiment_gauge(result['combined_score'], result['final_sentiment'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate sentiment gauge: {e}")
            st.markdown(f"""
            <div class="sentiment-card">
                <div class="sentiment-score">{result['combined_score']:.2f}</div>
                <div class="sentiment-label">{result['final_sentiment']}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- NEW CARD: MULTI-MODAL INSIGHT ---
        # PASTE this new block (in place of lines 297-308):

        # Check that it's a dictionary AND not empty
        if isinstance(speech_emotion_results, dict) and speech_emotion_results:
            with st.container(border=True):
                st.subheader("🤖 Multi-Modal Insight")
                try:
                    # Add a check for text emotions too
                    if result.get('emotions') and speech_emotion_results:
                        top_text_emotion = \
                        sorted(result['emotions'].items(), key=lambda item: item[1], reverse=True)[0][0]
                        top_speech_emotion = \
                        sorted(speech_emotion_results.items(), key=lambda item: item[1], reverse=True)[0][0]

                        insight = get_multi_modal_insight(top_text_emotion, top_speech_emotion)
                        st.markdown(insight)
                    else:
                        st.info("Could not generate multi-modal insight (no text or speech emotions found).")
                except Exception as e:
                    st.error(f"Could not generate multi-modal insight: {e}")


    # --- CARD 3: TEXT EMOTION ANALYSIS ---
    with st.container(border=True):
        st.subheader("😊 Text Emotion Analysis (from Words)")
        try:
            sorted_emotions = sorted(result['emotions'].items(), key=lambda item: item[1], reverse=True)
            top_7_emotions = dict(sorted_emotions[:7])

            if top_7_emotions:
                df_emotions = pd.DataFrame(top_7_emotions.items(), columns=['Emotion', 'Score (%)'])

                fig = px.bar(df_emotions, x='Emotion', y='Score (%)', color='Emotion',
                             title="Detected Emotions (Top 7)", text='Score (%)')
                fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')

                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False, margin=dict(t=50))

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emotions detected.")
        except Exception as e:
            st.error(f"Could not generate text emotion chart: {e}")
            st.write(result['emotions'])

    # --- CARD 4: SPEECH EMOTION ANALYSIS ---
    if speech_emotion_results:
        with st.container(border=True):
            st.subheader("🗣️ Speech Emotion Analysis (from Audio Tone)")
            try:
                speech_emotion_data_raw = {label.title(): score for label, score in speech_emotion_results.items() if
                                           score > 1}
                sorted_speech_emotions = sorted(speech_emotion_data_raw.items(), key=lambda item: item[1],
                                                reverse=True)[:7]

                if sorted_speech_emotions:
                    df_speech_emotions = pd.DataFrame(sorted_speech_emotions, columns=['Emotion', 'Score (%)'])
                    fig_speech = px.bar(df_speech_emotions, x='Emotion', y='Score (%)', color='Emotion',
                                        title="Detected Emotions from Voice Tone (Top 7)", text='Score (%)')
                    fig_speech.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
                    fig_speech.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False,
                                             margin=dict(t=50))
                    st.plotly_chart(fig_speech, use_container_width=True)
                else:
                    st.info("No strong speech emotions detected.")
            except Exception as e:
                st.error(f"Could not generate speech emotion chart: {e}")
                st.write(speech_emotion_results)

    # --- CARD 5: ASPECT-BASED SENTIMENT ---
    with st.container(border=True):
        st.subheader("🎯 Aspect-Based Sentiment")
        if not aspects:
            st.info("No specific aspects were detected in the text.")
        else:
            sentiment_emoji = {'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐'}
            for aspect in aspects:
                sentiment_class = f"aspect-{aspect['sentiment'].lower()}"
                st.markdown(f"""
                <div class="aspect-card {sentiment_class}">
                    <strong>📌 {aspect['aspect'].title()}</strong><br>
                    Sentiment: {sentiment_emoji.get(aspect['sentiment'], '😐')} {aspect['sentiment']} (Score: {aspect['score']:.2f})<br>
                    <em>Context: "{aspect['context'][:100]}..."</em>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

    # --- CARD 6: DETAILED BREAKDOWN ---
    with st.container(border=True):
        st.subheader("🔍 Detailed Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", result['word_count'])
            st.metric("Confidence", f"{result['confidence']:.2%}")

            sarcasm_score = result.get('sarcasm_score', 0)
            st.metric("Sarcasm Score", f"{sarcasm_score:.1%}")
            if sarcasm_score > 0.6:
                st.warning("High sarcasm detected! Sentiment may be ironic.")

        with col2:
            st.markdown("*RoBERTa Analysis (80%)*")
            st.write(f"- Label: {result['bert_label']}")

            bert_sent = result['bert_sentiment']
            if bert_sent == 'POSITIVE':
                st.markdown(f"- Sentiment: <span style='color:#4CAF50; font-weight:bold;'>{bert_sent}</span>",
                            unsafe_allow_html=True)
            elif bert_sent == 'NEGATIVE':
                st.markdown(f"- Sentiment: <span style='color:#F44336; font-weight:bold;'>{bert_sent}</span>",
                            unsafe_allow_html=True)
            else:  # NEUTRAL
                st.markdown(f"- Sentiment: <span style='color:#FBBC05; font-weight:bold;'>{bert_sent}</span>",
                            unsafe_allow_html=True)

            st.write(f"- Confidence: {result['bert_score']:.2%}")

        with col3:
            st.markdown("*VADER Analysis (20%)*")
            st.write(f"- Compound: {result['vader_compound']:.3f}")
            st.write(f"- Positive: {result['vader_pos']:.2%}")
            st.write(f"- Negative: {result['vader_neg']:.2%}")
            st.write(f"- Neutral: {result['vader_neu']:.2%}")


def save_to_history(result: Dict):
    """Save to history"""
    if result:
        st.session_state.history.append(result)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### Dashboard")
    st.markdown("Advanced Sentiment Analysis")
    st.markdown("---")

    st.radio(
        "Navigation",
        ["🏠 Home", "📝 Analyzer", "🎬 File Analysis", "🤖 Chatbot", "📚 History", "ℹ️ About"],
        key="page",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("📊 Total Analyses", len(st.session_state.history))

    if st.session_state.history:
        avg_score = np.mean([h['combined_score'] for h in st.session_state.history])
        st.metric("📈 Avg Score", f"{avg_score:.2f}")

    st.markdown("---")
    st.caption("Built with RoBERTa + VADER • SpaCy")

# ----------------------------
# Pages
# ----------------------------

if st.session_state.page == "🏠 Home":
    st.markdown('<div class="hero-title">Sentiment Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced Multi-Modal Sentiment Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-badges">
        <span class="badge">🤖 RoBERTa + VADER</span>
        <span class="badge">🎬 Video Analysis</span>
        <span class="badge">🎵 Audio Analysis</span>
        <span class="badge">🎯 Aspect Analysis</span>
        <span class="badge">💬 AI Chatbot</span>
        <span class="badge">🌐 Webpage Analysis</span>
        <span class="badge">🧐 Sarcasm Detection</span>
        <span class="badge">🗣️ Speech Emotion</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📝 Text Analysis", on_click=set_page, args=["📝 Analyzer"], use_container_width=True)
        st.markdown("<p style='text-align:center;'>Dual-model AI sentiment detection</p>", unsafe_allow_html=True)
    with col2:
        st.button("🎬 File Analysis", on_click=set_page, args=["🎬 File Analysis"], use_container_width=True)
        st.markdown("<p style='text-align:center;'>Transcribe video & audio files</p>", unsafe_allow_html=True)
    with col3:
        st.button("🤖 Chatbot", on_click=set_page, args=["🤖 Chatbot"], use_container_width=True)
        st.markdown("<p style='text-align:center;'>Emotion-aware conversations</p>", unsafe_allow_html=True)

    st.info("👉 Navigate to any section to start analyzing!")

elif st.session_state.page == "📝 Analyzer":
    st.markdown("## 📝 Advanced Sentiment Analyzer")

    # --- RE-ADDED THE "Voice" TAB ---
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🎤 Voice", "🎯 Aspects", "🌐 URL"])

    with tab1:
        st.markdown("### Text Analysis")
        text_input = st.text_area("Enter text", height=200, key="text_analysis_input",
                                  placeholder="Type or paste your text here...")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Analyze Text", use_container_width=True, key="analyze_text_btn"):
                if text_input.strip():
                    with st.spinner("🔄 Analyzing..."):
                        result = analyze_text_comprehensive(text_input, bert_analyzer, vader_analyzer,
                                                            emotion_pipeline_all, sarcasm_pipeline)
                        if result:
                            st.success("✅ Complete!")
                            show_new_results(result, aspects=None, speech_emotion_results=None)
                            save_to_history(result)
                else:
                    st.warning("⚠️ Please enter some text")

        with col2:
            if st.button("💡 Example", use_container_width=True, key="example_btn"):
                st.info("Example: 'I love this product! It's amazing and exceeded my expectations.'")

    # --- RE-ADDED THE "Voice" TAB CONTENT ---
    with tab2:
        st.markdown("### 🎤 Voice Recording")

        try:
            import pyaudio

            pyaudio_available = True
        except ImportError:
            pyaudio_available = False

        if not pyaudio_available:
            st.warning("⚠️ *PyAudio not installed.* This is required for microphone access.")
            st.info(
                "Install PyAudio with: pip install pyaudio")
        else:
            st.info("🎤 Click to record your voice")
            if st.button("🎙️ Start Recording", use_container_width=True, key="record_btn"):

                with st.spinner("Initializing...") as spinner:
                    transcript = recognize_speech(spinner, whisper_model)

                # --- PATCH: debug print so server console shows the raw transcript result ---
                try:
                    print("DEBUG: recognize_speech returned (repr):", repr(transcript))
                except Exception:
                    pass
                # --- end PATCH ---

                # --- PATCH: safe check to avoid NoneType.startswith crash ---
                if isinstance(transcript, str) and not transcript.startswith("❌"):
                    st.success(f"✅ Transcribed: {transcript}")
                    with st.spinner("🔄 Analyzing..."):
                        result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer,
                                                            emotion_pipeline_all, sarcasm_pipeline)
                        if result:
                            # Note: We can't do speech emotion here since we only saved the transcript, not the audio file.
                            # This is a limitation of the current quick record feature.
                            show_new_results(result)
                            save_to_history(result)
                else:
                    # If not a string, show a clearer message
                    if transcript is None:
                        st.error(
                            "❌ Transcription returned None (no audio captured). Try again or check microphone permissions.")
                    elif isinstance(transcript, str):
                        st.error(transcript)  # an error string like "❌ ..."
                    else:
                        st.error(
                            f"❌ Unexpected transcription result type: {type(transcript).__name__}. See server logs.")

    with tab3:
        st.markdown("### 🎯 Aspect-Based Analysis")
        aspect_text = st.text_area("Enter text with multiple aspects", height=200, key="aspect_input",
                                   placeholder="e.g., 'The camera is great, but the battery life is disappointing.'")

        if st.button("🔍 Analyze Aspects", use_container_width=True, key="aspect_btn"):
            if aspect_text.strip():
                with st.spinner("Extracting aspects..."):
                    aspects = extract_aspects(aspect_text, nlp_model, bert_analyzer, vader_analyzer,
                                              emotion_pipeline_all, sarcasm_pipeline)
                    overall = analyze_text_comprehensive(aspect_text, bert_analyzer, vader_analyzer,
                                                         emotion_pipeline_all, sarcasm_pipeline)

                    if overall:
                        st.markdown("#### Overall Sentiment")
                        show_new_results(overall, aspects, speech_emotion_results=None)
                        save_to_history(overall)
            else:
                st.warning("⚠️ Please enter some text")

    with tab4:
        st.markdown("### 🌐 Webpage Analysis")
        url_input = st.text_input("Enter a URL to scrape and analyze",
                                  placeholder="e.g., a news article or blog post URL")

        if st.button("🌐 Analyze URL", use_container_width=True, key="url_btn"):
            if url_input.strip():
                scraped_text = ""
                with st.spinner(f"Scraping text from {url_input}..."):
                    scraped_text = scrape_webpage_text(url_input)

                if scraped_text.startswith("❌"):
                    st.error(scraped_text)
                else:
                    st.success("✅ Scraping complete!")
                    with st.expander("View Scraped Text"):
                        st.text_area("Scraped Text", scraped_text, height=150, key="url_transcript",
                                     label_visibility="hidden")

                    with st.spinner("🔄 Analyzing text..."):
                        result = analyze_text_comprehensive(scraped_text, bert_analyzer, vader_analyzer,
                                                            emotion_pipeline_all, sarcasm_pipeline)
                        if result:
                            st.success("✅ Analysis Complete!")
                            show_new_results(result, aspects=None, speech_emotion_results=None)
                            save_to_history(result)
                        else:
                            st.error("Could not analyze the scraped text.")
            else:
                st.warning("⚠️ Please enter a URL")

elif st.session_state.page == "🎬 File Analysis":
    st.markdown("## 🎬 File Analysis (Video/Audio)")

    tab1, tab2, tab3 = st.tabs(["🎬 Upload Video", "🎵 Upload Audio", "🔗 YouTube URL"])

    with tab1:
        st.success("✅ *Recommended* - Upload your video file directly")

        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
            key="video_uploader"
        )

        if uploaded_video:
            col1, col2 = st.columns(2)

            with col1:
                st.video(uploaded_video)

            with col2:
                st.info(f"*File:* {uploaded_video.name}")
                st.info(f"*Size:* {uploaded_video.size / (1024 * 1024):.2f} MB")

                analyze_button_pressed = st.button("🎬 Analyze Video", use_container_width=True, key="analyze_video_btn")

            if analyze_button_pressed:
                try:
                    os.makedirs("temp_files", exist_ok=True)
                    tmp_path = os.path.join("temp_files", f"video_{int(time.time())}.mp4")

                    with st.spinner("💾 Processing..."):
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_video.read())

                    audio_path = ""  # Initialize
                    with st.spinner("🎵 Extracting audio..."):
                        audio_path = extract_audio_from_video(tmp_path)

                    if audio_path:
                        st.success("✅ Audio extracted")
                        st.audio(audio_path)
                    else:
                        st.error("Audio extraction failed. Cannot proceed.")
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                        st.stop()

                    transcript = ""  # Initialize
                    with st.spinner("📝 Transcribing with Whisper..."):
                        transcript = transcribe_audio(audio_path, whisper_model)

                    # --- PATCH: safe check usage here too ---
                    if isinstance(transcript, str) and not transcript.startswith("❌"):
                        st.success("✅ Transcription complete!")

                        with st.expander("📄 Transcript", expanded=True):
                            st.text_area("Transcript", transcript, height=200, key="transcript_display",
                                         label_visibility="hidden")
                            st.download_button(
                                "💾 Download",
                                transcript,
                                f"transcript_{int(time.time())}.txt",
                                key="download_transcript"
                            )

                        with st.spinner("🔄 Analyzing Text & Speech..."):
                            result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer,
                                                                emotion_pipeline_all, sarcasm_pipeline)
                            speech_emotions = analyze_speech_emotion(audio_path, ser_pipeline)

                            if result:
                                st.success("✅ Analysis complete!")
                                show_new_results(result, aspects=None, speech_emotion_results=speech_emotions)
                                save_to_history(result)
                                st.toast('Analysis complete!', icon='✅')
                            else:
                                st.error("Could not analyze the transcribed text.")
                    else:
                        st.error(transcript if isinstance(transcript,
                                                          str) else "❌ Error: Transcription returned unexpected type.")
                    # --- END PATCH ---

                    # Cleanup
                    for f in [tmp_path, audio_path]:
                        if f and os.path.exists(f):
                            os.remove(f)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.info("Upload an audio file (.mp3, .wav, .m4a) to transcribe and analyze.")

        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a'],
            key="audio_uploader"
        )

        if uploaded_audio:
            st.audio(uploaded_audio)

            analyze_button_pressed = st.button("🎵 Analyze Audio", use_container_width=True, key="analyze_audio_btn")

            if analyze_button_pressed:
                try:
                    os.makedirs("temp_files", exist_ok=True)
                    tmp_path = os.path.join("temp_files", uploaded_audio.name)

                    with st.spinner("💾 Processing..."):
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_audio.read())

                    transcript = ""
                    with st.spinner("📝 Transcribing with Whisper..."):
                        transcript = transcribe_audio(tmp_path, whisper_model)

                    # --- PATCH: safe check usage here too ---
                    if isinstance(transcript, str) and not transcript.startswith("❌"):
                        st.success("✅ Transcription complete!")
                        with st.expander("📄 Transcript", expanded=True):
                            st.text_area("Transcript", transcript, height=200, key="transcript_display_audio",
                                         label_visibility="hidden")
                            st.download_button(
                                "💾 Download",
                                transcript,
                                f"transcript_{int(time.time())}.txt",
                                key="download_transcript_audio"
                            )

                        with st.spinner("🔄 Analyzing Text & Speech..."):
                            result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer,
                                                                emotion_pipeline_all, sarcasm_pipeline)
                            speech_emotions = analyze_speech_emotion(tmp_path, ser_pipeline)

                            if result:
                                st.success("✅ Analysis complete!")
                                show_new_results(result, aspects=None, speech_emotion_results=speech_emotions)
                                save_to_history(result)
                                st.toast('Analysis complete!', icon='✅')
                            else:
                                st.error("Could not analyze the transcribed text.")
                    else:
                        st.error(transcript if isinstance(transcript,
                                                          str) else "❌ Error: Transcription returned unexpected type.")
                        # Cleanup
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    # --- END PATCH ---

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # REPLACE lines 777-898 in app.py with this:

    with tab3:
        st.info("🔗 YouTube: download audio, transcribe, and analyze.")

        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="yt_url_main")
        prefer_ext = st.selectbox("Audio format", ["m4a", "mp3"], index=0, key="yt_format_select")

        if youtube_url.strip():
            with st.spinner("Fetching video metadata..."):
                meta = fetch_youtube_metadata(youtube_url)
                if meta:
                    st.markdown(
                        f"**Title:** {meta.get('title')}  \n**Uploader:** {meta.get('uploader')}  \n**Duration (s):** {meta.get('duration')}")
                    if meta.get('thumbnail'):
                        st.image(meta.get('thumbnail'), width=240)
                else:
                    st.warning("Could not fetch metadata; proceeding anyway.")

        if st.button("📥 Download & Analyze", use_container_width=True, key="yt_download_btn"):
            if not youtube_url.strip():
                st.warning("Please enter a YouTube URL first.")
            else:
                audio_path = ""
                with st.spinner(f"Downloading audio as .{prefer_ext}... (This can take a while)"):
                    audio_path = download_youtube_audio_simple(youtube_url, prefer_ext=prefer_ext)

                if isinstance(audio_path, str) and audio_path.startswith("❌"):
                    st.error(audio_path)
                elif not audio_path or not os.path.exists(audio_path):
                    st.error("❌ Download failed. File path not found.")
                else:
                    st.success("✅ Audio downloaded successfully.")
                    st.audio(audio_path)

                    # Define a temp path for the converted WAV
                    # Use a timestamp to avoid conflicts
                    wav_path = os.path.join("temp_files", f"yt_audio_{int(time.time())}.wav")

                    with st.spinner("Converting audio to WAV..."):
                        conv_result = convert_to_wav_16k_mono(audio_path, wav_path)

                    if isinstance(conv_result, str) and conv_result.startswith("❌"):
                        st.error(conv_result)
                    else:
                        # Transcribe the WAV file
                        with st.spinner("Transcribing with Whisper..."):
                            transcript = transcribe_audio(wav_path, whisper_model)

                        if isinstance(transcript, str) and not transcript.startswith("❌"):
                            st.success("✅ Transcription complete")
                            with st.expander("Transcript", expanded=True):
                                st.text_area("Transcript", transcript, height=200, key="yt_transcript_area")
                                st.download_button("Download transcript", transcript, "transcript.txt",
                                                   key="yt_download_txt")

                            # Analyze
                            with st.spinner("Analyzing text and speech..."):
                                result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer,
                                                                    emotion_pipeline_all, sarcasm_pipeline)
                                speech_emotions = analyze_speech_emotion(wav_path, ser_pipeline)

                                if result:
                                    show_new_results(result, aspects=None, speech_emotion_results=speech_emotions)
                                    save_to_history(result)
                                else:
                                    st.error("❌ Analysis failed. See server logs.")
                        else:
                            st.error(transcript if isinstance(transcript, str) else "❌ Transcription failed.")

                    # Cleanup both temp files
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception as e:
                        print(f"Error during cleanup: {e}")
                        st.warning("Could not clean up temporary audio files.")


elif st.session_state.page == "🤖 Chatbot":
    st.markdown("### 🤖 AI Chatbot")
    st.info("Chat with our sentiment-aware AI assistant!")

    # Display chat history
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 {content}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("Your message:", key="chatbot_input",
                               placeholder="Ask me anything about sentiment analysis...")

    if st.button("Send", use_container_width=True, key="send_chat"):
        if user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get bot response
            with st.spinner("Thinking..."):
                bot_reply = chatbot_response(user_input, bert_analyzer, vader_analyzer, emotion_pipeline)

            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

            st.rerun()

elif st.session_state.page == "📚 History":
    st.markdown("### 📚 Analysis History")

    if not st.session_state.history:
        st.info("No analyses yet. Start analyzing to build your history!")
    else:
        # --- NEW CODE START: ACCURACY/CONFIDENCE CHART ---
        st.markdown("### 📈 Model Confidence Trend")
        st.caption("This chart shows how confident the AI was in its predictions over time.")

        # Prepare data for the chart
        history_df = pd.DataFrame(st.session_state.history)

        # Create a Line/Area chart for Confidence
        if 'confidence' in history_df.columns:
            # Add an index for the X-axis (1, 2, 3...)
            history_df['Analysis #'] = range(1, len(history_df) + 1)

            fig_conf = px.area(
                history_df,
                x='Analysis #',
                y='confidence',
                title='Confidence Score per Analysis',
                labels={'confidence': 'Confidence (0-1)'},
                markers=True,
                color_discrete_sequence=["#A79BFF"]  # Your purple theme color
            )

            # Fix the Y-axis range from 0 to 1 (0% to 100%)
            fig_conf.update_layout(yaxis_range=[0, 1.1])

            st.plotly_chart(fig_conf, use_container_width=True)

            # Calculate Average Confidence
            avg_conf = history_df['confidence'].mean()
            st.metric("🛡️ Average Model Confidence", f"{avg_conf:.1%}")
        # --- NEW CODE END ---

        st.markdown("---")
        st.success(f"Total Analyses: {len(st.session_state.history)}")

        # Display history in reverse order (newest first)
        for idx, record in enumerate(reversed(st.session_state.history)):
            with st.expander(
                    f"Analysis #{len(st.session_state.history) - idx} - {record.get('final_sentiment', 'N/A')} ({record.get('combined_score', 0):.2f})"):
                st.write(f"**Text:** {record.get('text', 'N/A')[:200]}...")
                st.write(f"**Sentiment:** {record.get('final_sentiment', 'N/A')}")
                st.write(f"**Score:** {record.get('combined_score', 0):.2f}")
                st.write(f"**Confidence:** {record.get('confidence', 0):.1%}")
                st.write(
                    f"**Top Emotion:** {max(record.get('emotions', {}).items(), key=lambda x: x[1])[0] if record.get('emotions') else 'N/A'}")

        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

elif st.session_state.page == "ℹ️ About":
    st.markdown("### ℹ️ About This Platform")

    st.markdown("""
    ## Advanced Sentiment Analysis Platform

    This application combines **state-of-the-art AI models** to provide comprehensive sentiment and emotion analysis.

    ---

    ### 🚀 Quick Start

    #### 1. Installation
    ```bash
    # Create and activate a virtual environment
    python3 -m venv venv
    .\ venv\Scripts\ activate

    # Install dependencies
    pip install -r requirements.txt

    # Download the SpaCy model
    python -m spacy download en_core_web_sm

    # Note: 'pyaudio' may require system libraries (like portaudio) first.
    # On Debian/Ubuntu: sudo apt-get install portaudio19-dev
    # On Mac: brew install portaudio
    ```

    #### 2. Run the App
    ```bash
    streamlit run app.py
    ```

    ---

    ### 🔬 Models Used:
    - **RoBERTa** - Twitter-trained sentiment classifier (80% weight)
    - **VADER** - Rule-based sentiment analysis (20% weight)
    - **SpaCy** - NLP for noun chunk extraction
    - **BERT** - 43-emotion classification
    - **Whisper** - OpenAI speech-to-text transcription
    - **Sarcasm Detection** - Irony and sarcasm classifier
    - **Speech Emotion Recognition** - Emotion detection from voice tone

    ### ✨ Key Features:
    - **Multi-Modal Analysis**: Combines text sentiment + speech emotion
    - **Hybrid Scoring**: 80% RoBERTa + 20% VADER for balanced results
    - **Aspect-Based**: Identifies specific aspects and their sentiments
    - **YouTube Support**: Download and analyze videos directly
    - **Voice Recording**: Real-time microphone transcription
    - **Webpage Scraping**: Analyze content from any URL

    ### 🎯 Use Cases:
    - Customer review analysis
    - Social media monitoring
    - Call center quality assessment
    - Content moderation
    - Market research

    ---

    **Version:** 2.0  
    **Built with:** Streamlit, Transformers, Whisper, SpaCy
    """)