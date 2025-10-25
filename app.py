import os

os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import time
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
import cv2
import yt_dlp
import spacy
from collections import Counter
import re

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SentiAI - Advanced Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-right: 1px solid #334155; }
    .hero-title { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; text-align: center; }
    .hero-subtitle { font-size: 1.2rem; color: #94a3b8; text-align: center; margin-bottom: 2rem; }
    .feature-badges { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin: 2rem 0; }
    .badge { background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; color: #6366f1; padding: 0.5rem 1rem; border-radius: 50px; font-size: 0.9rem; font-weight: 500; }
    .sentiment-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 2rem; text-align: center; color: white; margin: 1rem 0; }
    .sentiment-score { font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; }
    .sentiment-label { font-size: 1.3rem; text-transform: uppercase; letter-spacing: 2px; }
    .stButton>button { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600; transition: all 0.3s ease; width: 100%; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3); }
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
@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        bert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        vader = SentimentIntensityAnalyzer()

        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        return bert_pipeline, vader, nlp
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


bert_analyzer, vader_analyzer, nlp_model = load_models()

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------
# Analysis Functions
# ----------------------------
def analyze_text_comprehensive(text: str) -> Dict:
    """Comprehensive sentiment analysis"""
    if not text or not text.strip():
        return None

    try:
        bert_result = bert_analyzer(text[:512])[0]
        bert_label = bert_result['label']
        bert_score = float(bert_result['score'])

        if '5 stars' in bert_label or '4 stars' in bert_label:
            bert_sentiment = 'POSITIVE'
            bert_normalized = 0.7 + (bert_score * 0.3)
        elif '3 stars' in bert_label:
            bert_sentiment = 'NEUTRAL'
            bert_normalized = 0.4 + (bert_score * 0.2)
        else:
            bert_sentiment = 'NEGATIVE'
            bert_normalized = bert_score * 0.4

        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        if vader_compound >= 0.05:
            vader_sentiment = 'POSITIVE'
        elif vader_compound <= -0.05:
            vader_sentiment = 'NEGATIVE'
        else:
            vader_sentiment = 'NEUTRAL'

        vader_normalized = (vader_compound + 1) / 2
        combined_score = (bert_normalized * 0.6) + (vader_normalized * 0.4)

        if combined_score >= 0.6:
            final_sentiment = 'POSITIVE'
        elif combined_score <= 0.4:
            final_sentiment = 'NEGATIVE'
        else:
            final_sentiment = 'NEUTRAL'

        emotions = {
            'joy': max(0, vader_scores['pos'] * 100),
            'sadness': max(0, vader_scores['neg'] * 100),
            'anger': max(0, (vader_scores['neg'] * 0.7) * 100),
            'fear': max(0, (vader_scores['neg'] * 0.3) * 100),
            'surprise': abs(vader_scores['neu'] * 50),
            'trust': max(0, vader_scores['pos'] * 80)
        }

        confidence = (bert_score + abs(vader_compound)) / 2

        return {
            'text': text,
            'final_sentiment': final_sentiment,
            'combined_score': combined_score,
            'bert_sentiment': bert_sentiment,
            'bert_score': bert_score,
            'bert_label': bert_label,
            'vader_sentiment': vader_sentiment,
            'vader_compound': vader_compound,
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'emotions': emotions,
            'confidence': confidence,
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


def extract_aspects(text: str) -> List[Dict]:
    """Extract aspects and sentiments"""
    if not nlp_model:
        return []

    doc = nlp_model(text)
    aspects = []

    for chunk in doc.noun_chunks:
        aspect_text = chunk.text.lower()
        start_idx = max(0, chunk.start - 5)
        end_idx = min(len(doc), chunk.end + 5)
        context = doc[start_idx:end_idx].text

        sentiment_result = analyze_text_comprehensive(context)

        if sentiment_result:
            aspects.append({
                'aspect': aspect_text,
                'sentiment': sentiment_result['final_sentiment'],
                'score': sentiment_result['combined_score'],
                'context': context
            })

    seen = set()
    unique_aspects = []
    for aspect in aspects:
        if aspect['aspect'] not in seen and len(aspect['aspect'].split()) <= 3:
            seen.add(aspect['aspect'])
            unique_aspects.append(aspect)

    return unique_aspects[:10]


def download_youtube_video(url: str, output_path: str = "temp_yt_video.mp4") -> str:
    """Download YouTube video with multiple fallback methods"""

    # Method 1: Try with cookies and authentication
    try:
        st.info("🔄 Attempting Method 1: Standard download...")
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info and os.path.exists(output_path):
                st.success("✅ Method 1 succeeded!")
                return output_path
    except Exception as e:
        st.warning(f"⚠️ Method 1 failed: {str(e)[:100]}")

    # Method 2: Try audio only with different format
    try:
        st.info("🔄 Attempting Method 2: Audio-only download...")
        output_path2 = "temp_yt_audio.m4a"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path2,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Check for both possible output names
            possible_outputs = [output_path2, output_path2.replace('.m4a', '.mp3')]
            for path in possible_outputs:
                if os.path.exists(path):
                    st.success("✅ Method 2 succeeded!")
                    return path
    except Exception as e:
        st.warning(f"⚠️ Method 2 failed: {str(e)[:100]}")

    # Method 3: Try with alternate client
    try:
        st.info("🔄 Attempting Method 3: Alternate client...")
        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': output_path,
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                    'skip': ['hls', 'dash']
                }
            },
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info and os.path.exists(output_path):
                st.success("✅ Method 3 succeeded!")
                return output_path
    except Exception as e:
        st.warning(f"⚠️ Method 3 failed: {str(e)[:100]}")

    # Method 4: Try with minimal options
    try:
        st.info("🔄 Attempting Method 4: Minimal config...")
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info and os.path.exists(output_path):
                st.success("✅ Method 4 succeeded!")
                return output_path
    except Exception as e:
        st.warning(f"⚠️ Method 4 failed: {str(e)[:100]}")

    st.error("❌ All download methods failed")
    return None


def chatbot_response(user_message: str) -> str:
    """Generate sentiment-aware response"""
    analysis = analyze_text_comprehensive(user_message)

    if not analysis:
        return "I couldn't analyze that. Could you rephrase?"

    sentiment = analysis['final_sentiment']
    score = analysis['combined_score']

    if sentiment == 'POSITIVE':
        responses = [
            f"I'm glad to hear positive thoughts! Score: {score:.2f} 😊",
            f"Your positivity is contagious! (Sentiment: {score:.2f})",
            f"Great perspective! Score: {score:.2f} ✨"
        ]
    elif sentiment == 'NEGATIVE':
        responses = [
            f"I sense concern (score: {score:.2f}). I'm here to help 💙",
            f"Seems a bit down (score: {score:.2f}). How can I help?",
            f"Negative sentiment detected ({score:.2f}). Let's work through it 🤝"
        ]
    else:
        responses = [
            f"Neutral sentiment (score: {score:.2f}). What would you like to explore?",
            f"I understand (Sentiment: {score:.2f}). How can I assist?",
            f"Balanced message ({score:.2f}). What's next? 🤔"
        ]

    import random
    response = random.choice(responses)

    dominant_emotion = max(analysis['emotions'], key=analysis['emotions'].get)
    emotion_emoji = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'surprise': '😲', 'trust': '🤝'}
    response += f"\n\nDominant emotion: {dominant_emotion.title()} {emotion_emoji.get(dominant_emotion, '😐')}"

    return response


def recognize_speech() -> str:
    """Speech recognition"""
    try:
        import pyaudio
    except ImportError:
        return "❌ PyAudio not installed. Install with: pip install pyaudio"

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        return r.recognize_google(audio)
    except Exception as e:
        return f"❌ Error: {str(e)}"


def extract_audio_from_video(video_file_path: str) -> str:
    """Extract audio from video"""
    try:
        audio_path = "temp_audio.wav"
        audio = AudioSegment.from_file(video_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        st.error(f"Audio extraction error: {str(e)}")
        return None


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio"""
    r = sr.Recognizer()
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 30000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        full_transcript = []
        for idx, chunk in enumerate(chunks):
            chunk_path = f"temp_chunk_{idx}.wav"
            chunk.export(chunk_path, format="wav")

            try:
                with sr.AudioFile(chunk_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='en-US')
                if text:
                    full_transcript.append(text)
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except:
                continue

        return " ".join(full_transcript) if full_transcript else "❌ Could not transcribe"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ----------------------------
# Display Functions
# ----------------------------
def display_analysis_results(result: Dict):
    """Display analysis results"""
    if not result:
        return

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    sentiment_emoji = {'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐'}

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="sentiment-card">
            <div class="sentiment-score">{result['combined_score']:.2f}</div>
            <div class="sentiment-label">{sentiment_emoji.get(result['final_sentiment'], '😐')} {result['final_sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("BERT Score", f"{result['bert_score']:.2f}", result['bert_sentiment'])
        st.metric("Word Count", result['word_count'])

    with col3:
        st.metric("VADER Score", f"{result['vader_compound']:.2f}", result['vader_sentiment'])
        st.metric("Confidence", f"{result['confidence']:.2%}")

    st.markdown("### 😊 Emotion Analysis")
    emotion_cols = st.columns(6)
    emotion_icons = ['😄', '😢', '😠', '😨', '😲', '🤝']
    emotion_names = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Trust']

    for col, icon, name, key in zip(emotion_cols, emotion_icons, emotion_names, result['emotions'].keys()):
        with col:
            value = result['emotions'][key]
            st.markdown(f"""
            <div class="emotion-card">
                <div class="emotion-icon">{icon}</div>
                <div class="emotion-name">{name}</div>
                <div class="emotion-value">{value:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(value / 100)

    with st.expander("🔍 Detailed Breakdown"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*BERT Analysis*")
            st.write(f"- Label: {result['bert_label']}")
            st.write(f"- Sentiment: {result['bert_sentiment']}")
            st.write(f"- Confidence: {result['bert_score']:.2%}")
        with col2:
            st.markdown("*VADER Analysis*")
            st.write(f"- Compound: {result['vader_compound']:.3f}")
            st.write(f"- Positive: {result['vader_pos']:.2%}")
            st.write(f"- Negative: {result['vader_neg']:.2%}")
            st.write(f"- Neutral: {result['vader_neu']:.2%}")


def display_aspect_analysis(aspects: List[Dict]):
    """Display aspect analysis"""
    st.markdown("### 🎯 Aspect-Based Analysis")

    if not aspects:
        st.info("No specific aspects detected.")
        return

    for aspect in aspects:
        sentiment_class = f"aspect-{aspect['sentiment'].lower()}"
        sentiment_emoji = {'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐'}

        st.markdown(f"""
        <div class="aspect-card {sentiment_class}">
            <strong>📌 {aspect['aspect'].title()}</strong><br>
            Sentiment: {sentiment_emoji.get(aspect['sentiment'], '😐')} {aspect['sentiment']} (Score: {aspect['score']:.2f})<br>
            <em>Context: "{aspect['context'][:100]}..."</em>
        </div>
        """, unsafe_allow_html=True)


def save_to_history(result: Dict):
    """Save to history"""
    if result:
        st.session_state.history.append(result)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🧠 SentiAI Dashboard")
    st.markdown("*Advanced Sentiment Analysis*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Analyzer", "🎬 Video Analysis", "🤖 Chatbot", "📚 History", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Analyses", len(st.session_state.history))

    if st.session_state.history:
        avg_score = np.mean([h['combined_score'] for h in st.session_state.history])
        st.metric("Avg Score", f"{avg_score:.2f}")

    st.markdown("---")
    st.caption("Built with 🧠 BERT + VADER • SpaCy")

# ----------------------------
# Pages
# ----------------------------
if page == "🏠 Home":
    st.markdown('<div class="hero-title">🧠 SentiAI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced Multi-Modal Sentiment Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-badges">
        <span class="badge">🤖 BERT + VADER</span>
        <span class="badge">🎬 Video Analysis</span>
        <span class="badge">🎯 Aspect Analysis</span>
        <span class="badge">💬 AI Chatbot</span>
        <span class="badge">🎤 Voice Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📝 Text Analysis\nDual-model AI sentiment detection")
    with col2:
        st.markdown("### 🎬 Video Analysis\nTranscribe and analyze videos")
    with col3:
        st.markdown("### 🤖 Chatbot\nEmotion-aware conversations")

    st.info("👉 Navigate to any section to start analyzing!")

elif page == "🔍 Analyzer":
    st.markdown("## 🔍 Advanced Sentiment Analyzer")

    tab1, tab2, tab3 = st.tabs(["📝 Text", "🎤 Voice", "🎯 Aspects"])

    with tab1:
        st.markdown("### Text Analysis")
        text_input = st.text_area("Enter text", height=200, key="text_analysis_input",
                                  placeholder="Type or paste your text here...")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Analyze Text", use_container_width=True, key="analyze_text_btn"):
                if text_input.strip():
                    with st.spinner("🔄 Analyzing..."):
                        result = analyze_text_comprehensive(text_input)
                        if result:
                            st.success("✅ Complete!")
                            display_analysis_results(result)
                            save_to_history(result)
                else:
                    st.warning("⚠️ Please enter some text")

        with col2:
            if st.button("💡 Example", use_container_width=True, key="example_btn"):
                st.info("Example: 'I love this product! It's amazing and exceeded my expectations.'")

    with tab2:
        st.markdown("### 🎤 Voice Recording")

        try:
            import pyaudio

            pyaudio_available = True
        except ImportError:
            pyaudio_available = False

        if not pyaudio_available:
            st.warning("⚠️ *Voice Recording Not Available*")
            st.info(
                "Install PyAudio: pip install pyaudio (or brew install portaudio && pip install pyaudio on Mac)")
        else:
            st.info("🎤 Click to record your voice")
            if st.button("🎙️ Start Recording", use_container_width=True, key="record_btn"):
                with st.spinner("Recording..."):
                    transcript = recognize_speech()
                    if not transcript.startswith("❌"):
                        st.success(f"✅ Transcribed: {transcript}")
                        result = analyze_text_comprehensive(transcript)
                        if result:
                            display_analysis_results(result)
                            save_to_history(result)
                    else:
                        st.error(transcript)

    with tab3:
        st.markdown("### 🎯 Aspect-Based Analysis")
        aspect_text = st.text_area("Enter text with multiple aspects", height=200, key="aspect_input",
                                   placeholder="e.g., 'The camera is great, but the battery life is disappointing.'")

        if st.button("🔍 Analyze Aspects", use_container_width=True, key="aspect_btn"):
            if aspect_text.strip():
                with st.spinner("Extracting aspects..."):
                    aspects = extract_aspects(aspect_text)
                    overall = analyze_text_comprehensive(aspect_text)

                    if overall:
                        st.markdown("#### Overall Sentiment")
                        display_analysis_results(overall)
                        st.markdown("---")
                        display_aspect_analysis(aspects)
                        save_to_history(overall)
            else:
                st.warning("⚠️ Please enter some text")

elif page == "🎬 Video Analysis":
    st.markdown("## 🎬 Video Analysis")

    tab1, tab2 = st.tabs(["📤 Upload Video", "🔗 YouTube URL"])

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

                if st.button("🎬 Analyze Video", use_container_width=True, key="analyze_video_btn"):
                    try:
                        os.makedirs("temp_files", exist_ok=True)
                        tmp_path = os.path.join("temp_files", f"video_{int(time.time())}.mp4")

                        with st.spinner("💾 Processing..."):
                            with open(tmp_path, "wb") as f:
                                f.write(uploaded_video.read())

                        with st.spinner("🎵 Extracting audio..."):
                            audio_path = extract_audio_from_video(tmp_path)
                            if audio_path:
                                st.success("✅ Audio extracted")
                                st.audio(audio_path)

                        if audio_path:
                            with st.spinner("📝 Transcribing..."):
                                transcript = transcribe_audio(audio_path)

                                if not transcript.startswith("❌"):
                                    st.success("✅ Transcription complete!")

                                    with st.expander("📄 Transcript", expanded=True):
                                        st.text_area("", transcript, height=200, key="transcript_display")
                                        st.download_button(
                                            "💾 Download",
                                            transcript,
                                            f"transcript_{int(time.time())}.txt",
                                            key="download_transcript"
                                        )

                                    result = analyze_text_comprehensive(transcript)
                                    if result:
                                        display_analysis_results(result)
                                        save_to_history(result)
                                        st.balloons()
                                else:
                                    st.error(transcript)

                        # Cleanup
                        for f in [tmp_path, audio_path]:
                            if f and os.path.exists(f):
                                os.remove(f)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.info("🔗 Try YouTube direct download - Multiple methods will be attempted")

        st.markdown("""
        ### 💡 Tips for Success:
        - Update yt-dlp first: pip install --upgrade yt-dlp
        - Try shorter videos (under 10 minutes)
        - Educational/tutorial videos work better
        - Avoid music videos or copyrighted content
        """)

        yt_url = st.text_input("YouTube URL",
                               placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                               key="yt_url")

        if st.button("🚀 Download & Analyze", use_container_width=True, key="yt_btn"):
            if yt_url:
                st.info("⏳ Trying multiple download methods... Please wait...")

                try:
                    with st.spinner("📥 Downloading (this may take 1-2 minutes)..."):
                        video_path = download_youtube_video(yt_url)

                        if not video_path or not os.path.exists(video_path):
                            st.error("❌ All download methods failed")
                            st.markdown("""
                            ### 😔 YouTube Download Failed

                            *Why this happens:*
                            - YouTube actively blocks automated downloads
                            - Video may be region-restricted
                            - Copyright protection
                            - Rate limiting

                            ### ✅ *What to do now:*

                            *Quick Solution (2 minutes):*
                            1. Go to [Y2Mate.com](https://y2mate.com)
                            2. Paste your YouTube URL
                            3. Download the video
                            4. Use the *"Upload Video"* tab above ⬆️
                            5. Upload your downloaded file

                            *Or update yt-dlp:*
                            bash
                            pip install --upgrade yt-dlp

                            Then restart the app and try again.
                            """)
                        else:
                            st.success(f"✅ Successfully downloaded!")
                            st.balloons()

                            file_size = os.path.getsize(video_path) / (1024 * 1024)
                            st.info(f"📊 File size: {file_size:.2f} MB")

                            with st.spinner("🎵 Extracting audio..."):
                                audio_path = extract_audio_from_video(video_path)
                                if audio_path:
                                    st.success("✅ Audio extracted")
                                    st.audio(audio_path)

                                    with st.spinner("📝 Transcribing... This may take a while..."):
                                        transcript = transcribe_audio(audio_path)

                                        if not transcript.startswith("❌"):
                                            st.success("✅ Transcription complete!")
                                            st.balloons()

                                            with st.expander("📄 View Full Transcript", expanded=True):
                                                st.text_area("Transcript", transcript, height=200, key="yt_transcript")
                                                st.info(
                                                    f"📊 Words: {len(transcript.split())} | Characters: {len(transcript)}")

                                                st.download_button(
                                                    "💾 Download Transcript",
                                                    transcript,
                                                    f"transcript_{int(time.time())}.txt",
                                                    "text/plain",
                                                    key="yt_download_transcript"
                                                )

                                            with st.spinner("🔄 Analyzing sentiment..."):
                                                result = analyze_text_comprehensive(transcript)
                                                if result:
                                                    st.success("✅ Analysis complete!")
                                                    display_analysis_results(result)
                                                    save_to_history(result)
                                        else:
                                            st.error(transcript)
                                            st.warning("Video may not contain clear speech")
                                else:
                                    st.error("❌ Audio extraction failed")

                            # Cleanup
                            for f in [video_path, audio_path]:
                                if f and os.path.exists(f):
                                    try:
                                        os.remove(f)
                                    except:
                                        pass

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Use the 'Upload Video' tab instead - it always works!")
            else:
                st.warning("⚠️ Please enter a YouTube URL")

elif page == "🤖 Chatbot":
    st.markdown("## 🤖 Sentiment-Aware Chatbot")
    st.info("Chat with AI that understands your emotions!")

    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message">👤 {chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 {chat["message"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input("Your message:", key="chat_input")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("💬 Send", use_container_width=True, key="send_btn"):
            if user_input.strip():
                st.session_state.chat_history.append({'role': 'user', 'message': user_input})
                bot_reply = chatbot_response(user_input)
                st.session_state.chat_history.append({'role': 'bot', 'message': bot_reply})
                st.rerun()

    with col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

elif page == "📚 History":
    st.markdown("## 📚 Analysis History")

    if not st.session_state.history:
        st.info("No history yet. Start analyzing!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
                st.session_state.history = []
                st.rerun()
        with col2:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                               "text/csv", use_container_width=True, key="download_csv")

        st.markdown("---")

        for idx, analysis in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{len(st.session_state.history) - idx} - {analysis['final_sentiment']}"):
                st.write(f"*Text:* {analysis['text'][:200]}...")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment", analysis['final_sentiment'])
                with col2:
                    st.metric("Score", f"{analysis['combined_score']:.2f}")
                with col3:
                    st.metric("Confidence", f"{analysis['confidence']:.2%}")

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About SentiAI")

    st.markdown("""
    ### 🚀 Features

    1. ✅ Text Sentiment Analysis (BERT + VADER)
    2. ✅ Voice Recording & Transcription
    3. ✅ Video Analysis
    4. ✅ YouTube Video Analysis
    5. ✅ Aspect-Based Sentiment Analysis
    6. ✅ AI Chatbot with Emotion Detection
    7. ✅ Analysis History & Export

    ### 📦 Installation

    bash
    pip install "numpy<2.0" protobuf==3.20.3
    pip install torch transformers streamlit vaderSentiment
    pip install SpeechRecognition pydub opencv-python
    pip install yt-dlp spacy plotly pandas
    python -m spacy download en_core_web_sm


    ### 🚀 Run

    bash
    streamlit run app.py


    ### 🎯 How It Works

    - *BERT* (60%): Deep contextual understanding
    - *VADER* (40%): Lexicon-based analysis
    - *Combined*: Weighted average with confidence metrics

    ### 👥 Project

    - *Type:* Final Year Project
    - *Status:* Production Ready ✅
    """)

    st.success("✅ All features working perfectly!")