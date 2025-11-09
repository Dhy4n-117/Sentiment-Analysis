import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from typing import Dict, List

# Import all logic functions
from analysis_logic import (
    load_models,
    analyze_text_comprehensive,
    extract_aspects,
    download_youtube_video,
    chatbot_response,
    recognize_speech,
    extract_audio_from_video,
    transcribe_audio,
    scrape_webpage_text
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Advanced Sentiment Analysis",
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
# We now load models from our logic file
bert_analyzer, vader_analyzer, nlp_model = load_models()
if not bert_analyzer:
    st.error("Fatal Error: Could not load AI models. The app cannot start.")
    st.stop()

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------
# Display Functions (UI Logic)
# ----------------------------
def create_sentiment_gauge(score, sentiment_label):
    """
    Creates a Plotly gauge chart for the sentiment score.
    Score is assumed to be from 0 to 1.
    """

    # Determine gauge color AND title color
    if sentiment_label == 'POSITIVE':
        gauge_color = "#4CAF50"  # Green
        title_color = "#4CAF50"
    elif sentiment_label == 'NEGATIVE':
        gauge_color = "#F44336"  # Red
        title_color = "#F44336"
    else:  # NEUTRAL
        gauge_color = "#FBBC05"  # Yellow
        title_color = "#FBBC05"

    # Create the title text with HTML for coloring
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


def show_new_results(result: Dict, aspects: List[Dict] = None):
    """
    Display analysis results in a clean one-page layout.
    """
    if not result:
        st.error("No analysis result to display.")
        return

    if aspects is None:
        with st.spinner("Extracting aspects..."):
            aspects = extract_aspects(result['text'], nlp_model, bert_analyzer, vader_analyzer)

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

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

    st.divider()

    st.subheader("😊 Emotion Analysis")
    try:
        emotion_data = {label.title(): score for label, score in result['emotions'].items() if score > 0}
        if emotion_data:
            df_emotions = pd.DataFrame(emotion_data.items(), columns=['Emotion', 'Score (%)'])
            fig = px.bar(df_emotions, x='Emotion', y='Score (%)', color='Emotion',
                         title="Detected Emotions", text='Score (%)')
            fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distinct emotions detected.")
    except Exception as e:
        st.error(f"Could not generate emotion chart: {e}")
        st.write(result['emotions'])

    st.divider()

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

    st.divider()

    st.subheader("🔍 Detailed Breakdown")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", result['word_count'])
        st.metric("Confidence", f"{result['confidence']:.2%}")
    with col2:
        st.markdown("*BERT Analysis*")
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
        st.markdown("*VADER Analysis*")
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
    st.markdown("### 🧠 Dashboard")
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
    st.markdown('<div class="hero-title">🧠 Sentiment Analysis Platform</div>', unsafe_allow_html=True)
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
                        result = analyze_text_comprehensive(text_input, bert_analyzer, vader_analyzer)
                        if result:
                            st.success("✅ Complete!")
                            show_new_results(result)
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
                        with st.spinner("🔄 Analyzing..."):
                            result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer)
                            if result:
                                show_new_results(result)
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
                    aspects = extract_aspects(aspect_text, nlp_model, bert_analyzer, vader_analyzer)
                    overall = analyze_text_comprehensive(aspect_text, bert_analyzer, vader_analyzer)

                    if overall:
                        st.markdown("#### Overall Sentiment")
                        show_new_results(overall, aspects)
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
                        st.text_area("", scraped_text, height=150)

                    with st.spinner("🔄 Analyzing text..."):
                        result = analyze_text_comprehensive(scraped_text, bert_analyzer, vader_analyzer)
                        if result:
                            st.success("✅ Analysis Complete!")
                            show_new_results(result)
                            save_to_history(result)
                        else:
                            st.error("Could not analyze the scraped text.")
            else:
                st.warning("⚠️ Please enter a URL")

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

                analyze_button_pressed = st.button("🎬 Analyze Video", use_container_width=True, key="analyze_video_btn")

            if analyze_button_pressed:
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
                        else:
                            st.error("Audio extraction failed. Cannot proceed.")
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                            st.stop()

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

                            with st.spinner("🔄 Analyzing..."):
                                result = analyze_text_comprehensive(transcript, bert_analyzer, vader_analyzer)
                                if result:
                                    show_new_results(result)
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
                                                result = analyze_text_comprehensive(transcript, bert_analyzer,
                                                                                    vader_analyzer)
                                                if result:
                                                    st.success("✅ Analysis complete!")
                                                    show_new_results(result)
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
                bot_reply = chatbot_response(user_input, bert_analyzer, vader_analyzer)
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
    st.markdown("## ℹ️ About This Platform")

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
