import os
import re
import cv2
import yt_dlp
import spacy
import torch
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from typing import Dict, List
import streamlit as st
import requests
from bs4 import BeautifulSoup
import whisper


# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        # 1. BERT Sentiment Model
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        bert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # 2. VADER Sentiment Model
        vader = SentimentIntensityAnalyzer()

        # 3. SpaCy NLP Model
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        # 4. Emotion Model (for chatbot)
        emotion_pipeline = pipeline("text-classification",
                                    model="j-hartmann/emotion-english-distilroberta-base",
                                    return_all_scores=False)

        # 5. Emotion Model (for charts)
        emotion_pipeline_all = pipeline("text-classification",
                                        model="j-hartmann/emotion-english-distilroberta-base",
                                        return_all_scores=True)

        # 6. Whisper Transcription Model
        whisper_model = whisper.load_model("base")

        return bert_pipeline, vader, nlp, emotion_pipeline, emotion_pipeline_all, whisper_model

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None


# ----------------------------
# Analysis Functions
# ----------------------------
def analyze_text_comprehensive(text: str, bert_analyzer, vader_analyzer, emotion_pipeline_all) -> Dict:
    """Comprehensive sentiment analysis"""
    if not text or not text.strip():
        return None

    try:
        # --- 1. BERT Analysis (for 60% of score) ---
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

        # --- 2. VADER Analysis (for 40% of score) ---
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        if vader_compound >= 0.05:
            vader_sentiment = 'POSITIVE'
        elif vader_compound <= -0.05:
            vader_sentiment = 'NEGATIVE'
        else:
            vader_sentiment = 'NEUTRAL'

        vader_normalized = (vader_compound + 1) / 2

        # --- 3. Hybrid Score (60% BERT + 40% VADER) ---
        combined_score = (bert_normalized * 0.6) + (vader_normalized * 0.4)

        if combined_score >= 0.6:
            final_sentiment = 'POSITIVE'
        elif combined_score <= 0.4:
            final_sentiment = 'NEGATIVE'
        else:
            final_sentiment = 'NEUTRAL'

        # --- 4. NEW: Emotion Analysis ---
        emotion_results = emotion_pipeline_all(text[:512])[0]
        emotions = {e['label']: e['score'] * 100 for e in emotion_results}

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
        print(f"Analysis error: {str(e)}")
        return None


def extract_aspects(text: str, nlp_model, bert_analyzer, vader_analyzer, emotion_pipeline_all) -> List[Dict]:
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

        sentiment_result = analyze_text_comprehensive(context, bert_analyzer, vader_analyzer, emotion_pipeline_all)

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
    """Download YouTube video. Returns the path or None if all methods fail."""

    # Method 1
    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 'outtmpl': output_path, 'quiet': True, 'no_warnings': True,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.youtube.com/', 'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
                             'Accept-Language': 'en-us,en;q=0.5', 'Sec-Fetch-Mode': 'navigate'}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            if os.path.exists(output_path): return output_path
    except Exception as e:
        print(f"YT Download Method 1 failed: {e}")

    # Method 2
    try:
        output_path2 = "temp_yt_audio.m4a"
        ydl_opts = {
            'format': 'bestaudio/best', 'outtmpl': output_path2, 'quiet': True, 'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'prefer_ffmpeg': True, 'keepvideo': False,
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            for path in [output_path2, output_path2.replace('.m4a', '.mp3')]:
                if os.path.exists(path): return path
    except Exception as e:
        print(f"YT Download Method 2 failed: {e}")

    # Method 3
    try:
        ydl_opts = {
            'format': 'worstaudio/worst', 'outtmpl': output_path, 'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['android'], 'skip': ['hls', 'dash']}}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            if os.path.exists(output_path): return output_path
    except Exception as e:
        print(f"YT Download Method 3 failed: {e}")

    # Method 4
    try:
        ydl_opts = {'format': 'best', 'outtmpl': output_path, 'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            if os.path.exists(output_path): return output_path
    except Exception as e:
        print(f"YT Download Method 4 failed: {e}")

    return None


def chatbot_response(user_message: str, bert_analyzer, vader_analyzer, emotion_pipeline) -> str:
    """Generate sentiment-aware response"""

    dummy_all_pipeline = lambda x: [[{'label': 'neutral', 'score': 1.0}]]
    analysis = analyze_text_comprehensive(user_message, bert_analyzer, vader_analyzer, dummy_all_pipeline)

    if not analysis:
        return "I couldn't analyze that. Could you rephrase?"

    sentiment = analysis['final_sentiment']
    score = analysis['combined_score']

    if sentiment == 'POSITIVE':
        responses = [f"I'm glad to hear positive thoughts! Score: {score:.2f} 😊",
                     f"Your positivity is contagious! (Sentiment: {score:.2f})",
                     f"Great perspective! Score: {score:.2f} ✨"]
    elif sentiment == 'NEGATIVE':
        responses = [f"I sense concern (score: {score:.2f}). I'm here to help 💙",
                     f"Seems a bit down (score: {score:.2f}). How can I help?",
                     f"Negative sentiment detected ({score:.2f}). Let's work through it 🤝"]
    else:
        responses = [f"Neutral sentiment (score: {score:.2f}). What would you like to explore?",
                     f"I understand (Sentiment: {score:.2f}). How can I assist?",
                     f"Balanced message ({score:.2f}). What's next? 🤔"]

    import random
    response = random.choice(responses)

    dominant_emotion_result = emotion_pipeline(user_message)[0]
    dominant_emotion = dominant_emotion_result['label']
    emotion_score = dominant_emotion_result['score']

    emotion_emoji = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'surprise': '😲', 'neutral': '😐',
                     'disgust': '🤢'}
    response += f"\n\nDominant emotion: {dominant_emotion.title()} {emotion_emoji.get(dominant_emotion, '😐')} ({emotion_score:.0%})"

    return response


def extract_audio_from_video(video_file_path: str) -> str:
    """Extract audio from video"""
    try:
        audio_path = "temp_audio.wav"
        audio = AudioSegment.from_file(video_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        return None


def transcribe_audio(audio_path: str, whisper_model) -> str:
    """Transcribe audio using Whisper"""
    try:
        spinner_text = "Transcribing with Whisper... (This may take a moment)"
        st.spinner(spinner_text)

        result = whisper_model.transcribe(audio_path, fp16=False)

        if result and "text" in result:
            transcript = result["text"]
            if not transcript.strip():
                return "❌ Could not transcribe (no speech detected)."
            return transcript
        else:
            return "❌ Transcription failed (no result)."

    except Exception as e:
        return f"❌ Error during transcription: {str(e)}"


def scrape_webpage_text(url: str) -> str:
    """Scrape text from a webpage URL."""
    try:
        if not url.startswith('http'):
            url = 'https://' + url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])

        if not text_elements:
            text_elements = soup.find('body')
            if not text_elements:
                return "❌ Error: Could not find any text content on this page."

        full_text = ' '.join(elem.get_text(separator=' ', strip=True) for elem in text_elements)
        full_text = re.sub(r'\s+', ' ', full_text)

        return full_text

    except requests.exceptions.RequestException as e:
        return f"❌ Error: Could not fetch the URL. {e}"
    except Exception as e:
        return f"❌ Error: An unknown error occurred. {e}"
