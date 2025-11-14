import os
import re
import cv2
import yt_dlp
import spacy
import torch
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from typing import Dict, List
import streamlit as st
import requests
from bs4 import BeautifulSoup
import whisper
import numpy as np
import librosa
import traceback

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        # 1. NEW: RoBERTa Sentiment Model
        sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

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
                                    model="borisn70/bert-43-multilabel-emotion-detection",
                                    top_k=1)

        # 5. Emotion Model (for charts)
        emotion_pipeline_all = pipeline("text-classification",
                                        model="borisn70/bert-43-multilabel-emotion-detection",
                                        top_k=None)

        # 6. Whisper Transcription Model
        whisper_model = whisper.load_model("small") # tiny, base, small, medium, and large

        # 7. Sarcasm Detection Model
        sarcasm_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

        # 8. NEW: Speech Emotion Recognition (SER) Model
        
        try:
            ser_pipeline = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        except Exception as e:
            print("Speech Emotion Recognition failed to load:", e)
            ser_pipeline = None


        return sentiment_pipeline, vader, nlp, emotion_pipeline, emotion_pipeline_all, whisper_model, sarcasm_pipeline, ser_pipeline

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None, None, None


# ----------------------------
# Analysis Functions
# ----------------------------
def analyze_text_comprehensive(text: str, bert_analyzer, vader_analyzer, emotion_pipeline_all,
                               sarcasm_pipeline) -> Dict:
    """Comprehensive sentiment analysis"""
    if not text or not text.strip():
        return None

    try:
        # --- 1. RoBERTa Analysis (for 80% of score) ---
        bert_result = bert_analyzer(text[:512])[0]
        bert_label = bert_result['label']
        bert_score = float(bert_result['score'])

        if bert_label == 'Positive':
            bert_sentiment = 'POSITIVE'
            bert_normalized = (0.5 + bert_score / 2)
        elif bert_label == 'Negative':
            bert_sentiment = 'NEGATIVE'
            bert_normalized = (0.5 - bert_score / 2)
        else:  # Neutral
            bert_sentiment = 'NEUTRAL'
            bert_normalized = 0.5

        # --- 2. VADER Analysis (for 20% of score) ---
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        if vader_compound >= 0.05:
            vader_sentiment = 'POSITIVE'
        elif vader_compound <= -0.05:
            vader_sentiment = 'NEGATIVE'
        else:
            vader_sentiment = 'NEUTRAL'

        vader_normalized = (vader_compound + 1) / 2

        # --- 3. Hybrid Score (80/20 SPLIT) ---
        combined_score = (bert_normalized * 0.8) + (vader_normalized * 0.2)

        if combined_score >= 0.6:
            final_sentiment = 'POSITIVE'
        elif combined_score <= 0.4:
            final_sentiment = 'NEGATIVE'
        else:
            final_sentiment = 'NEUTRAL'

        # --- 4. Emotion Analysis (43-label model) ---
        emotion_results = emotion_pipeline_all(text[:512])[0]
        emotions = {e['label']: e['score'] * 100 for e in emotion_results}

        # --- 5. Sarcasm Analysis ---
        sarcasm_result = sarcasm_pipeline(text[:512])[0]
        sarcasm_label = sarcasm_result['label']
        if sarcasm_label.lower() == 'irony':
            sarcasm_score = sarcasm_result['score']
        else:
            sarcasm_score = 1 - sarcasm_result['score']

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
            'sarcasm_score': sarcasm_score,
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None


def extract_aspects(text: str, nlp_model, bert_analyzer, vader_analyzer, emotion_pipeline_all, sarcasm_pipeline) -> \
List[Dict]:
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

        sentiment_result = analyze_text_comprehensive(context, bert_analyzer, vader_analyzer, emotion_pipeline_all,
                                                      sarcasm_pipeline)

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
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 'outtmpl': output_path2, 'quiet': True, 'no_warnings': True,
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

    # 1. RoBERTa Analysis
    bert_result = bert_analyzer(user_message[:512])[0]
    bert_label = bert_result['label']
    bert_score = float(bert_result['score'])
    if bert_label == 'Positive':
        bert_normalized = (0.5 + bert_score / 2)
    elif bert_label == 'Negative':
        bert_normalized = (0.5 - bert_score / 2)
    else:
        bert_normalized = 0.5

    # 2. VADER Analysis
    vader_scores = vader_analyzer.polarity_scores(user_message)
    vader_compound = vader_scores['compound']
    vader_normalized = (vader_compound + 1) / 2

    # 3. Hybrid Score (80/20)
    combined_score = (bert_normalized * 0.8) + (vader_normalized * 0.2)

    if combined_score >= 0.6:
        final_sentiment = 'POSITIVE'
    elif combined_score <= 0.4:
        final_sentiment = 'NEGATIVE'
    else:
        final_sentiment = 'NEUTRAL'

    if final_sentiment == 'POSITIVE':
        responses = [f"I'm glad to hear positive thoughts! Score: {combined_score:.2f} 😊",
                     f"Your positivity is contagious! (Sentiment: {combined_score:.2f})",
                     f"Great perspective! Score: {combined_score:.2f} ✨"]
    elif final_sentiment == 'NEGATIVE':
        responses = [f"I sense concern (score: {combined_score:.2f}). I'm here to help 💙",
                     f"Seems a bit down (score: {combined_score:.2f}). How can I help?",
                     f"Negative sentiment detected ({combined_score:.2f}). Let's work through it 🤝"]
    else:
        responses = [f"Neutral sentiment (score: {combined_score:.2f}). What would you like to explore?",
                     f"I understand (Sentiment: {combined_score:.2f}). How can I assist?",
                     f"Balanced message ({combined_score:.2f}). What's next? 🤔"]

    import random
    response = random.choice(responses)

    dominant_emotion_result = emotion_pipeline(user_message)[0]
    dominant_emotion = dominant_emotion_result['label']
    emotion_score = dominant_emotion_result['score']

    emotion_emoji = {
        'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'surprise': '😲', 'neutral': '😐', 'disgust': '🤢',
        'admiration': '🌟', 'amusement': '😄', 'caring': '❤️', 'desire': '🔥', 'excitement': '🎉', 'gratitude': '🙏',
        'love': '❤️', 'optimism': '👍', 'relief': '😌', 'disappointment': '😞', 'remorse': '😔'
    }
    response += f"\n\nDominant emotion: {dominant_emotion.title()} {emotion_emoji.get(dominant_emotion, '😐')} ({emotion_score:.0%})"

    return response


def recognize_speech(spinner=None, whisper_model=None) -> str:
    """
    Debugging wrapper for microphone -> whisper transcription.
    Prints detailed debug info and full traceback on error so we know the exact failing line.
    Always returns a string (transcript or ❌-prefixed error).
    """
    try:
        # sanity check
        if whisper_model is None:
            return "❌ Error: whisper_model is None. Make sure load_models() returned a model and it was passed."

        # list devices for debug
        try:
            print("DEBUG: speech_recognition version:", sr.__version__)
            names = sr.Microphone.list_microphone_names()
            print("DEBUG: Available microphones:", names)
        except Exception as e:
            print("DEBUG: Could not list microphones:", e)

        r = sr.Recognizer()
        tmp_file = "temp_mic_audio.wav"

        # open mic and listen
        try:
            if spinner is not None:
                try:
                    spinner.text = "🎤 Listening..."
                except Exception:
                    pass
            with sr.Microphone() as source:
                print("DEBUG: opened Microphone", source)
                r.adjust_for_ambient_noise(source, duration=1)
                print("DEBUG: adjusted for ambient noise, starting listen()")
                audio = r.listen(source, timeout=15, phrase_time_limit=30)
                print("DEBUG: listen() returned, audio bytes:", len(audio.get_wav_data()))
            # save
            with open(tmp_file, "wb") as f:
                f.write(audio.get_wav_data())
            print("DEBUG: saved microphone audio to", tmp_file)
        except Exception as mic_e:
            print("DEBUG: microphone capture error:", repr(mic_e))
            # attempt a fallback to try device-by-index capture (short attempt)
            try:
                names = sr.Microphone.list_microphone_names()
                for idx in range(len(names)):
                    try:
                        print(f"DEBUG: trying device index {idx} -> {names[idx]}")
                        with sr.Microphone(device_index=idx) as source:
                            r = sr.Recognizer()
                            r.adjust_for_ambient_noise(source, duration=0.7)
                            audio = r.record(source, duration=2)
                            if len(audio.get_wav_data()) > 0:
                                tmp_file = f"temp_mic_audio_dev{idx}.wav"
                                with open(tmp_file, "wb") as f:
                                    f.write(audio.get_wav_data())
                                print("DEBUG: fallback captured audio to", tmp_file)
                                break
                    except Exception as inner:
                        print(f"DEBUG: device {idx} failed: {inner}")
            except Exception as final_dev_e:
                print("DEBUG: fallback device enumeration failed:", repr(final_dev_e))
                return f"❌ Error: Could not capture microphone audio. {mic_e}"

        # Now transcribe using whisper_model
        try:
            print("DEBUG: calling whisper_model.transcribe on", tmp_file)
            res = whisper_model.transcribe(tmp_file, fp16=False)
            print("DEBUG: whisper returned (repr):", repr(res))
        except Exception as w_e:
            print("DEBUG: whisper.transcribe raised an exception:")
            traceback.print_exc()
            # cleanup
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass
            return f"❌ Error during Whisper transcription: {w_e}"

        # cleanup temp file
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass

        # Validate result shape
        if res is None:
            return "❌ Error: whisper returned None (no result)."
        if isinstance(res, dict) and "text" in res:
            txt = (res.get("text") or "").strip()
            if txt:
                return txt
            else:
                return "❌ Could not transcribe (no speech detected)."
        # Some whisper wrappers return an object with .text — handle that defensively
        if hasattr(res, "text"):
            try:
                txt = (res.text or "").strip()
                if txt:
                    return txt
                else:
                    return "❌ Could not transcribe (no speech detected)."
            except Exception as e_attr:
                print("DEBUG: accessing res.text raised:", e_attr)
                traceback.print_exc()
                return f"❌ Error: failed to access transcription text attribute: {e_attr}"

        # unknown shape
        return f"❌ Transcription failed (unexpected whisper result shape): {type(res).__name__}: {repr(res)}"

    except Exception as e:
        print("DEBUG: recognize_speech top-level exception:")
        traceback.print_exc()
        return f"❌ Error: {e}"


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
    """Transcribe audio using Whisper with defensive checks and debug info."""
    try:
        # Debug: check file exists and basic properties
        if not os.path.exists(audio_path):
            return f"❌ Input audio missing: {audio_path}"
        try:
            size_bytes = os.path.getsize(audio_path)
        except Exception:
            size_bytes = None
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_path)
        except Exception:
            duration = None

        print(f"DEBUG: transcribe_audio -> path={audio_path}, size={size_bytes}, duration={duration}")

        if size_bytes is None or size_bytes < 1000:
            return "❌ Audio file too small — possible download/conversion error."

        if duration is not None and duration < 0.3:
            return "❌ Audio duration too short for reliable transcription."

        spinner_text = "Transcribing with Whisper... (This may take a moment)"
        # Use a Streamlit spinner context so it displays to the user
        with st.spinner(spinner_text):
            # Force language to English to avoid wrong-language detection on short clips
            # Increase temperature or task if you want alternatives
            result = whisper_model.transcribe(audio_path, fp16=False, language="en", task="transcribe")

        print("DEBUG: whisper transcribe result keys:", result.keys() if isinstance(result, dict) else type(result))
        if result and isinstance(result, dict) and "text" in result:
            transcript = result["text"].strip()
            if not transcript:
                return "❌ Could not transcribe (no speech detected)."
            return transcript
        else:
            # some wrappers return other shapes; handle defensively:
            if hasattr(result, "text"):
                txt = getattr(result, "text", "").strip()
                if txt:
                    return txt
            return f"❌ Transcription failed (unexpected result): {type(result).__name__}"
    except Exception as e:
        print("DEBUG: transcribe_audio exception:", e)
        traceback.print_exc()
        return f"❌ Error during transcription: {e}"

def scrape_webpage_text(url: str) -> str:
    """Scrape text from a webpage URL."""
    try:
        if not url.startswith('http'):
            url = 'https' + url

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


# --- NEW FUNCTION FOR SPEECH EMOTION RECOGNITION ---
def analyze_speech_emotion(audio_path: str, ser_pipeline):
    """Analyzes the emotion from the audio file itself."""
    try:
        speech, sample_rate = librosa.load(audio_path, sr=16000)
        results = ser_pipeline(speech, top_k=None)
        speech_emotions = {e['label'].title(): e['score'] * 100 for e in results}
        return speech_emotions
    except Exception as e:
        print(f"Speech Emotion Recognition error: {e}")
        return None

# ----------------------------
# YouTube / Cache helper functions
# ----------------------------
import glob
import shutil
import hashlib
import time

def get_youtube_video_id(url: str) -> str:
    """Attempt to extract a YouTube video id from common URL formats."""
    try:
        if not url:
            return None
        # typical watch?v=...
        m = re.search(r"(?:v=|\/videos\/|embed\/|youtu\.be\/)([A-Za-z0-9_\-]{6,})", url)
        if m:
            return m.group(1)
        # fallback: last path segment
        parts = url.rstrip('/').split('/')
        if parts:
            last = parts[-1]
            if len(last) >= 6 and re.match(r'^[A-Za-z0-9_\-]+$', last):
                return last
        return None
    except Exception:
        return None


def fetch_youtube_metadata(url: str) -> dict:
    """
    Fetch metadata (title, uploader, duration, thumbnail) for a YouTube URL without downloading.
    Returns dict or None on failure.
    """
    try:
        if not url:
            return None
        ydl_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None
            return {
                'title': info.get('title'),
                'uploader': info.get('uploader'),
                'duration': info.get('duration'),
                'thumbnail': info.get('thumbnail'),
                'id': info.get('id')
            }
    except Exception as e:
        print(f"fetch_youtube_metadata error: {e}")
        return None


def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def download_youtube_audio(url: str, prefer_ext: str = "m4a") -> str:
    """
    Download YouTube audio WITHOUT caching.
    Always downloads a fresh file into a temporary folder 'yt_temp'.
    Returns the downloaded file path, or an error string starting with '❌'.
    """
    try:
        if not url:
            return "❌ Invalid URL."

        temp_dir = "yt_temp"
        os.makedirs(temp_dir, exist_ok=True)

        # output: yt_temp/audio.<ext>
        output_template = os.path.join(temp_dir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": prefer_ext,
            }],
            "prefer_ffmpeg": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_path = os.path.join(temp_dir, f"audio.{prefer_ext}")

        if os.path.exists(final_path):
            return final_path

        return "❌ Download finished but file not found."

    except Exception as e:
        return f"❌ Download failed: {e}"



def clear_youtube_cache(keep_last_n: int = 3) -> bool:
    """
    Clears old cached YouTube audio files while keeping the newest `keep_last_n`.
    Returns True if operation ran (doesn't guarantee deletions).
    """
    try:
        cache_dir = os.path.join("yt_cache")
        if not os.path.exists(cache_dir):
            return True
        files = glob.glob(os.path.join(cache_dir, "*"))
        if not files:
            return True
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        to_remove = files_sorted[keep_last_n:]
        for f in to_remove:
            try:
                os.remove(f)
            except Exception:
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"clear_youtube_cache error: {e}")
        return False


def convert_to_wav_16k_mono(input_path: str, out_path: str) -> str:
    """
    Convert input audio file to 16kHz mono WAV suitable for Whisper.
    Returns out_path on success, or an error string starting with "❌".
    """
    try:
        if not os.path.exists(input_path):
            return f"❌ Input file does not exist: {input_path}"
        # pydub will use ffmpeg/avlib under the hood
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")
        if os.path.exists(out_path):
            return out_path
        return f"❌ Conversion finished but output missing: {out_path}"
    except Exception as e:
        return f"❌ Conversion failed: {e}"


def get_youtube_video_id(url: str) -> str:
    """
    Extract a YouTube video id from common URL formats, including:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/embed/VIDEOID
      - https://youtube.com/shorts/VIDEOID?si=...
    Returns the id string (e.g. 'r9v6D0rtTZ0') or None if it can't parse.
    """
    try:
        if not url or not isinstance(url, str):
            return None

        url = url.strip()

        # Remove trailing params for easier matching (keep original for fallback)
        url_no_qs = url.split('?', 1)[0].split('#', 1)[0]

        # 1) watch?v=VIDEOID (may have other query params)
        m = re.search(r'[?&]v=([A-Za-z0-9_\-]{6,})', url)
        if m:
            return m.group(1)

        # 2) youtu.be/VIDEOID or youtu.be/VIDEOID?t=...
        m = re.search(r'youtu\.be\/([A-Za-z0-9_\-]{6,})', url_no_qs)
        if m:
            return m.group(1)

        # 3) embed/VIDEOID
        m = re.search(r'embed\/([A-Za-z0-9_\-]{6,})', url_no_qs)
        if m:
            return m.group(1)

        # 4) shorts/VIDEOID (common for short URLs)
        m = re.search(r'shorts\/([A-Za-z0-9_\-]{6,})', url_no_qs)
        if m:
            return m.group(1)

        # 5) last path segment fallback (strip trailing slash)
        parts = url_no_qs.rstrip('/').split('/')
        if parts:
            last = parts[-1]
            # sometimes last segment contains extra stuff; ensure it's id-like
            if re.fullmatch(r'[A-Za-z0-9_\-]{6,}', last):
                return last

        # Nothing matched
        return None
    except Exception:
        return None

