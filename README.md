-----

# 🧠 Advanced Sentiment Analysis

[](https://streamlit.io/)
[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

**Advanced Sentiment Analysis** is a comprehensive, multi-modal sentiment analysis platform built with Streamlit. It goes beyond simple positive/negative ratings to provide nuanced insights from text, voice, and video content.

The application leverages a powerful hybrid model combining **BERT** for deep contextual understanding and **VADER** for lexicon-based emotional intensity, along with **SpaCy** for aspect-based analysis.

-----

## ✨ Features

  * **Hybrid Sentiment Model:** Combines `nlptown/bert-base-multilingual-uncased-sentiment` and `vaderSentiment` for a more accurate and robust final score (Positive, Negative, Neutral).
  * **Emotion Analysis:** Detects a range of emotions from the text, including Joy, Sadness, Anger, Fear, Surprise, and Trust.
  * **Aspect-Based Sentiment (ABSA):** Automatically extracts key topics (aspects) from text using `SpaCy` and determines the sentiment for each one (e.g., "The *camera* is **great**, but the *battery life* is **disappointing**.").
  * **Multi-Modal Input:**
      * **📝 Text Analysis:** Analyze any pasted text.
      * **🎤 Voice Analysis:** Record audio directly from your microphone for transcription and sentiment analysis (requires `PyAudio`).
      * **🎬 Video File Analysis:** Upload `mp4`, `mov`, `avi`, or `webm` files. The app extracts the audio, transcribes it, and provides a full sentiment breakdown.
      * **🔗 YouTube Video Analysis:** Paste a YouTube URL. The app attempts to download the audio using `yt-dlp`, transcribes it, and analyzes the content.
  * **🤖 Sentiment-Aware Chatbot:** An interactive chatbot that recognizes the sentiment of your messages and responds accordingly.
  * **📊 Analysis History:** All results are saved in your session. View past analyses and download your entire history as a CSV file.
  * **🎨 Modern UI:** A clean, responsive, and aesthetically pleasing interface built with `streamlit`.

-----

## 🛠️ How It Works

The core analysis engine combines two powerful models for a weighted average sentiment score:

1.  **BERT (60% weight):** The `nlptown/bert-base-multilingual-uncased-sentiment` model provides deep, contextual understanding of the text. Its 1-5 star rating is normalized into a sentiment score.
2.  **VADER (40% weight):** The `vaderSentiment` library provides a lexicon- and rule-based sentiment score (a "compound" score) that is particularly effective at capturing intensity, capitalization, and emojis.
3.  **SpaCy (for Aspects):** The `en_core_web_sm` model from SpaCy is used to perform Natural Language Processing (NLP) tasks, specifically identifying noun chunks. These chunks are treated as "aspects," and their surrounding context is analyzed to determine their specific sentiment.

For audio/video, the media is first processed with `pydub` (to extract audio) and then transcribed to text using `SpeechRecognition`. This transcript is then fed into the analysis engine.

-----

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### 1\. Prerequisites

  * Python 3.9+
  * `ffmpeg`: Required by `pydub` for audio processing.
      * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system's PATH.
      * **macOS (via Homebrew):** `brew install ffmpeg`
      * **Linux (via apt):** `sudo apt update && sudo apt install ffmpeg`
  * `portaudio`: Required by `PyAudio` for microphone access.
      * **macOS (via Homebrew):** `brew install portaudio`
      * **Linux (via apt):** `sudo apt-get install portaudio19-dev python3-pyaudio`

### 2\. Installation

1.  **Clone the repository:** (Replace `your-repo-name` with your actual repository name)

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    You can install the packages directly or create a `requirements.txt` file.

    **Option A: Create `requirements.txt`**
    Create a file named `requirements.txt` and paste the following content:

    ```txt
    streamlit
    numpy<2.0
    protobuf==3.20.3
    torch
    transformers
    vaderSentiment
    SpeechRecognition
    pydub
    opencv-python
    yt-dlp
    spacy
    plotly
    pandas
    pyaudio
    ```

    Then, install from the file:

    ```bash
    pip install -r requirements.txt
    ```

    **Option B: Install directly (Copy-paste to terminal)**

    ```bash
    pip install "numpy<2.0" protobuf==3.20.3
    pip install streamlit torch transformers vaderSentiment
    pip install SpeechRecognition pydub opencv-python
    pip install yt-dlp spacy plotly pandas pyaudio
    ```

4.  **Download the SpaCy model:**
    After installation, you must download the English language model for SpaCy.

    ```bash
    python -m spacy download en_core_web_sm
    ```

### 3\. Running the Application

Once all dependencies are installed, run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

This will automatically open the application in your default web browser.

-----

## 📂 Project Structure

(Assuming your repo folder is named `advanced-sentiment-analysis`)

```
advanced-sentiment-analysis/
│
├── app.py           # The main Streamlit application script
├── requirements.txt # List of Python dependencies
├── temp_files/      # (Auto-generated) Temporary storage for video uploads
└── README.md        # This file
```

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
