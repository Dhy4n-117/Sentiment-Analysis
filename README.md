# 🤖 Advanced Sentiment Analysis Platform ✨

An intelligent, multi-modal sentiment analysis tool built with Streamlit, Transformers, and Whisper. This platform doesn't just read text; it understands emotion, sarcasm, and even the tone of voice from video and audio files.

---

## 🚀 Key Features

This platform provides a deep, comprehensive understanding of sentiment by combining multiple AI models.

* ✨ **Multi-Modal Analysis:** Analyze text, audio (`.mp3`, `.wav`), and video (`.mp4`) files.
* 📈 **Hybrid Scoring:** Combines the power of **RoBERTa** (a Transformer model) with the speed of **VADER** (a rule-based model) for a more accurate and nuanced sentiment score.
* 🎯 **Aspect-Based Sentiment:** Goes beyond a simple overall score. It finds *what* you're talking about (e.g., "camera," "battery life") and analyzes the sentiment for each specific aspect.
* 🎬 **File & YouTube Analysis:** Upload your own files or just paste a **YouTube URL** to download, transcribe, and analyze the content.
* 🗣️ **Speech Emotion Recognition:** It's not just *what* you say, but *how* you say it. The tool analyzes the audio's tone to detect emotions like joy, sadness, or anger.
* 🧐 **Sarcasm Detection:** A dedicated model looks for irony and sarcasm, warning you if the positive words might not mean what you think.
* 🤖 **Sentiment-Aware Chatbot:** Have a conversation with an AI that understands the emotion behind your messages.

---

## 🛠️ How It Works (The Tech)

This app's power comes from a "committee" of AI models that work together.

1.  **Input:** You provide text, a file, or a URL.
2.  **Transcription:** If it's audio or video, **OpenAI's Whisper** model creates a highly accurate text transcript.
3.  **Sentiment Analysis:**
    * **RoBERTa** and **VADER** analyze the text for a hybrid sentiment score.
    * **BERT** (a 43-label emotion model) detects emotions like *joy*, *anger*, or *admiration*.
    * A **Sarcasm-detection** model flags potential irony.
4.  **Speech Analysis:** A **Wav2Vec2** model analyzes the raw audio (from the file or video) to find the emotion in the *tone of voice*.
5.  **NLP:** **SpaCy** is used to break down sentences and find the "aspects" (noun chunks) for analysis.
6.  **Output:** All this information is presented in a clean, interactive dashboard with charts and detailed breakdowns.

---

## 🏛️ Project Structure

This project has been refactored for clarity, maintainability, and scalability. The logic is now separated from the user interface.

* `app.py`: The **frontend** of the application. This file contains all the Streamlit code responsible for the user interface (UI), such as pages, buttons, charts, and layout.
* `analysis_logic.py`: The **backend** "brain" of the application. This file contains all the core data processing and AI functions (model loading, text analysis, web scraping, video transcription, etc.).
* `requirements.txt`: A list of all required Python packages.

---

## 🛠️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Dhy4n-117/Sentiment-Analysis.git]
    cd Sentiment-Analysis
    ```
2.  **Install Prerequisites:**
    * This project requires **FFmpeg** (for video/audio) and **PortAudio** (for voice recording).

    * **On Windows:**
        * Download and install FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (get the `ffmpeg-release-full.7z` file).
        * Extract it and add the `bin` folder to your Windows PATH environment variable.
        * `pyaudio` (for voice) is installed via `pip`, but it may require you to install "Microsoft C++ Build Tools" if it fails.

    * **On Linux (Debian/Ubuntu):**
        * Run the following command to install both prerequisites from the terminal:
            ```bash
            sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev
            ```
    * **On macOS:**
        * Use [Homebrew](https://brew.sh/) to install:
            ```bash
            brew install ffmpeg portaudio
            ```
3.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
6.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---
