# 🧠 Advanced Sentiment Analysis

This is an advanced multi-modal sentiment analysis platform built with Streamlit. It analyzes sentiment from text, voice, video files, YouTube URLs, and live webpages, providing a comprehensive breakdown of sentiment, emotion, and key topics.

The core of the project uses a hybrid model, combining **BERT** for deep contextual understanding and **VADER** for emotional intensity, to produce a nuanced and accurate final score.

---

## 🚀 Key Features

* **Multi-Modal Analysis:** Get sentiment from 5 different sources:
    1.  **Text:** Paste in any block of text.
    2.  **Voice:** Record live audio from your microphone for transcription and analysis.
    3.  **Video:** Upload `.mp4`, `.mov`, etc., to automatically transcribe and analyze the content.
    4.  **YouTube:** Paste a YouTube URL to download, transcribe, and analyze the video's audio.
    5.  **Webpage URL:** Paste a URL to scrape and analyze the text content of a news article or blog post.
* **Hybrid AI Model:** Uses a weighted average of **BERT** and **VADER** for a robust, nuanced sentiment score.
* **Rich Visualizations:** Results are displayed with interactive **Plotly charts**, including a sentiment gauge and an emotion bar chart, with all sentiments color-coded.
* **Aspect-Based Sentiment (ABSA):** Automatically extracts key topics and nouns (e.g., "camera," "battery life") from the text and provides a specific sentiment for each one.
* **AI Chatbot:** An emotion-aware chatbot that responds to you differently based on the sentiment of your message.
* **Analysis History:** Automatically saves all analysis results to a session history, which can be downloaded as a CSV file.

---

## 🏛️ Project Structure

This project has been refactored for clarity, maintainability, and scalability. The logic is now separated from the user interface.

* `app.py`: The **frontend** of the application. This file contains all the Streamlit code responsible for the user interface (UI), such as pages, buttons, charts, and layout.
* `analysis_logic.py`: The **backend** "brain" of the application. This file contains all the core data processing and AI functions (model loading, text analysis, web scraping, video transcription, etc.).
* `requirements.txt`: A list of all required Python packages.

---

## 🎯 How It Works

This application follows a clear logic flow depending on the input type:

1.  **Hybrid Sentiment Core:**
    * **BERT** (from `transformers`): Analyzes the text for deep contextual understanding (e.g., "I'm not happy" is negative, even with the word "happy").
    * **VADER**: Analyzes the text for emotional intensity (e.g., "I LOVE this" is more positive than "I like this").
    * **Combined Score:** A weighted average (60% BERT, 40% VADER) is calculated to produce a single, robust score.

2.  **Text, Voice, and Video Analysis:**
    * **Text:** Fed directly into the Hybrid Sentiment Core.
    * **Voice:** Recorded and transcribed using the `SpeechRecognition` library. The resulting text is then fed into the core.
    * **Video/YouTube:** The file is processed with `FFmpeg` and `pydub` to extract the audio. The audio is then transcribed, and the text is fed into the core.

3.  **Webpage Analysis:**
    * The URL is fetched using the `requests` library.
    * `BeautifulSoup4` parses the HTML to scrape and clean all the main article text.
    * The resulting text is fed into the core.

4.  **Aspect-Based Analysis:**
    * After the main analysis, `SpaCy` is used to parse the text and find key nouns and topics.
    * A sentiment analysis is run on the text surrounding each topic to get a specific score for that aspect.

---

## 🛠️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Dhy4n-117/Sentiment-Analysis.git](https://github.com/Dhy4n-117/Sentiment-Analysis.git)
    cd Sentiment-Analysis
    ```
2.  **Install Prerequisites (Windows):**
    * This project requires **FFmpeg** for processing video and audio files.
    * Download and install FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (get the `ffmpeg-release-full.7z` file).
    * Extract it and add the `bin` folder to your Windows PATH environment variable.

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
