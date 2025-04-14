# social-media-content-analysis-youtube

# 🎥 YouTube Video Analyzer

This is a powerful desktop application built with **Python & Tkinter** that analyzes YouTube videos for **content safety**, **sentiment analysis**, and **transcript insights**.

It detects **age-restricted content**, visualizes **comment sentiment**, and extracts **key topics** from the video transcript — all in a beautiful GUI format.

---

## 🧰 Features

- 🔍 Fetch and display video transcript
- 🛡️ Detect age-restricted and inappropriate content using keywords and profanity detection
- 💬 Sentiment analysis on video comments and transcript
- 🧠 Topic extraction using text summarization techniques
- 📊 Pie and bar charts for sentiment visualization
- 📝 Export transcript as a text file

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nandhan574/social-media-content-analysis-youtube.git
cd social-media-content-analysis-youtube
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup (YouTube Data API v3)

This app requires access to the YouTube Data API v3 to fetch video statistics and comments.

### ▶️ How to get your API key:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use an existing one)
3. Enable **YouTube Data API v3**
4. Go to **Credentials** → Create credentials → **API Key**
5. Copy the API key

### ▶️ Add API Key to Environment File

- Create a `.env` file in the root directory:
```env
YOUTUBE_API_KEY=YOUR_API_KEY_HERE
```

> ✅ Make sure `.env` is **NOT** pushed to GitHub. It's already ignored in `.gitignore`.

---

## ✅ Running the App

Once setup is complete, launch the app using:

```bash
python app.py
```

It opens a desktop window where you can paste any YouTube video URL or ID to start analyzing.

---

## 🛠️ Built With

- Python 3.x
- Tkinter (for GUI)
- `youtube_transcript_api`
- `google-api-python-client`
- `vaderSentiment`
- `better_profanity`
- `textblob`
- `sumy`
- `matplotlib`
- `dotenv`
- `nltk`

---

## 📊 How It Works

1. User inputs a YouTube URL or Video ID
2. App fetches:
   - Transcript (if available)
   - Video stats (views, likes, comments)
   - Top 100 comments
3. It checks for:
   - Keywords related to violence/inappropriate content
   - Profanity count
   - Restricted channel or video IDs
4. It performs:
   - Sentiment analysis on comments & transcript
   - Keyword/topic extraction using LexRank summarizer
5. Results are displayed in different GUI tabs with interactive charts

---

## 📁 Project Structure

```
youtube-video-analyzer/
├── app.py                  # Main Python GUI app
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .gitignore              # Ignored files
└── README.md               # You're reading it!
```

---

## 📸 Screenshots *(Optional)*

> _You can include some images of your UI here using:_
```markdown
![Main UI](screenshots/main_ui.png)
```

---

## 📬 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for more info.
