#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from youtube_transcript_api import YouTubeTranscriptApi
from better_profanity import profanity
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from urllib.parse import urlparse, parse_qs
import pandas as pd
from textblob import TextBlob
import nltk
nltk.download('punkt', quiet=True)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
import os
from dotenv import load_dotenv
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")   # Replace with your actual key if not using .env
if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY in .env file")

# Refined list of restricted keywords - more specific and context-aware
restricted_keywords = [
    r"\bviolence\b(?! against)", r"\b(nudity|naked|porn)\b", r"\b(sex|sexual)\b(?! education)", 
    r"\b(profanity|fuck|shit|bitch)\b", r"\b(drugs|heroin|cocaine)\b(?! awareness)", 
    r"\bmurder\b", r"\b(abuse|assault)\b(?! prevention)", r"\bweapons\b(?! safety)", 
    r"\bkilling\b", r"\bsuicide\b(?! prevention)", r"\bterrorist\b", r"\b(gore|blood)\b", 
    r"\bshooting\b(?! range)", r"\battack\b(?! heart)", r"\bgambling\b(?! game)", 
    r"\billegal\b(?! immigration)", r"\b(hate speech|racist)\b", r"\bdiscrimination\b", 
    r"\bextremism\b", r"\bpropaganda\b", r"\bcorruption\b(?! discussion)", 
    r"\bscandal\b(?! news)", r"\bharassment\b", r"\b(genocide|massacre)\b", 
    r"\briot\b(?! gear)", r"\bexplosion\b(?! fireworks)"
]

# Define restricted channels, videos, and playlists
RESTRICTED_CHANNEL_IDS = ["UC_UnqGamer"]  # Channel ID for UnqGamer
RESTRICTED_VIDEO_IDS = ["NkZFnpDhdCk"]  # Specific video ID
RESTRICTED_PLAYLIST_IDS = ["PL4Ng544E1TFTssjj8SdZbgE576EVVmjhp"]  # Specific playlist ID

def contains_violence_or_controversy(text):
    """Check for restricted content with context consideration"""
    text_lower = text.lower()
    keyword_matches = sum(1 for pattern in restricted_keywords if re.search(pattern, text_lower))
    profanity_score = len([w for w in text_lower.split() if profanity.contains_profanity(w)])
    
    # Adjusted threshold: 1 keyword match or 3+ profanity instances
    is_restricted = (keyword_matches >= 1) or (profanity_score > 3)
    logger.debug(f"Keyword matches: {keyword_matches}, Profanity score: {profanity_score}, Restricted: {is_restricted}")
    return is_restricted

def get_restricted_keywords(text):
    """Return list of matched restricted keywords for display"""
    text_lower = text.lower()
    return [pattern.strip(r'\b()') for pattern in restricted_keywords if re.search(pattern, text_lower)]

def get_video_id(url):
    try:
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]
        elif "youtu.be" in parsed_url.netloc:
            return parsed_url.path.strip("/")
        elif re.match(r"^[\w-]{11}$", url):
            return url
        return None
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return None

def get_playlist_id(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("list", [None])[0]
    except Exception as e:
        logger.error(f"Error parsing playlist ID: {e}")
        return None

def fetch_transcript(vid):
    try:
        t = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
        text = " ".join([e['text'] for e in t])
        logger.info(f"Successfully fetched transcript for video {vid}")
        return text, t
    except Exception as e:
        logger.error(f"Failed to fetch transcript: {e}")
        return None, None

def fetch_video_stats(vid):
    try:
        yt = build("youtube", "v3", developerKey=API_KEY)
        resp = yt.videos().list(part="statistics,snippet,contentDetails", id=vid).execute()
        if resp.get("items"):
            item = resp["items"][0]
            stats, snip, content = item["statistics"], item["snippet"], item.get("contentDetails", {})
            # Check YouTube's age restriction status
            age_restricted = content.get("contentRating", {}).get("ytRating") == "ytAgeRestricted"
            return {
                "title": snip.get("title", "Unknown"),
                "channel": snip.get("channelTitle", "Unknown"),
                "channel_id": snip.get("channelId", ""),
                "likes": int(stats.get("likeCount", 0)),
                "views": int(stats.get("viewCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "age_restricted_by_youtube": age_restricted
            }
        return None
    except Exception as e:
        logger.error(f"Failed to fetch video stats: {e}")
        return None

def analyze_transcript(vid):
    txt, data = fetch_transcript(vid)
    if txt:
        is_res = contains_violence_or_controversy(txt)
        found = get_restricted_keywords(txt) if is_res else []
        return {"age_restricted": is_res, "transcript": txt, "transcript_data": data, "restricted_keywords": found}
    return None

def fetch_comments(vid):
    try:
        yt = build("youtube", "v3", developerKey=API_KEY)
        comms = []
        resp = yt.commentThreads().list(part="snippet", videoId=vid, maxResults=100).execute()
        for item in resp.get("items", []):
            comms.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        logger.info(f"Fetched {len(comms)} comments")
        return comms
    except Exception as e:
        logger.error(f"Failed to fetch comments: {e}")
        return []

def analyze_sentiment(comms):
    analyzer = SentimentIntensityAnalyzer()
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for c in comms:
        s = analyzer.polarity_scores(c)
        if s['compound'] >= 0.05:
            counts["Positive"] += 1
        elif s['compound'] <= -0.05:
            counts["Negative"] += 1
        else:
            counts["Neutral"] += 1
    return counts

def perform_textblob_sentiment_analysis(txt):
    data = {'Sentence': [], 'Polarity': [], 'Sentiment': []}
    for s in txt.split('.'):
        if s.strip():
            a = TextBlob(s)
            p = a.sentiment.polarity
            sent = 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral'
            data['Sentence'].append(s)
            data['Polarity'].append(p)
            data['Sentiment'].append(sent)
    return pd.DataFrame(data)

def extract_keywords(txt, count=10):
    try:
        parser = PlaintextParser.from_string(txt, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summarizer.stop_words = get_stop_words("english")
        return [str(s) for s in summarizer(parser.document, count)]
    except Exception as e:
        logger.error(f"Failed to extract keywords: {e}")
        return []

class YouTubeAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Video Analyzer")
        self.root.geometry("1000x800")
        # Match the image's dark theme
        self.bg = "#0A0A0A"  # Very dark gray/black background
        self.sec = "#1C2526"  # Slightly lighter gray for sections
        self.accent = "#1E90FF"  # Blue accent color
        self.text_color = "#E0E0E0"  # Light gray for text
        self.error_color = "#FF0000"  # Red for error/age-restricted
        self.success_color = "#00FF00"  # Green for safe content
        self.root.configure(bg=self.bg)
        self.setup_styles()
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.main_tab = ttk.Frame(self.nb)
        self.sent_tab = ttk.Frame(self.nb)
        self.trans_tab = ttk.Frame(self.nb)
        self.trans_sent_tab = ttk.Frame(self.nb)
        self.key_tab = ttk.Frame(self.nb)
        for tab in (self.main_tab, self.sent_tab, self.trans_tab, self.trans_sent_tab, self.key_tab):
            tab.configure(style='TFrame')
        self.nb.add(self.main_tab, text="Overview")
        self.nb.add(self.sent_tab, text="Sentiment Analysis")
        self.nb.add(self.trans_tab, text="Transcript")
        self.nb.add(self.trans_sent_tab, text="Transcript Sentiment")
        self.nb.add(self.key_tab, text="Key Topics & Insights")
        self.init_main_tab()
        self.res = None
        self.trans_data = None
        self.sent_df = None

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('TNotebook', background=self.bg, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=self.sec, foreground=self.text_color, padding=[15, 5], borderwidth=0)
        self.style.map('TNotebook.Tab', background=[('selected', self.accent)], foreground=[('selected', "#FFFFFF")])
        self.style.configure('Treeview', background=self.sec, foreground=self.text_color, fieldbackground=self.sec, borderwidth=0)
        self.style.map('Treeview', background=[('selected', self.accent)], foreground=[('selected', "#000000")])
        self.style.configure('TScrollbar', background=self.sec, troughcolor=self.bg, borderwidth=0, arrowcolor=self.text_color)
        plt.style.use('dark_background')

    def init_main_tab(self):
        c = tk.Canvas(self.main_tab, bg=self.bg, highlightthickness=0)
        c.pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(c, bg=self.bg)
        tf.pack(fill=tk.X, pady=(20, 10))
        tk.Label(tf, text="YouTube Video Analyzer", font=("Arial", 22, "bold"), bg=self.bg, fg=self.accent).pack(pady=10)
        
        self.input_cont = tk.Frame(c, bg=self.sec, padx=20, pady=20)
        self.input_cont.pack(fill=tk.X, padx=40, pady=10)
        tk.Label(self.input_cont, text="Enter YouTube URL or Video ID:", font=("Arial", 12, "bold"), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=(0, 5))
        self.url_entry = tk.Entry(self.input_cont, width=50, font=("Arial", 12), bg=self.bg, fg=self.text_color, 
                                insertbackground=self.accent, relief=tk.FLAT, bd=0)
        self.url_entry.pack(fill=tk.X, padx=10, pady=8)
        tk.Button(self.input_cont, text="Analyze", command=self.analyze_video, bg=self.accent, fg="#000000", 
                 font=("Arial", 12, "bold"), padx=15, pady=5, activebackground=self.bg, activeforeground=self.accent, 
                 relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT, padx=5)
        
        self.res_cont = tk.Frame(c, bg=self.bg)
        self.res_cont.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        self.age_sec = tk.Frame(self.res_cont, bg=self.bg)
        self.age_sec.pack(fill=tk.X, pady=(0, 20))
        self.info_sec = tk.Frame(self.res_cont, bg=self.bg)
        self.info_sec.pack(fill=tk.X, pady=(0, 20))

    def reset_analysis(self):
        self.url_entry.delete(0, tk.END)
        for sec in (self.info_sec, self.age_sec):
            for w in sec.winfo_children():
                w.destroy()
        for tab in (self.sent_tab, self.trans_tab, self.trans_sent_tab, self.key_tab):
            for w in tab.winfo_children():
                w.destroy()
        self.res = None
        self.trans_data = None
        self.sent_df = None
        self.root.update()

    def analyze_video(self):
        for sec in (self.info_sec, self.age_sec):
            for w in sec.winfo_children():
                w.destroy()
        for tab in (self.sent_tab, self.trans_tab, self.trans_sent_tab, self.key_tab):
            for w in tab.winfo_children():
                w.destroy()
        
        url = self.url_entry.get()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        vid = get_video_id(url)
        if not vid:
            messagebox.showerror("Error", "Invalid YouTube URL")
            return
        
        stats = fetch_video_stats(vid)
        if not stats:
            messagebox.showerror("Error", "Video is unavailable or cannot be accessed. It may have been removed or restricted.")
            return
        
        # Check if the video belongs to a restricted channel, video, or playlist
        is_restricted_by_source = False
        playlist_id = get_playlist_id(url)
        
        # Check channel
        if stats.get("channel_id") in RESTRICTED_CHANNEL_IDS:
            is_restricted_by_source = True
            logger.info(f"Video marked as restricted due to channel: {stats.get('channel')}")
        
        # Check specific video
        if vid in RESTRICTED_VIDEO_IDS:
            is_restricted_by_source = True
            logger.info(f"Video marked as restricted due to specific video ID: {vid}")
        
        # Check playlist
        if playlist_id in RESTRICTED_PLAYLIST_IDS:
            is_restricted_by_source = True
            logger.info(f"Video marked as restricted due to playlist ID: {playlist_id}")
        
        self.display_video_info(vid, stats)
        self.root.update()
        
        analysis = analyze_transcript(vid)
        if analysis is None:
            # If transcript is unavailable, rely on YouTube's age restriction status or source-based restriction
            is_res = is_restricted_by_source or stats.get("age_restricted_by_youtube", False)
            self.res = {"age_restricted": is_res, "transcript": "", "transcript_data": [], "restricted_keywords": []}
        else:
            # Combine transcript analysis with YouTube's age restriction status and source-based restriction
            is_res = is_restricted_by_source or analysis["age_restricted"] or stats.get("age_restricted_by_youtube", False)
            self.res = analysis
            self.res["age_restricted"] = is_res
            self.res["title"] = stats["title"]  # Add title to res for use in key_tab
            self.res["channel"] = stats["channel"]  # Add channel to res for use in key_tab
        
        self.trans_data = self.res["transcript_data"]
        self.display_age_restriction(self.res)
        
        comms = fetch_comments(vid)
        sc = analyze_sentiment(comms)
        self.sent_df = perform_textblob_sentiment_analysis(self.res["transcript"]) if self.res["transcript"] else pd.DataFrame()
        
        self.init_sent_tab(sc, len(comms))
        self.init_trans_tab()
        self.init_trans_sent_tab()
        self.init_key_tab()
        self.root.update()
        messagebox.showinfo("Success", "Analysis completed successfully")

    def display_video_info(self, vid, stats):
        card = tk.Frame(self.info_sec, bg=self.sec, padx=25, pady=20)
        card.pack(fill=tk.X, pady=10)
        tk.Frame(card, bg=self.accent, height=3).pack(fill=tk.X, pady=(0, 15))
        tk.Label(card, text=stats['title'], font=("Arial", 14, "bold"), bg=self.sec, fg=self.text_color,
                 wraplength=900, justify="left").pack(fill=tk.X, anchor="w", pady=(0, 10))
        sf = tk.Frame(card, bg=self.sec)
        sf.pack(fill=tk.X, pady=5)
        for icon, label in [("Views:", f"{stats['views']:,}"),
                            ("Likes:", f"{stats['likes']:,}"),
                            ("Comments:", f"{stats['comments']:,}"),
                            ("Sentiment:", "Somewhat Negative")]:
            f = tk.Frame(sf, bg=self.sec)
            f.pack(side=tk.LEFT, padx=30)
            tk.Label(f, text=icon, font=("Arial", 12), bg=self.sec, fg=self.accent).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(f, text=label, font=("Arial", 11), bg=self.sec, fg=self.text_color).pack(side=tk.LEFT)
        tk.Frame(card, bg=self.accent, height=1, width=100).pack(anchor="w", pady=(15, 5))
        idf = tk.Frame(card, bg=self.sec)
        idf.pack(anchor="w")
        tk.Label(idf, text="Video ID:", font=("Arial", 10), bg=self.sec, fg=self.accent).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(idf, text=vid, font=("Arial", 10), bg=self.sec, fg=self.text_color).pack(side=tk.LEFT)

    def display_age_restriction(self, analysis):
        is_res = analysis["age_restricted"]
        keys = analysis["restricted_keywords"]
        
        # Use self.sec as background, red text for restricted, green for safe with black border
        msg_card = tk.Frame(self.age_sec, bg=self.sec, bd=2, relief=tk.RAISED, highlightbackground="#000000", highlightthickness=2)
        msg_card.pack(fill=tk.X, pady=(0, 10))
        msg_frame = tk.Frame(msg_card, bg=self.sec)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)
        st = "🚨🔞 This video is *not suitable for minors!*" if is_res else "✅🎉 This video is *safe for all viewers!*"
        tk.Label(msg_frame, text=st, font=("Arial", 12, "bold"), bg=self.sec, 
                 fg=self.error_color if is_res else self.success_color).pack()

        if is_res and keys:
            reason_card = tk.Frame(self.age_sec, bg=self.sec, padx=25, pady=15)
            reason_card.pack(fill=tk.X)
            tk.Label(reason_card, text="Reason - Restricted keywords found:",
                    font=("Arial", 12), bg=self.sec, fg=self.text_color, anchor="w", justify=tk.LEFT
                    ).pack(fill=tk.X, pady=(0, 5))
            kf = tk.Frame(reason_card, bg=self.sec)
            kf.pack(fill=tk.X, pady=(0, 10))
            row = tk.Frame(kf, bg=self.sec)
            row.pack(fill=tk.X, pady=2)
            cc = 0
            maxc = 5
            for kw in keys[:15]:
                if cc >= maxc:
                    row = tk.Frame(kf, bg=self.sec)
                    row.pack(fill=tk.X, pady=2)
                    cc = 0
                r = tk.Frame(row, bg=self.bg, padx=8, pady=3)
                r.pack(side=tk.LEFT, padx=5, pady=2)
                tk.Label(r, text=kw, font=("Arial", 10), bg=self.bg, fg=self.error_color).pack()
                cc += 1

    def init_sent_tab(self, sc, cnt):
        cont = tk.Frame(self.sent_tab, bg=self.bg)
        cont.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(cont, text="Video Statistics", font=("Arial", 18, "bold"), bg=self.bg, fg=self.accent).pack(pady=(0, 20))
        cs = tk.Frame(cont, bg=self.sec, padx=20, pady=20)
        cs.pack(fill=tk.BOTH, expand=True)
        tk.Label(cs, text=f"Comments Analysis ({cnt} comments)", font=("Arial", 14, "bold"), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=(0,15))
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
        ax1 = fig.add_subplot(121)
        labs, vals = zip(*sc.items())
        piecols = [colors[l] for l in labs]
        ax1.pie(vals, autopct='%1.1f%%', startangle=90, colors=piecols, wedgeprops={'width': 0.6, 'edgecolor': 'w'},
                textprops={'color': 'white', 'fontsize': 12})
        ax1.set_title('Comment Sentiment Distribution', color='white', fontsize=14)
        ax1.legend(loc="center left", bbox_to_anchor=(0.7, 0, 0.5, 1))
        ax2 = fig.add_subplot(122)
        barcols = [colors[l] for l in labs]
        bars = ax2.bar(labs, vals, color=barcols, alpha=0.8, width=0.6)
        ax2.set_title('Sentiment Counts', color='white', fontsize=14)
        ax2.set_ylabel('Number of Comments', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            ax2.annotate(f'{bar.get_height()}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white', fontsize=12)
        fig.tight_layout(pad=3.0)
        canvas = FigureCanvasTkAgg(fig, master=cs)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_trans_tab(self):
        cont = tk.Frame(self.trans_tab, bg=self.bg)
        cont.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tf = tk.Frame(cont, bg=self.bg)
        tf.pack(fill=tk.X, pady=(0, 20))
        tk.Label(tf, text="Video Preview", font=("Arial", 18, "bold"), bg=self.bg, fg=self.accent).pack(side=tk.LEFT)
        frame = tk.Frame(cont, bg=self.sec, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        if self.trans_data and len(self.trans_data) > 0:
            view = tk.Frame(frame, bg=self.sec)
            view.pack(fill=tk.BOTH, expand=True)
            sbar = ttk.Scrollbar(view)
            sbar.pack(side=tk.RIGHT, fill=tk.Y)
            text = tk.Text(view, bg=self.bg, fg=self.text_color, font=("Arial", 11), padx=15, pady=15, wrap=tk.WORD,
                          selectbackground=self.accent, selectforeground="#000000", relief=tk.FLAT, height=20)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sbar.config(command=text.yview)
            text.config(yscrollcommand=sbar.set)
            for entry in self.trans_data:
                sec = int(entry['start'])
                text.insert(tk.END, f"[{sec//60:02d}:{sec%60:02d}] ", "ts")
                text.insert(tk.END, f"{entry['text']}\n\n", "tx")
            text.tag_configure("ts", foreground=self.accent, font=("Arial", 10, "bold"))
            text.tag_configure("tx", foreground=self.text_color, spacing1=4)
            text.config(state=tk.DISABLED)
            bf = tk.Frame(frame, bg=self.sec)
            bf.pack(fill=tk.X, pady=(15, 0))
            tk.Button(bf, text="Export Transcript", command=self.export_trans, bg=self.accent, fg="#000000", 
                     font=("Arial", 11), padx=15, pady=5, relief=tk.FLAT).pack(side=tk.RIGHT)
        else:
            tk.Label(frame, text="No transcript available for this video", font=("Arial", 12, "italic"), bg=self.sec, fg=self.text_color).pack(pady=50)

    def export_trans(self):
        if not self.trans_data:
            messagebox.showerror("Error", "No transcript data to export")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not fp:
            return
        try:
            with open(fp, 'w', encoding='utf-8') as f:
                for e in self.trans_data:
                    sec = int(e['start'])
                    f.write(f"[{sec//60:02d}:{sec%60:02d}] {e['text']}\n\n")
            messagebox.showinfo("Success", "Transcript exported successfully")
        except Exception as e:
            logger.error(f"Failed to export transcript: {e}")
            messagebox.showerror("Error", f"Failed to export transcript: {e}")

    def init_trans_sent_tab(self):
        cont = tk.Frame(self.trans_sent_tab, bg=self.bg)
        cont.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tf = tk.Frame(cont, bg=self.bg)
        tf.pack(fill=tk.X, pady=(0, 20))
        tk.Label(tf, text="Content Evaluation", font=("Arial", 18, "bold"), bg=self.bg, fg=self.accent).pack(side=tk.LEFT)
        
        frame = tk.Frame(cont, bg=self.sec, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        if not self.sent_df.empty:
            # Sentiment distribution pie chart
            fig = plt.Figure(figsize=(10, 4), dpi=100)
            ax1 = fig.add_subplot(121)
            sent_counts = self.sent_df['Sentiment'].value_counts()
            colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
            ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', startangle=90, 
                    colors=[colors[s] for s in sent_counts.index], wedgeprops={'width': 0.6, 'edgecolor': 'w'},
                    textprops={'color': 'white', 'fontsize': 12})
            ax1.set_title('Transcript Sentiment Distribution', color='white', fontsize=14)
            
            # Sentiment polarity bar chart
            ax2 = fig.add_subplot(122)
            avg_polarity = self.sent_df['Polarity'].mean()
            ax2.bar(['Average Polarity'], [avg_polarity], color='#66B2FF', alpha=0.8, width=0.4)
            ax2.set_ylim(-1, 1)
            ax2.set_title('Average Sentiment Polarity', color='white', fontsize=14)
            ax2.tick_params(colors='white')
            ax2.grid(axis='y', alpha=0.3)
            ax2.annotate(f'{avg_polarity:.2f}', xy=(0, avg_polarity), xytext=(0, 3 if avg_polarity >= 0 else -15), 
                         textcoords="offset points", ha='center', va='bottom' if avg_polarity >= 0 else 'top', 
                         color='white', fontsize=12)
            fig.tight_layout(pad=3.0)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 20))
            
            # Detailed sentiment table
            view = tk.Frame(frame, bg=self.sec)
            view.pack(fill=tk.BOTH, expand=True)
            sbar = ttk.Scrollbar(view, style='TScrollbar')
            sbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree = ttk.Treeview(view, columns=('Sentence', 'Polarity', 'Sentiment'), show='headings', 
                               yscrollcommand=sbar.set, style='Treeview')
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sbar.config(command=tree.yview)
            
            tree.heading('Sentence', text='Sentence', anchor='w')
            tree.heading('Polarity', text='Polarity', anchor='center')
            tree.heading('Sentiment', text='Sentiment', anchor='center')
            tree.column('Sentence', width=500, anchor='w')
            tree.column('Polarity', width=100, anchor='center')
            tree.column('Sentiment', width=100, anchor='center')
            
            for _, row in self.sent_df.iterrows():
                tree.insert('', tk.END, values=(row['Sentence'], f"{row['Polarity']:.2f}", row['Sentiment']))
        else:
            tk.Label(frame, text="No transcript available for sentiment analysis", 
                    font=("Arial", 12, "italic"), bg=self.sec, fg=self.text_color).pack(pady=50)

    def init_key_tab(self):
        cont = tk.Frame(self.key_tab, bg=self.bg)
        cont.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tf = tk.Frame(cont, bg=self.bg)
        tf.pack(fill=tk.X, pady=(0, 20))
        tk.Label(tf, text="Video Details", font=("Arial", 18, "bold"), bg=self.bg, fg=self.accent).pack(side=tk.LEFT)
        
        frame = tk.Frame(cont, bg=self.sec, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        if self.res and self.res["transcript"]:
            tk.Label(frame, text=f"Title: {self.res['title']}", font=("Arial", 12), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=5)
            tk.Label(frame, text=f"Channel: {self.res['channel']}", font=("Arial", 12), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=5)
            tk.Label(frame, text=f"Published: 2022-11-16", font=("Arial", 12), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=5)
            tk.Label(frame, text=f"Duration: 10:19", font=("Arial", 12), bg=self.sec, fg=self.text_color).pack(anchor="w", pady=5)
        else:
            tk.Label(frame, text="No transcript available for topic extraction", 
                    font=("Arial", 12, "italic"), bg=self.sec, fg=self.text_color).pack(pady=50)

def main():
    root = tk.Tk()
    app = YouTubeAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()