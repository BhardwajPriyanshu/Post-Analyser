import streamlit as st
# import spacy
# import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from transformers import pipeline
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
import nltk

# Add this to download required data on first run
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Downloads and setup ---
nltk.download("vader_lexicon")
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

# --- Hugging Face pipelines (forced to CPU) ---
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1  # CPU
)

toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True,
    device=-1  # CPU
)

topic_model = pipeline(
    "text-classification",
    model="cardiffnlp/tweet-topic-21-multi",
    return_all_scores=True,
    device=-1  # CPU
)

# --- News API key ---
NEWS_API_KEY = "4f8e88f62861411aa8c0384833bda54a"

# --- Function to fetch news ---
def get_news_articles(keyword, api_key):
    url = f"https://newsapi.org/v2/everything?q={keyword}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])[:3]
        return articles
    except Exception as e:
        return [{"title": "Error fetching news", "description": str(e), "url": "#"}]

# --- UI ---
st.set_page_config(page_title="Post Analyzer", layout="centered")
st.title("🔍 Post Analyzer")
st.subheader("Analyze post sentiment, emotional and toxicity and providing related News")

user_input = st.text_area("Paste a tweet or post:")

if st.button("Analyze Post") and user_input:
    # --- Sentiment ---
    st.markdown("### 💬 Sentiment")
    scores = sid.polarity_scores(user_input)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive 😊"
    elif compound <= -0.05:
        sentiment = "Negative 😠"
    else:
        sentiment = "Neutral 😐"
    st.write(f"**{sentiment}** (Score: `{compound}`)")

    # --- Emotion ---
    st.markdown("### 🧠 Emotion Detection")
    emotions = emotion_model(user_input)[0]
    top_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:2]
    emoji_map = {
        "joy": "😊", "anger": "😡", "sadness": "😢", "fear": "😨",
        "surprise": "😲", "disgust": "🤢", "neutral": "😐"
    }
    for emo in top_emotions:
        label = emo["label"].lower()
        emoji = emoji_map.get(label, "")
        st.write(f"**{label.capitalize()} {emoji}** - {emo['score']*100:.2f}%")

    # --- Toxicity ---
    st.markdown("### ☠️ Toxicity Check")
    toxicity = toxicity_model(user_input)[0]
    toxic_score = [x for x in toxicity if x["label"] == "toxic"][0]["score"]
    if toxic_score > 0.5:
        st.error(f"**Toxic** 🤬 - Score: {toxic_score:.2f}")
    else:
        st.success(f"**Not Toxic** ✅ - Score: {toxic_score:.2f}")

    # --- Topic classification ---
    st.markdown("### 📚 Topic Classification")
    topics = topic_model(user_input)[0]
    top_topic = max(topics, key=lambda x: x["score"])
    topic_labels = {
        "politics": "🏛️", "sports": "🏅", "tech": "💻", "health": "🩺",
        "entertainment": "🎬", "business": "💼", "science": "🔬", "travel": "✈️",
        "education": "📚", "gaming": "🎮", "music": "🎵"
    }
    topic_label = top_topic["label"].replace("LABEL_", "").lower()
    topic_emoji = topic_labels.get(topic_label, "📝")
    st.write(f"**{topic_label.capitalize()} {topic_emoji}** - {top_topic['score']*100:.2f}%")

    # --- Entity Extraction ---
    st.markdown("### 🔎 Extracted Keywords")
    doc = nlp(user_input)
    keywords = list(set([ent.text for ent in doc.ents]))
    if keywords:
        st.success(", ".join(keywords))
    else:
        st.info("No named entities found.")

    # --- Related News ---
    st.markdown("### 📰 Related News")
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        st.warning("⚠️ Please add your NewsAPI key to fetch real articles.")
    else:
        if keywords:
            for kw in keywords[:3]:  # Top 3 entities
                st.markdown(f"**🔑 Keyword:** `{kw}`")
                articles = get_news_articles(kw, NEWS_API_KEY)
                for article in articles:
                    st.markdown(f"**📰 {article['title']}**")
                    st.markdown(f"{article['description'] or '_No summary available_'}")
                    st.markdown(f"[Read more →]({article['url']})")
                    st.markdown("---")
        else:
            st.info("No keywords found to search news.")
