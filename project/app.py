import json
import pandas as pd
import psycopg2
import nltk
import re
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
import os 
import traceback
import subprocess
from textblob import TextBlob
import requests
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import string
import torch
from transformers import pipeline

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Database connection function
def connect_db():
    return psycopg2.connect(
        user="postgres",
        password="Mac.phil.007",
        host="127.0.0.1",
        port="5432",
        database="Platform"
    )

def get_market_data():
    """Fetch market data from PostgreSQL"""
    try:
        connection = connect_db()
        cursor = connection.cursor()
        query = "SELECT market_question, market_creation_date, market_chance FROM polymarket_data"
        cursor.execute(query)
        rows = cursor.fetchall()
        return [{"market_question": row[0], "market_creation_date": str(row[1]), "market_chance": row[2]} for row in rows]
    except Exception as e:
        return {"error": str(e)}
    finally:
        if connection:
            cursor.close()
            connection.close()

def get_market_questions():
    """Fetch market questions from the PostgreSQL database."""
    try:
        connection = connect_db()
        cursor = connection.cursor()
        query = "SELECT market_question FROM polymarket_data"
        cursor.execute(query)
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print("Error fetching data:", e)
        return []
    finally:
        if connection:
            cursor.close()
            connection.close()

def clean_text(text):
    """Basic text preprocessing (lowercasing, removing special characters, stopwords)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def classify_topics(num_topics=5, num_words=5):
    """Perform topic modeling and classify each question into a topic."""
    market_questions = get_market_questions()
    
    if not market_questions:
        return {"error": "No data available"}

    cleaned_questions = [clean_text(q) for q in market_questions]
    
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(cleaned_questions)

    # Apply NMF for topic extraction
    nmf = NMF(n_components=num_topics, random_state=42)
    W_nmf = nmf.fit_transform(X)
    H_nmf = nmf.components_

    # Apply LDA for topic extraction
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    
    topics = {}
    for i, topic in enumerate(H_nmf):
        topics[f"Topic {i+1}"] = [feature_names[j] for j in topic.argsort()[:-num_words-1:-1]]

    # Assign topics to each question
    topic_assignments = []
    for i, row in enumerate(W_nmf):
        topic_index = row.argmax()
        topic_name = f"Topic {topic_index+1}"
        topic_assignments.append({"question": market_questions[i], "topic": topic_name})

    return {"topics": topics, "assignments": topic_assignments}


# Set default dtype to float32 to prevent precision mismatches
torch.set_default_dtype(torch.float32)

# Load the BART summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
# summarizer = pipeline("summarization", model="t5-small")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Force CPU


def summarize_questions_by_topic(topic_assignments):
    """Generate summaries for questions in each topic."""
    summaries = {}
    grouped_questions = {}

    # Group questions by topic
    for entry in topic_assignments:
        topic = entry["topic"]
        if topic not in grouped_questions:
            grouped_questions[topic] = []
        grouped_questions[topic].append(entry["question"])

    # Generate summary per topic
    for topic, questions in grouped_questions.items():
        text = " ".join(questions).strip()

        if not text:
            summaries[topic] = "No valid text available for summarization."
            continue
        
        text = " ".join(questions).strip()
        # Ensure input does not exceed model token limit (512 for most models)
        max_input_tokens = 512
        text = " ".join(text.split()[:max_input_tokens])  # Trim text to fit model

        max_summary_length = min(150, max(50, len(text.split()) // 3))
        min_summary_length = min(30, len(text.split()) // 4)

        try:
            summary = summarizer(
                text,
                max_length=max_summary_length,
                min_length=min_summary_length,
                do_sample=False
            )[0]["summary_text"]

        except IndexError:
            summary = "Summarization failed due to input constraints."

        summaries[topic] = summary

    return summaries

def extract_ner_and_pos(words):
    """Extract named entities and relevant POS tags."""
    ner_words = set()
    pos_words = set()
    
    for word in words:
        tree = ne_chunk(pos_tag(word_tokenize(word)))
        for subtree in tree:
            if isinstance(subtree, Tree):
                ner_words.add(" ".join([token for token, pos in subtree.leaves()]))

    # Extract NOUNS and ADJECTIVES
    tagged_words = pos_tag(words)
    for word, tag in tagged_words:
        if tag in ["NN", "NNS", "NNP", "JJ"]:  # Nouns and adjectives
            pos_words.add(word)

    return list(ner_words), list(pos_words)

def extract_broad_keywords(topic_assignments):
    """Extract broad range of words per topic using different techniques."""
    keywords_per_topic = {}
    grouped_questions = {}

    for entry in topic_assignments:
        topic = entry["topic"]
        if topic not in grouped_questions:
            grouped_questions[topic] = []
        grouped_questions[topic].append(clean_text(entry["question"]))

    for topic, questions in grouped_questions.items():
        words = " ".join(questions).split()
        word_freq = Counter(words)
        most_common_words = [word for word, _ in word_freq.most_common(15)]
        
        # Apply NER and POS tagging
        ner_words, pos_words = extract_ner_and_pos(words)

        keywords_per_topic[topic] = {
            "Most Common Words": most_common_words,
            "NER Extracted Words": ner_words,
            "POS Extracted Words": pos_words
        }

    return keywords_per_topic

import time

@app.route('/nlp_analysis')
def nlp_analysis():
    """Perform extended NLP analysis on market questions."""
    print("🟢 NLP analysis started...")
    start_time = time.time()

    topic_data = classify_topics()
    summaries = summarize_questions_by_topic(topic_data["assignments"])
    keywords = extract_broad_keywords(topic_data["assignments"])

    end_time = time.time()
    print(f"🟢 NLP analysis completed in {end_time - start_time:.2f} seconds.")

    return jsonify({
        "summaries": summaries,
        "keywords": keywords
    })

@app.route('/topics')
def topics():
    """API endpoint for fetching topic modeling results."""
    return jsonify(classify_topics())

@app.route('/market_data')
def market_data():
    """Filter market data by topic category."""
    topic_filter = request.args.get('topic', None)
    market_data = get_market_data()

    if topic_filter:
        classified_data = classify_topics()["assignments"]
        filtered_questions = {q["question"]: q["topic"] for q in classified_data if q["topic"] == topic_filter}
        market_data = [entry for entry in market_data if entry["market_question"] in filtered_questions]

    return jsonify(market_data)

def analyze_sentiment(text):
    """Analyze sentiment of a question."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"
    
@app.route('/analysis')
def analysis():
    """Perform sentiment analysis and extract key words for the selected topic."""
    topic_filter = request.args.get('topic', None)
    classified_data = classify_topics()["assignments"]

    # Get questions that match the topic filter
    if topic_filter:
        filtered_questions = [q for q in classified_data if q["topic"] == topic_filter]
    else:
        filtered_questions = classified_data

    # Perform sentiment analysis
    sentiment_summary = {"Positive": 0, "Neutral": 0, "Negative": 0}
    sentiment_results = []
    
    for entry in filtered_questions:
        sentiment = analyze_sentiment(entry["question"])
        sentiment_results.append({"question": entry["question"], "topic": entry["topic"], "sentiment": sentiment})
        sentiment_summary[sentiment] += 1

    # Extract key words for the topic
    topic_words = classify_topics()["topics"].get(topic_filter, [])

    return jsonify({
        "sentiment_summary": sentiment_summary,
        "details": sentiment_results,
        "topic_words": topic_words
    })
    
JINA_API_ENDPOINT = "https://deepsearch.jina.ai/api/search"  # Update with actual endpoint
JINA_API_KEY = "jina_d10fdb7468f94f2bbe2efd79efba5edeZJCSSHl9CgD5v1FvHqQ8Mnd3bb8e"  # Replace with your actual Jina.ai API key
JINA_WORKING_DIR = '/Users/philippebeliveau/Desktop/Notebook/Prediction_market/node-DeepResearch'
# Conda environment name
CONDA_ENV = "base"

import re

def clean_jina_output(raw_output):
    """Extracts and formats key parts of Jina.ai's response."""
    
    # Extract the final answer using regex
    final_answer_match = re.search(r"Final Answer:\s*(.*?)\s*Token Usage Summary:", raw_output, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else "No answer found."

    # Extract question
    question_match = re.search(r'Gaps: \[ (.*?) \]', raw_output)
    question = question_match.group(1) if question_match else "Unknown question"

    # Extract references using regex
    references = re.findall(r'\{ exactQuote: "(.*?)", url: "(.*?)" \}', raw_output)

    # Format references
    formatted_references = [
        f'<li>"{quote}" - <a href="{url}" target="_blank">Source</a></li>' for quote, url in references
    ]

    # Build structured output
    formatted_output = (
        f"<h3>AI Response:</h3>"
        f"<p><strong>Question:</strong> {question}</p>"
        f"<p><strong>Answer:</strong> {final_answer}</p>"
    )

    if formatted_references:
        formatted_output += "<h4>References:</h4><ul>" + "".join(formatted_references) + "</ul>"

    return formatted_output

@app.route('/query_llm', methods=['POST'])
def query_llm():
    """Runs Jina.ai Deepsearch in the Conda `base` environment and returns the response."""
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        print(f"Running Jina.ai query: {user_query}")
        print(f"Using working directory: {JINA_WORKING_DIR}")

        if not os.path.exists(JINA_WORKING_DIR):
            return jsonify({"error": f"Jina working directory not found: {JINA_WORKING_DIR}"}), 500

        command = f"""
        cd {JINA_WORKING_DIR} && \
        eval "$(conda shell.zsh hook)" && \
        conda activate {CONDA_ENV} && \
        npm run dev "{user_query}"
        """
        print(f"Executing command: {command}")

        process = subprocess.run(
            ["zsh", "-i", "-c", command],
            capture_output=True,
            text=True
        )

        print(f"Subprocess Output: {process.stdout}")
        print(f"Subprocess Error: {process.stderr}")

        if process.returncode != 0:
            return jsonify({"error": "Jina.ai execution failed", "stderr": process.stderr}), 500

        response_text = process.stdout.strip()
        if not response_text:
            return jsonify({"error": "No response from Jina.ai"}), 500
        
        response_text = clean_jina_output(response_text)
        return jsonify({"response": response_text})

    except Exception as e:
        error_message = traceback.format_exc()  # Capture full traceback
        print(f"EXCEPTION ERROR:\n{error_message}")  # Print the full error in logs
        return jsonify({"error": "Failed to execute Jina.ai", "details": error_message}), 500


READER_API_URL = "http://localhost:3000"  # Local Reader API URL
import requests
@app.route('/fetch_url_content', methods=['POST'])
def fetch_url_content():
    """Fetch and convert webpage content using the Reader API."""
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    headers = {"X-Respond-With": "markdown"}  # Request markdown-formatted content

    try:
        response = requests.get(f"{READER_API_URL}/{url}", headers=headers)
        response.raise_for_status()
        return jsonify({"content": response.text})

    except requests.RequestException as e:
        return jsonify({"error": "Failed to fetch content", "details": str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# give me some career advice regarding being a laywer
# Give me a very short description of what is the profession of lawer