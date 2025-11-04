# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
import plotly.graph_objects as go

# Custom USE Embedding Layer
# -------------------------------
class USE_Embedding(tf.keras.layers.Layer):
    def __init__(self, link="https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = hub.KerasLayer(link, trainable=trainable)

    def call(self, inputs):
        return self.embedding_layer(inputs)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_sentiment_model():
    model = tf.keras.models.load_model(
        "sentiment_model.h5",
        custom_objects={
            'USE_Embedding': USE_Embedding,
            'KerasLayer': hub.KerasLayer
        }
    )
    return model

model = load_sentiment_model()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="AI Sentiment Analysis", page_icon="💬", layout="wide")

# -------------------------------
# Custom Dark CSS
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #E1E1E1;
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #0E1117;
        color: #E1E1E1;
    }
    .title {
        text-align: center;
        color: #4DD0E1;
        font-size: 2.4em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 0 0 10px #4DD0E1;
    }
    .subtitle {
        text-align: center;
        color: #bbb;
        font-size: 1.1em;
        margin-bottom: 40px;
    }
    .result-card {
        background: linear-gradient(135deg, #1B263B 0%, #0E1117 100%);
        border: 1px solid #4DD0E1;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0px 0px 12px rgba(77, 208, 225, 0.3);
        text-align: center;
        margin-top: 20px;
    }
    .positive {
        color: #4DD0E1;
        font-size: 1.6em;
        font-weight: bold;
        text-shadow: 0 0 8px #4DD0E1;
    }
    .negative {
        color: #FF4081;
        font-size: 1.6em;
        font-weight: bold;
        text-shadow: 0 0 8px #FF4081;
    }
    .metric-box {
        background: #141821;
        border: 1px solid #2D3748;
        border-radius: 12px;
        text-align: center;
        padding: 15px;
        color: #EAEAEA;
    }
    .stTextArea textarea {
        background-color: #141821 !important;
        color: white !important;
        border-radius: 10px;
        border: 1px solid #4DD0E1;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 class='title'>💬 Amazon Fine Food Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>An intelligent NLP model built with Transfer Learning on a pre-trained Universal Sentence Encoder (USE) from TensorFlow Hub, designed to analyze review texts and predict whether the sentiment expressed is Positive 😊 or Negative 😠..</p>", unsafe_allow_html=True)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("✍️ Enter your review below:", height=150, placeholder="Example: The meal was absolutely fantastic, will definitely come again!")

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if user_input.strip():
        input_data = tf.constant([user_input])
        prediction = model.predict(input_data)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "😠 Negative"

        word_count = len(user_input.split())
        char_count = len(user_input)

        if prediction > 0.5:
            st.markdown(f"""
                <div class='result-card'>
                    <p class='positive'>✅ {sentiment}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-card'>
                    <p class='negative'>❌ {sentiment}</p>
                </div>
            """, unsafe_allow_html=True)


        # Confidence Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence (%)", 'font': {'color': '#E1E1E1'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#E1E1E1'},
                'bar': {'color': "#4DD0E1"},
                'bgcolor': "black",
                'steps': [
                    {'range': [0, 50], 'color': "#3E1F47"},
                    {'range': [50, 100], 'color': "#10323E"}
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="#0E1117", font={'color': '#E1E1E1'})
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-box'><h4>📝 Words</h4><h3>{word_count}</h3></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><h4>🔠 Characters</h4><h3>{char_count}</h3></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>© 2025 <b>Ahmed Shlaby</b> — Built with ❤️ using <b>Transfer Learning</b> on TensorFlow Hub (USE) and deployed via <b>Streamlit</b></p>", unsafe_allow_html=True)
