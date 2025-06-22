
import os
import streamlit as st
import torch
from transformers import pipeline
import tempfile

# Ignore unnecessary module reloads
os.environ["STREAMLIT_WATCHER_IGNORE_MODULES"] = "torch,torch.*"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Page setup
st.set_page_config(
    page_title="Audio Describer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern UI
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f8f9fc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        border-radius: 12px;
        background-color: #4F46E5;
        color: white;
        font-size: 16px;
        padding: 0.6em 1.2em;
    }
    .stTextArea textarea {
        border-radius: 12px;
        font-family: 'Courier New', monospace;
    }
    .stSuccess {
        background-color: #f0f4ff;
        color: #1d3557;
        border-left: 6px solid #4F46E5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.image("logo.png", width=160)  # Replace with your logo filename
st.sidebar.title("🎛️ Audio Describer")
st.sidebar.markdown("**Version:** Audio Version 1")
st.sidebar.markdown("---")
st.sidebar.markdown("Upload, Transcribe, Summarize")

# Main title
st.title("🎧 Audio Describer")
st.caption("Turn your audio into clean, timestamped text and concise summaries with AI.")

# Upload audio
audio_file = st.file_uploader("📤 Upload an audio file (.mp3, .ogg, .wav)", type=["mp3", "ogg", "wav"])

# Load Whisper once
@st.cache_resource
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device,
        generate_kwargs={"language": "English", "task": "translate"}
    )

# Load summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == 'cuda' else -1)

# Main logic
if audio_file:
    st.audio(audio_file, format="audio/wav")

    if st.button("📝 Describe Audio"):
        with st.spinner("🔄 Transcribing... please wait"):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            # Transcribe
            whisper_pipe = load_whisper()
            transcription = whisper_pipe(tmp_path)
            os.remove(tmp_path)

            # Format output
            formatted_lyrics = ""
            for line in transcription['chunks']:
                text = line["text"]
                ts = line["timestamp"]
                formatted_lyrics += f"{ts} --> {text}\n"

            full_text = " ".join(chunk["text"] for chunk in transcription["chunks"])

        st.subheader("📄 Transcription with Timestamps")
        st.text_area("Transcription", formatted_lyrics.strip(), height=300)

        with st.spinner("🧠 Summarizing..."):
            summarizer = load_summarizer()
            summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

        st.subheader("🧠 Summary")
        st.success(summary)
