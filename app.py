import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import tempfile
from gtts import gTTS
from io import BytesIO
import requests
import os

# Load OpenRouter API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528:free"

st.set_page_config(page_title="AnnaData Voice Assistant", page_icon="🌱")
st.title("🧑‍🌾 AnnaData - किसानों की सेवा में")

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

model = load_whisper_model()

# Voice input via microphone
st.subheader("🎙️ Voice Input (click to start)")
webrtc_ctx = webrtc_streamer(key="voice-input", mode=WebRtcMode.SENDRECV)

if webrtc_ctx.audio_receiver:
    # Collect audio chunks
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
    if audio_frames:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            for frame in audio_frames:
                tmp_file.write(frame.to_ndarray().tobytes())
            tmp_file_path = tmp_file.name

        # Transcribe audio
        result = model.transcribe(tmp_file_path, language="hi")
        user_text = result["text"]
        st.markdown(f"**आपने कहा:** {user_text}")

        # Call OpenRouter API
        messages = [{"role": "user", "content": user_text}]
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages},
        )
        assistant_text = ""
        if response.status_code == 200:
            data = response.json()
            assistant_text = data["choices"][0]["message"]["content"]

        st.markdown(f"**सहायक:** {assistant_text}")

        # Convert assistant response to speech
        tts = gTTS(assistant_text, lang="hi")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")
