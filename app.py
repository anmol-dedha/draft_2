import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import whisper
from gtts import gTTS
from io import BytesIO
import requests
import json
import tempfile
import numpy as np
import soundfile as sf

# Load Whisper model once
model = whisper.load_model("tiny")

# OpenRouter API key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528:free"

st.title("🧑‍🌾 AnnaData Voice Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.subheader("🎙️ Click 'Start Microphone' and speak")

# Start microphone
webrtc_ctx = webrtc_streamer(
    key="voice-assistant",
    mode=WebRtcMode.SENDONLY,  # just record audio
    audio_receiver_size=1024,
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
    if audio_frames:
        # Convert audio frames to numpy array
        audio_data = np.hstack([f.to_ndarray() for f in audio_frames])
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_file.name, audio_data, 16000)  # 16kHz sampling

        # Transcribe with Whisper
        result = model.transcribe(tmp_file.name, language="hi")
        user_text = result["text"]
        st.markdown(f"**आपने कहा:** {user_text}")

        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Call OpenRouter API
        messages = st.session_state.messages
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages},
        )
        if response.status_code == 200:
            full_response = response.json()["choices"][0]["message"]["content"]
        else:
            full_response = "⚠️ Error fetching response."

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.markdown(f"**सहायक:** {full_response}")

        # Convert assistant response to speech
        tts = gTTS(full_response, lang="hi")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")
