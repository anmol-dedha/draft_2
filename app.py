import streamlit as st
import requests, json
import whisper
from gtts import gTTS
import tempfile
import os
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="AnnaData Voice Assistant", page_icon="🌱")
st.title("🧑‍🌾 AnnaData - किसानों की सेवा में")

# -------------------------------
# OpenRouter Setup
# -------------------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528:free"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Load Whisper STT model
# -------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")  # small = fast & accurate

whisper_model = load_whisper_model()

def transcribe_audio_file(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

# -------------------------------
# Display past messages
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Input Mode Selector
# -------------------------------
voice_mode = st.checkbox("🎤 Use Voice Input")

prompt = None

# -------------------------------
# Handle Voice Input (Live Mic)
# -------------------------------
if voice_mode:
    st.write("🎙️ Press start and speak your query:")

    ctx = webrtc_streamer(
        key="live-mic",
        mode=WebRtcMode.SENDRECV,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False}
        ),
    )

    if ctx.audio_receiver:
        audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_np = np.concatenate([f.to_ndarray() for f in audio_frames])
            sf.write(temp_wav.name, audio_np, 48000)
            prompt = transcribe_audio_file(temp_wav.name)
            st.markdown(f"**आपने कहा:** {prompt}")
            os.remove(temp_wav.name)

# -------------------------------
# Handle Text Input
# -------------------------------
if not voice_mode or prompt is None:
    prompt = st.chat_input("अपना सवाल लिखें...")

# -------------------------------
# Call OpenRouter API
# -------------------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": st.session_state.messages,
                    "stream": True,
                },
                stream=True,
            ) as r:
                for line in r.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[len(b"data: "):].decode("utf-8")
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json["choices"][0]["delta"].get("content", "")
                            full_response += delta
                            placeholder.markdown(full_response)
                        except Exception as e:
                            placeholder.markdown(f"⚠️ Error parsing: {e}")

        except Exception as e:
            placeholder.markdown(f"⚠️ API Error: {e}")

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # -------------------------------
        # Text-to-Speech (TTS) using gTTS
        # -------------------------------
        try:
            tts = gTTS(text=full_response, lang="hi")  # 'hi' for Hindi, 'en' for English
            tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tts_file.name)
            st.audio(tts_file.name)
            tts_file.close()
            os.remove(tts_file.name)
        except Exception as e:
            st.warning(f"⚠️ TTS Error: {e}")
