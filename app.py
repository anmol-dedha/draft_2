import streamlit as st
import requests, json
import whisper
import tempfile
import os
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Load OpenRouter API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528:free"

st.set_page_config(page_title="AnnaData Voice Assistant", page_icon="🌱")
st.title("🧑‍🌾 AnnaData - किसानों की सेवा में")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Voice input
st.subheader("🎙️ Voice Input (optional)")
audio_file = st.file_uploader("अपना सवाल रिकॉर्ड करें (mp3/wav)", type=["mp3", "wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    # Load Whisper model
    model = whisper.load_model("small")
    result = model.transcribe(tmp_file_path, language="hi")  # auto-detect language
    voice_text = result["text"]
    st.markdown(f"**आपने कहा:** {voice_text}")

    # Append to chat
    st.session_state.messages.append({"role": "user", "content": voice_text})
    prompt = voice_text
else:
    # Chat input
    prompt = st.chat_input("अपना सवाल लिखें...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call OpenRouter API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

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

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Convert assistant response to speech
        tts = gTTS(full_response, lang="hi")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")
