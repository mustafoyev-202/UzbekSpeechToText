import streamlit as st
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Load the Hugging Face model and processor for Uzbek speech-to-text
@st.cache_resource
def load_uzbek_stt_model():
    processor = Wav2Vec2Processor.from_pretrained("oyqiz/uzbek_stt")
    model = Wav2Vec2ForCTC.from_pretrained("oyqiz/uzbek_stt")
    return processor, model


# Function to perform speech-to-text using 'oyqiz/uzbek_stt'
def uzbek_speech_to_text(audio_file):
    processor, model = load_uzbek_stt_model()

    # Convert UploadedFile to a waveform using librosa
    audio, _ = librosa.load(audio_file, sr=16000)

    # Prepare the audio input for the Wav2Vec2 model
    input_values = processor(audio, return_tensors="pt", padding="longest").input_values

    # Perform inference (Uzbek speech-to-text)
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Convert the predicted IDs to transcribed text
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


# Use Gemini to summarize text
def generate_gemini_content(text, prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Advanced prompt for summarization and analysis
    full_prompt = (
        f"{prompt}\n"
        "Matnni qisqartirib, asosiy fikrlarni bullet point ko'rinishida taqdim qiling. "
        "Matnni ijobiy yoki salbiy ekanligini ham aniqlang."
    )
    response = model.generate_content(full_prompt + text)
    return response.text


# Custom audio processor class for microphone input
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.sr = 16000
        self.audio_data = np.zeros((0,), dtype=np.float32)

    def recv(self, frame):
        audio_data = np.frombuffer(frame.to_ndarray().flatten(), dtype=np.float32)
        self.audio_data = np.concatenate((self.audio_data, audio_data))
        return frame

    def get_audio_data(self):
        return self.audio_data


# Streamlit UI
st.title("Uzbek Speech-to-Text and Summarization App")

# Choose input method: file upload or microphone
option = st.selectbox("Choose input method:", ("Upload audio file", "Use microphone"))

# Upload audio file option
if option == "Upload audio file":
    audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        st.write("Converting speech to text...")

        # Perform speech-to-text conversion using 'oyqiz/uzbek_stt'
        uzbek_text = uzbek_speech_to_text(audio_file)
        st.write("Transcribed Text:", uzbek_text)

        # Perform summarization with Gemini
        st.write("Summarizing text...")
        prompt = "Quyidagi matnni tahlil qilib, qisqacha mazmunini chiqarib bering:"
        summary = generate_gemini_content(uzbek_text, prompt)
        st.write("Summary:", summary)

# Microphone input option
elif option == "Use microphone":
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        audio_data = webrtc_ctx.audio_processor.get_audio_data()
        if len(audio_data) > 0:
            st.write("Converting speech to text...")

            # Perform speech-to-text conversion using 'oyqiz/uzbek_stt'
            uzbek_text = uzbek_speech_to_text(audio_data)
            st.write("Transcribed Text:", uzbek_text)

            # Perform summarization with Gemini
            st.write("Summarizing text...")
            prompt = "Quyidagi matnni tahlil qilib, qisqacha mazmunini chiqarib bering:"
            summary = generate_gemini_content(uzbek_text, prompt)
            st.write("Summary:", summary)

# Option to clear cache if needed
if st.button("Clear Cache"):
    st.cache_resource.clear()
