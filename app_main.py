import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
from dotenv import load_dotenv

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
    audio, _ = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, return_tensors="pt", padding="longest").input_values

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


# Use Gemini to correct, format, and summarize text
def generate_gemini_content(text, prompt_correction, prompt_summarize):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Correct and format the transcription
    correction_prompt = f"{prompt_correction}\n{text}"
    corrected_response = model.generate_content(correction_prompt)
    corrected_text = corrected_response.text.strip()

    # If corrected text is still too short, skip summarization
    if len(corrected_text) < 20:
        return corrected_text, "Summary not needed for very short transcriptions."

    # Summarize the corrected text
    summary_prompt = f"{prompt_summarize}\n{corrected_text}"
    summary_response = model.generate_content(summary_prompt)

    return corrected_text, summary_response.text.strip()


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

# Updated prompts to include meaning-based corrections
prompt_correction = "Iltimos, quyidagi transkripsiyani to'g'ri formatda yozing va noto'g'ri so'zlar mavjud bo'lsa, mazmun bo'yicha to'g'rilab, punktuatsiya bilan yozing."
prompt_summarize = "Quyidagi matnni tahlil qilib, qisqacha mazmunini chiqarib bering:"

# Choose input method: file upload or microphone
option = st.selectbox("Choose input method:", ("Upload audio file", "Use microphone"))

# Upload audio file option
if option == "Upload audio file":
    audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        st.write("Converting speech to text...")

        try:
            uzbek_text = uzbek_speech_to_text(audio_file)
            st.write("Raw Transcription:", uzbek_text)

            if uzbek_text.strip():  # Proceed only if transcription is non-empty
                st.write("Formatting and summarizing text...")
                corrected_text, summary = generate_gemini_content(uzbek_text, prompt_correction, prompt_summarize)

                st.write("Corrected & Formatted Text:", corrected_text)
                st.write("Summary:", summary)
            else:
                st.warning("The audio did not contain enough content to transcribe.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

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

            try:
                uzbek_text = uzbek_speech_to_text(audio_data)
                st.write("Raw Transcription:", uzbek_text)

                if uzbek_text.strip():  # Proceed only if transcription is non-empty
                    st.write("Formatting and summarizing text...")
                    corrected_text, summary = generate_gemini_content(uzbek_text, prompt_correction, prompt_summarize)

                    st.write("Corrected & Formatted Text:", corrected_text)
                    st.write("Summary:", summary)
                else:
                    st.warning("The audio did not contain enough content to transcribe.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Option to clear cache if needed
if st.button("Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared successfully.")

