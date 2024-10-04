import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import google.generativeai as genai


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
