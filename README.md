# Uzbek Speech-to-Text and Summarization App 🎤📝

An advanced Streamlit application that converts Uzbek speech to text and provides AI-powered summarization using Wav2Vec2 and Google's Gemini AI. The app supports both audio file uploads and real-time microphone input.

## ✨ Features

- Uzbek speech-to-text conversion using Hugging Face's Wav2Vec2 model
- Real-time microphone input processing
- Audio file upload support (WAV, MP3, M4A)
- Text summarization using Google's Gemini AI
- Sentiment analysis of the transcribed text
- Interactive web interface powered by Streamlit
- Bullet-point summary generation

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install streamlit
pip install torch
pip install librosa
pip install transformers
pip install streamlit-webrtc
pip install google-generativeai
pip install python-dotenv
```

### Environment Setup

1. Create a `.env` file in the root directory
2. Add your Google API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

## 💡 How to Use

### Method 1: Audio File Upload
1. Select "Upload audio file" from the dropdown
2. Upload a supported audio file (WAV, MP3, M4A)
3. Wait for transcription and summarization
4. View results in the interface

### Method 2: Microphone Input
1. Select "Use microphone" from the dropdown
2. Grant microphone permissions
3. Speak in Uzbek
4. View real-time transcription and summary

## 🛠️ Technical Components

### Speech-to-Text
- Model: `oyqiz/uzbek_stt` (Wav2Vec2)
- Sampling Rate: 16kHz
- Audio Processing: Librosa
- Inference: PyTorch

### Text Summarization
- Model: Google Gemini-1.5-Flash
- Features:
  - Bullet-point summarization
  - Sentiment analysis
  - Key points extraction

### Audio Processing
- Real-time audio streaming
- WebRTC integration
- Custom AudioProcessor class
- Multiple audio format support

## ⚙️ Key Functions

1. `load_uzbek_stt_model()`
   - Loads Wav2Vec2 model and processor
   - Cached for performance

2. `uzbek_speech_to_text()`
   - Converts audio to waveform
   - Performs inference
   - Returns transcribed text

3. `generate_gemini_content()`
   - Processes text with Gemini AI
   - Generates summaries
   - Performs sentiment analysis

4. `AudioProcessor` Class
   - Handles real-time audio
   - Maintains audio buffer
   - Processes audio frames

## 🔧 Configuration

### Model Settings
```python
# Audio settings
sample_rate = 16000
model_name = "oyqiz/uzbek_stt"
gemini_model = "gemini-1.5-flash"
```

### Supported Audio Formats
- WAV
- MP3
- M4A

## 🔐 Security

- Environment variables for API keys
- Secure WebRTC implementation
- Local audio processing

## 🔍 Troubleshooting

1. **Audio Input Issues**
   - Check microphone permissions
   - Verify audio format compatibility
   - Ensure proper sampling rate

2. **Model Loading Issues**
   - Check internet connection
   - Verify API keys
   - Clear cache if needed

3. **Performance Issues**
   - Monitor system resources
   - Check audio file size
   - Use recommended audio formats

## 📝 Notes

- Optimized for Uzbek language
- Real-time processing capabilities
- Cached model loading for better performance
- Built-in error handling
- Memory-efficient audio processing

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- Enhanced audio processing
- UI/UX improvements
- Performance optimizations

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Hugging Face team for Wav2Vec2 model
- Google for Gemini AI
- Streamlit community
- Contributors to the Uzbek STT model

## 📞 Support

For issues and feature requests, please create an issue in the repository.