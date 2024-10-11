import speech_recognition as sr


def real_time_uzbek_stt():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Open microphone for capturing audio
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Continuous loop for real-time recognition
        while True:
            try:
                # Capture audio from microphone
                audio_data = recognizer.listen(source, timeout=2)  # Adjust timeout as needed

                # Recognize speech using Google Speech Recognition
                recognized_text = recognizer.recognize_google(audio_data, language='uz-UZ')
                print("Recognized:", recognized_text)

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

            except KeyboardInterrupt:
                print("Stopping...")
                break


# Call the function to start real-time speech recognition
real_time_uzbek_stt()
