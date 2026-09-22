from pathlib import Path
import openai

speech_file_path = Path(__file__).parent / "speech.mp3"
with openai.audio.speech.with_streaming_response.create(
  model="gpt-4o-mini-tts",
  voice="alloy",
  input="Hei! Mitä Miten päiväsi on sujunut? Toivottavasti hyvin!"
) as response:
  response.stream_to_file(speech_file_path)