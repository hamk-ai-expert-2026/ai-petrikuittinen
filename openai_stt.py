from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="gpt-4o-transcribe",
  file=audio_file
)
print(transcript.text)