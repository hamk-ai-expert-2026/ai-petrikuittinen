from google import genai
import base64

client = genai.Client()

uploaded_file = client.files.upload(file="ai_news.wav")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {"type": "text", "text": "Transcribe the following audio file to text and translate it to Finnish.  "},
        {
            "type": "audio",
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type
        }
    ]
)
print(interaction.output_text)