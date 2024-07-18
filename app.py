from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS
import whisper
import os
import openai

app = Flask(__name__)
CORS(app)

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your OpenAI API key

def convert_to_wav(input_file, output_file):
    command = ['ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', output_file]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def get_chatgpt_response(prompt):
    stream = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return stream.choices[0].message.content

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    if file:
        input_path = os.path.join('/tmp', file.filename)
        file.save(input_path)
        output_path = os.path.splitext(input_path)[0] + '.wav'
        convert_to_wav(input_path, output_path)
        transcription =  transcribe_audio(output_path) #"Let me tell you something about solar system. So our solar system is one of the many bodies presented in the Milky Way. In the center of the solar system, we have sun, which is a medium-sized star. We have a total of nine planets starting from Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Jupiter is one of the biggest planets in the solar system. Earth is one of the most unique planets in the solar system, which sustains life and has water, oxygen, and has an atmosphere which can sustain life. Yeah, that's it."
        os.remove(input_path)
        os.remove(output_path)

        # Append the transcription to a predefined prompt
        predefined_text = "Can you judge my spoken english based on various parameters like grammar, use of phrases, active passive voices, direct/ indirect speeches etc. Give descriptive qualitative feedback on each parameter, in which area I can improve, as well as quantitive score out of 10?  \n\n"  # Placeholder text
        prompt = predefined_text + transcription

        # Get response from OpenAI GPT-3
        chatgpt_response = get_chatgpt_response(prompt)

        return jsonify(transcription=transcription, chatgpt_response=chatgpt_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
