from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS
import whisper
import os
import openai
import time
from faster_whisper import WhisperModel


app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/transcribe": {"origins": "https://speaklish-frontend.vercel.app"}})

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your OpenAI API key

def convert_to_wav(input_file, output_path):
    start_time = time.time()
    if os.path.exists(output_path):
        os.remove(output_path)
    command = ['ffmpeg', '-i', input_file, '-ar', '12000', '-ac', '1', '-b:a', '16k', output_path]
    subprocess.run(command, check=True)
    elapsed_time = time.time() - start_time
    print("time to convert file ", elapsed_time)

def transcribe_audio(audio_path):
    start_time = time.time()
    faster_whisper_model = WhisperModel("small", device="cpu", compute_type="float32")
    segments, _ = faster_whisper_model.transcribe(audio_path)
    segments = list(segments)
    elapsed_time = time.time() - start_time
    print("time to transcribe audio ", elapsed_time)
    return segments[0].text

def get_chatgpt_response(prompt):
    start_time = time.time()
    chat_gpt_model = os.getenv('CHATGPT_MODEL')
    stream = openai.chat.completions.create(
        model=chat_gpt_model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    elapsed_time = time.time() - start_time
    print("time to get gpt response ", elapsed_time)
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
        transcription =  transcribe_audio(output_path) 
        os.remove(input_path)
        os.remove(output_path)

        # Append the transcription to a predefined prompt
        predefined_text = "Can you judge my spoken english based on various parameters like grammar, use of phrases, active passive voices, direct/ indirect speeches etc. Give descriptive qualitative feedback on each parameter, in which area I can improve, as well as quantitive score out of 10?  \n\n"  # Placeholder text
        prompt = predefined_text + transcription

        # Get response from OpenAI GPT-3
        chatgpt_response = get_chatgpt_response(prompt)

        return jsonify(transcription=transcription, chatgpt_response=chatgpt_response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
