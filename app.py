import os
import argparse
import threading
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from audio_stream import DenoiseTranscriber
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app)

args = argparse.Namespace(
        device='cpu',
        num_workers=1,
        inference=False,
        mode='file',
        audio_path=None,
        manifest_path=None,
        disable_denoiser=False,
        denoiser_dry=0.05,
        denoiser_output_save=False,
        denoiser_output_dir='./enhanced',
        denoiser_model_path='./checkpoint/denoiser.th',
        asr_model_path='./checkpoint/Conformer-CTC-BPE.nemo',
        chunk_length=1,
        context_length=1
    )

transcriber = DenoiseTranscriber(args)
lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully', 200
    else:
        return 'File type not allowed', 400

@socketio.on('start transcription')
def handle_start_transcription(data):
    audio_path = data['audio_path']
    # Check if the file exists
    if not os.path.isfile(audio_path):
        print(f'File not found: {audio_path}')
        return
    def callback(transcription):
        emit('transcription result', {'transcription': transcription})
    with lock:
        transcriber.transcribe(audio_path, callback=callback)


# handle incoming audio data
@socketio.on('audio stream')
def handle_audio_stream(data):
    # convert data to appropriate format if necessary (depends on your transcriber)
    signal = np.frombuffer(data, dtype=np.int16)
    
    # transcribe the received audio data
    transcription = transcriber.asr_decoder.transcribe_signal(signal)

    # send the transcription back to the client
    emit('transcription result', {'transcription': transcription})