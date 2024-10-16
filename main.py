from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import whisper
import torch
import subprocess
import wave
import contextlib
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
import datetime
import os

app = FastAPI()

# Preload models and Pyannote Audio object
audio = Audio()

# Set device to CPU explicitly
device = torch.device("cpu")
whisper_model = whisper.load_model("large", device=device)

def convert_to_wav(path):
    # Convert non-WAV files to WAV format using ffmpeg
    if not path.endswith('.wav'):
        output_path = "temp_audio.wav"
        subprocess.call(['ffmpeg', '-i', path, output_path, '-y'])
        return output_path
    return path

def transcribe_audio(path, model_size, language):
    # Load the model dynamically based on language and model size
    if language == 'English' and model_size != 'large':
        model_name = model_size + '.en'
        model = whisper.load_model(model_name, device=device)
    else:
        model = whisper_model  # Use preloaded large model for other cases

    # Transcribe the audio
    path = convert_to_wav(path)
    result = model.transcribe(path)
    segments = result["segments"]
    return segments, path

def segment_embedding(segment, path, duration, embedding_model):
    # Extract embeddings for each segment
    start = segment["start"]
    end = min(duration, segment["end"])  # Avoid overshooting in the last segment
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def diarize_and_transcribe(path, num_speakers, model_size, language):
    # Perform transcription and diarization
    segments, path = transcribe_audio(path, model_size, language)
    
    # Open the WAV file to get duration
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Load speaker embedding model with CPU support
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", 
        device=device  # Explicitly set to CPU
    )

    # Create an empty matrix for embeddings
    embeddings = np.zeros(shape=(len(segments), 192))

    # Extract embeddings for each segment
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, path, duration, embedding_model)

    # Handle NaN values in embeddings
    embeddings = np.nan_to_num(embeddings)

    # Perform speaker clustering
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_

    return segments, labels

def format_transcript(segments, labels):
    # Format transcription with speaker labels
    transcript = []

    def time(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    for i, segment in enumerate(segments):
        speaker = 'SPEAKER ' + str(labels[i] + 1)
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            transcript.append({
                "speaker": speaker,
                "start": time(segment["start"]),
                "text": segment["text"].strip()  # Remove leading space
            })
        else:
            transcript[-1]["text"] += " " + segment["text"].strip()

    return transcript

@app.post("/transcribe_and_diarize/")
async def transcribe_and_diarize(
    file: UploadFile = File(...),
    num_speakers: int = Form(...),
    language: str = Form("any"),
    model_size: str = Form("large")
):
    # Create temporary directory if it doesn't exist
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = f"{temp_dir}/{file.filename}"
    
    try:
        # Save uploaded file to a temporary location
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Perform transcription and diarization
        segments, labels = diarize_and_transcribe(file_path, num_speakers, model_size, language)

        # Format transcript with speaker information
        transcript = format_transcript(segments, labels)

        # Send JSON response with transcription and diarization results
        return JSONResponse(content={"transcription": transcript})
    
    except Exception as e:
        # If any error occurs during the process, return 500 with error message
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

        # Optionally remove temp directory if empty
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
