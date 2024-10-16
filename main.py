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
from typing import List
from pydantic import BaseModel
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Preload models and Pyannote Audio object
audio = Audio()

# Set device to CPU explicitly
device = torch.device("cpu")

class TranscriptionSegment(BaseModel):
    speaker: str
    start: str
    text: str

class TranscriptionResponse(BaseModel):
    transcription: List[TranscriptionSegment]

def convert_to_wav(path: str) -> str:
    if not path.endswith('.wav'):
        output_path = path.rsplit('.', 1)[0] + '.wav'
        logger.info(f"Converting {path} to {output_path}")
        # Specify audio codec and sample rate
        subprocess.call(['ffmpeg', '-i', path, '-acodec', 'pcm_s16le', '-ar', '16000', output_path, '-y'])
        os.remove(path)  # Remove the original file
        logger.info(f"Removed original file: {path}")
        return output_path
    return path

def transcribe_audio(path: str, model_name: str, language: str):
    logger.info(f"Transcribing audio from {path} with model {model_name} and language {language}")
    
    try:
        # Check if the specified model is available
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}'. Please check the model name.")
    
    result = model.transcribe(path, language=language)  # Pass the language parameter if needed
    return result["segments"]

def segment_embedding(segment: dict, path: str, duration: float, embedding_model):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def diarize_and_transcribe(path: str, num_speakers: int, model_name: str, language: str):
    path = convert_to_wav(path)
    segments = transcribe_audio(path, model_name, language)
    
    with contextlib.closing(wave.open(path, 'r')) as f:
        duration = f.getnframes() / float(f.getframerate())
        logger.info(f"Audio duration: {duration:.2f} seconds")

    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", 
        device=device
    )

    embeddings = np.array([segment_embedding(segment, path, duration, embedding_model) for segment in segments])
    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    
    return segments, clustering.labels_

def format_transcript(segments: List[dict], labels: np.ndarray) -> List[TranscriptionSegment]:
    transcript = []
    for i, segment in enumerate(segments):
        speaker = f'SPEAKER {labels[i] + 1}'
        if i == 0 or speaker != f'SPEAKER {labels[i-1] + 1}':
            transcript.append(TranscriptionSegment(
                speaker=speaker,
                start=str(datetime.timedelta(seconds=round(segment["start"]))),
                text=segment["text"].strip()
            ))
        else:
            transcript[-1].text += " " + segment["text"].strip()
    return transcript

@app.post("/transcribe_and_diarize/", response_model=TranscriptionResponse)
async def transcribe_and_diarize(
    file: UploadFile = File(...),
    num_speakers: int = Form(...),
    language: str = Form("any"),
    model_name: str = Form(...)  # User must specify model_name
):
    logger.info("Received a file upload for transcription and diarization")
    
    if not isinstance(num_speakers, int) or num_speakers <= 0:
        raise HTTPException(status_code=400, detail="Number of speakers must be a positive integer.")

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File saved to {file_path}")

        segments, labels = diarize_and_transcribe(file_path, num_speakers, model_name, language)
        transcript = format_transcript(segments, labels)

        logger.info("Transcription and diarization completed successfully.")
        return TranscriptionResponse(transcription=transcript)
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
