from fastapi import FastAPI, HTTPException, Form
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
import requests

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

def download_audio_from_url(url: str, output_dir: str) -> str:
    file_name = url.split("/")[-1]
    file_path = os.path.join(output_dir, file_name)
    
    logger.info(f"Downloading audio file from {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        logger.error(f"Failed to download file from {url}")
        raise HTTPException(status_code=400, detail="Failed to download file from the provided URL.")
    
    with open(file_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded and saved to {file_path}")
    
    return file_path

def convert_to_wav(path: str) -> str:
    if not path.endswith('.wav'):
        output_path = path.rsplit('.', 1)[0] + '.wav'
        logger.info(f"Converting {path} to {output_path}")
        subprocess.call(['ffmpeg', '-i', path, '-acodec', 'pcm_s16le', '-ar', '16000', output_path, '-y'])
        os.remove(path)  # Remove the original file
        logger.info(f"Removed original file: {path}")
        return output_path
    return path

def transcribe_audio(path: str, model_name: str, language: str):
    logger.info(f"Transcribing audio from {path} with model {model_name} and language {language}")
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(path, language=language)
    return result["segments"]

def segment_embedding(segment: dict, path: str, duration: float, embedding_model):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    embedding = embedding_model(waveform[None])
    return embedding.squeeze()

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

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    
    for i in range(len(segments)):
        segments[i]["speaker"] = f'SPEAKER {labels[i] + 1}'
    
    return segments, labels

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
    audio_url: str = Form(...),
    num_speakers: int = Form(...),
    language: str = Form("any"),
    model_name: str = Form(...)  
):
    logger.info("Received a request for transcription and diarization with S3 link")
    
    if not isinstance(num_speakers, int) or num_speakers <= 0:
        raise HTTPException(status_code=400, detail="Number of speakers must be a positive integer.")

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        file_path = download_audio_from_url(audio_url, temp_dir)
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
