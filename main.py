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

app = FastAPI()

# Preload models and Pyannote Audio object
audio = Audio()

# Set device to CPU explicitly
device = torch.device("cpu")
whisper_model = whisper.load_model("large", device=device)

class TranscriptionSegment(BaseModel):
    speaker: str
    start: str
    text: str

class TranscriptionResponse(BaseModel):
    transcription: List[TranscriptionSegment]

def convert_to_wav(path: str) -> str:
    if not path.endswith('.wav'):
        output_path = path.rsplit('.', 1)[0] + '.wav'
        subprocess.call(['ffmpeg', '-i', path, output_path, '-y'])
        os.remove(path)  # Remove the original file
        return output_path
    return path

def transcribe_audio(path: str, model_size: str, language: str):
    if language == 'English' and model_size != 'large':
        model_name = f"{model_size}.en"
        model = whisper.load_model(model_name, device=device)
    else:
        model = whisper_model

    result = model.transcribe(path)
    return result["segments"]

def segment_embedding(segment: dict, path: str, duration: float, embedding_model):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def diarize_and_transcribe(path: str, num_speakers: int, model_size: str, language: str):
    path = convert_to_wav(path)
    segments = transcribe_audio(path, model_size, language)
    
    with contextlib.closing(wave.open(path, 'r')) as f:
        duration = f.getnframes() / float(f.getframerate())

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
    model_size: str = Form("large")
):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        segments, labels = diarize_and_transcribe(file_path, num_speakers, model_size, language)
        transcript = format_transcript(segments, labels)

        return TranscriptionResponse(transcription=transcript)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)