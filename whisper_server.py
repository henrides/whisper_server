#!/usr/bin/env python3
import os
# Disable torchaudio backend check for speechbrain compatibility
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import asyncio
import json
import numpy as np
import whisper
import websockets
import torch

# Monkey-patch torchaudio to fix speechbrain compatibility issue
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: []

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Global variables for models
model = None
speaker_encoder = None

# Speaker tracking database (in-memory)
# Structure: list of dicts with {'id': 'SPEAKER_0', 'embeddings': [np.array, ...]}
# Store multiple embeddings per speaker for better matching
speaker_database = []

# Audio parameters
SAMPLE_RATE = 16000
CHUNK_BUFFER_SIZE = 50  # Number of chunks to accumulate before transcribing

# Speaker diarization parameters
EMBEDDING_WINDOW_SIZE = 3.0  # seconds
EMBEDDING_WINDOW_OVERLAP = 1.0  # seconds
SPEAKER_SIMILARITY_THRESHOLD = 0.4  # cosine similarity threshold for speaker matching (lowered further)
CLUSTERING_DISTANCE_THRESHOLD = 0.5  # distance threshold for clustering (lowered to group similar speakers)
MAX_EMBEDDINGS_PER_SPEAKER = 10  # Maximum embeddings to store per speaker

def extract_speaker_embeddings(audio_data, sample_rate):
    """
    Extract speaker embeddings from audio using sliding windows.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: audio sample rate
    
    Returns:
        list of dicts with {'embedding': np.array, 'start': float, 'end': float}
    """
    window_samples = int(EMBEDDING_WINDOW_SIZE * sample_rate)
    overlap_samples = int(EMBEDDING_WINDOW_OVERLAP * sample_rate)
    step_samples = window_samples - overlap_samples
    
    embeddings = []
    audio_length = len(audio_data)
    
    for start_sample in range(0, audio_length - window_samples + 1, step_samples):
        end_sample = start_sample + window_samples
        window = audio_data[start_sample:end_sample]
        
        # Convert to torch tensor and add batch dimension
        window_tensor = torch.tensor(window).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = speaker_encoder.encode_batch(window_tensor)
            embedding = embedding.squeeze().cpu().numpy()
            # SpeechBrain embeddings should already be normalized, but ensure it
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Calculate timestamps
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        
        embeddings.append({
            'embedding': embedding,
            'start': start_time,
            'end': end_time
        })
    
    return embeddings

def match_speaker_to_database(embedding, threshold=SPEAKER_SIMILARITY_THRESHOLD):
    """
    Match an embedding to existing speakers in the database.
    
    Args:
        embedding: numpy array of speaker embedding (normalized)
        threshold: cosine similarity threshold for matching
    
    Returns:
        speaker_id (str) if match found, None otherwise
    """
    if not speaker_database:
        return None
    
    best_speaker_id = None
    best_similarity = -1
    
    # Compare against ALL embeddings from ALL speakers
    for speaker in speaker_database:
        speaker_embeddings = np.array(speaker['embeddings'])
        similarities = cosine_similarity([embedding], speaker_embeddings)[0]
        max_similarity = np.max(similarities)
        
        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_speaker_id = speaker['id']
    
    print(f"  Speaker matching: best similarity = {best_similarity:.3f} (threshold = {threshold:.3f})")
    
    if best_similarity >= threshold:
        print(f"  Matched to existing {best_speaker_id}")
        return best_speaker_id
    
    print(f"  No match found, will create new speaker")
    return None

def cluster_and_label_speakers(embeddings_list):
    """
    Cluster embeddings and assign speaker labels, matching with existing database.
    
    Args:
        embeddings_list: list of dicts with embeddings and timestamps
    
    Returns:
        list of dicts with added 'speaker_id' field
    """
    if not embeddings_list:
        return []
    
    # Extract just the embeddings for clustering
    embeddings = np.array([e['embedding'] for e in embeddings_list])
    
    # Handle single embedding case (can't cluster)
    if len(embeddings) == 1:
        representative_embedding = embeddings[0]
        # Ensure normalized
        norm = np.linalg.norm(representative_embedding)
        if norm > 0:
            representative_embedding = representative_embedding / norm
        
        speaker_id = match_speaker_to_database(representative_embedding)
        
        if speaker_id is None:
            speaker_id = f"SPEAKER_{len(speaker_database)}"
            speaker_database.append({
                'id': speaker_id,
                'embeddings': [representative_embedding]
            })
            print(f"  Created new {speaker_id}")
        else:
            # Add this embedding to the matched speaker's collection
            for speaker in speaker_database:
                if speaker['id'] == speaker_id:
                    if len(speaker['embeddings']) < MAX_EMBEDDINGS_PER_SPEAKER:
                        speaker['embeddings'].append(representative_embedding)
                    break
        
        embeddings_list[0]['speaker_id'] = speaker_id
        return embeddings_list
    
    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTERING_DISTANCE_THRESHOLD,
        metric='cosine',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Map cluster labels to global speaker IDs
    cluster_to_speaker = {}
    
    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label not in cluster_to_speaker:
            # Get representative embedding for this cluster (mean of all embeddings in cluster)
            cluster_mask = cluster_labels == cluster_label
            cluster_embeddings = embeddings[cluster_mask]
            representative_embedding = np.mean(cluster_embeddings, axis=0)
            # Normalize the representative embedding
            norm = np.linalg.norm(representative_embedding)
            if norm > 0:
                representative_embedding = representative_embedding / norm
            
            # Try to match with existing speaker in database
            speaker_id = match_speaker_to_database(representative_embedding)
            
            if speaker_id is None:
                # Create new speaker
                speaker_id = f"SPEAKER_{len(speaker_database)}"
                speaker_database.append({
                    'id': speaker_id,
                    'embeddings': [representative_embedding]
                })
            else:
                # Add this embedding to the matched speaker's collection
                for speaker in speaker_database:
                    if speaker['id'] == speaker_id:
                        if len(speaker['embeddings']) < MAX_EMBEDDINGS_PER_SPEAKER:
                            speaker['embeddings'].append(representative_embedding)
                        break
            
            cluster_to_speaker[cluster_label] = speaker_id
    
    # Assign speaker IDs to all embeddings
    for i, embedding_info in enumerate(embeddings_list):
        embedding_info['speaker_id'] = cluster_to_speaker[cluster_labels[i]]
    
    return embeddings_list

def align_speakers_with_transcription(speaker_segments, transcription_segments):
    """
    Align speaker segments with transcription segments by time overlap.
    
    Args:
        speaker_segments: list of dicts with speaker_id, start, end
        transcription_segments: list of dicts with text, start, end from Whisper
    
    Returns:
        list of dicts with speaker, text, start, end
    """
    aligned_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        trans_mid = (trans_start + trans_end) / 2
        
        # Find speaker segment with maximum overlap
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0
        
        for spk_seg in speaker_segments:
            spk_start = spk_seg['start']
            spk_end = spk_seg['end']
            
            # Calculate overlap
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = spk_seg['speaker_id']
        
        # Also check if transcription midpoint falls within speaker segment
        if max_overlap == 0:
            for spk_seg in speaker_segments:
                if spk_seg['start'] <= trans_mid <= spk_seg['end']:
                    best_speaker = spk_seg['speaker_id']
                    break
        
        aligned_segments.append({
            'speaker': best_speaker,
            'text': trans_seg['text'].strip(),
            'start': trans_start,
            'end': trans_end
        })
    
    return aligned_segments

async def handle_client(websocket):
    """Handle a single client connection."""
    print(f"Client connected: {websocket.remote_address}")
    
    audio_buffer = []
    sample_rate = SAMPLE_RATE
    dtype = np.float32
    
    try:
        async for message in websocket:
            # Handle different message types
            if isinstance(message, str):
                # JSON message (config handshake)
                data = json.loads(message)
                if data.get("type") == "config":
                    sample_rate = data.get("sample_rate", SAMPLE_RATE)
                    dtype_str = data.get("dtype", "float32")
                    dtype = np.dtype(dtype_str)
                    print(f"Config received: sample_rate={sample_rate}, dtype={dtype}")
                    # Send acknowledgment
                    await websocket.send(json.dumps({"type": "ack", "status": "ready"}))
            
            elif isinstance(message, bytes):
                # Binary message (audio data)
                audio_chunk = np.frombuffer(message, dtype=dtype)
                audio_buffer.append(audio_chunk)
                
                # Transcribe when buffer is full enough
                if len(audio_buffer) >= CHUNK_BUFFER_SIZE:
                    print(f"Processing {len(audio_buffer)} chunks...")
                    
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(audio_buffer)
                    audio_flat = audio_data.flatten()
                    
                    # Extract speaker embeddings
                    print("Extracting speaker embeddings...")
                    speaker_segments = await asyncio.to_thread(
                        extract_speaker_embeddings,
                        audio_flat,
                        sample_rate
                    )
                    
                    # Cluster and label speakers
                    print("Clustering speakers...")
                    speaker_segments = await asyncio.to_thread(
                        cluster_and_label_speakers,
                        speaker_segments
                    )
                    
                    # Run Whisper transcription with timestamps
                    print("Transcribing...")
                    result = await asyncio.to_thread(
                        model.transcribe,
                        audio_flat,
                        language="en",
                        verbose=False
                    )
                    
                    # Get segments with timestamps
                    transcription_segments = result.get('segments', [])
                    
                    if transcription_segments:
                        # Align speakers with transcription
                        print("Aligning speakers with transcription...")
                        aligned_segments = align_speakers_with_transcription(
                            speaker_segments,
                            transcription_segments
                        )
                        
                        # Filter out empty segments
                        aligned_segments = [s for s in aligned_segments if s['text']]
                        
                        if aligned_segments:
                            print(f"Transcription: {len(aligned_segments)} segments")
                            for seg in aligned_segments:
                                print(f"  [{seg['speaker']}] {seg['text']}")
                            
                            # Send segmented transcription result back to client
                            await websocket.send(json.dumps({
                                "type": "transcription",
                                "segments": aligned_segments
                            }))
                    
                    # Clear buffer after transcription
                    audio_buffer.clear()
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        print("Connection closed")

async def main():
    """Start the WebSocket server."""
    global model, speaker_encoder
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("tiny.en")
    print("Whisper model loaded successfully!")
    
    # Load SpeechBrain speaker encoder
    print("Loading speaker encoder model...")
    print("(First run will download ~50MB model, subsequent runs use cache)")
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./speaker_models",
        run_opts={"device": "cpu"}
    )
    print("Speaker encoder loaded successfully!")
    
    print("\nStarting WebSocket server on localhost:8765...")
    async with websockets.serve(handle_client, "localhost", 8765):
        print("Server ready! Waiting for connections...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
