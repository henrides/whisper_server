#!/usr/bin/env python3
import asyncio
import json
import sounddevice as sd
import numpy as np
import websockets

# Audio parameters
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024

# Silence detection parameters
SILENCE_THRESHOLD = 0.05  # RMS amplitude threshold for silence detection
SILENCE_DURATION_THRESHOLD = 10.0  # seconds - send chunk after 60s of silence
MIN_CHUNK_DURATION = 5.0  # seconds - minimum audio length to send
MAX_CHUNK_DURATION = 240.0  # seconds - maximum chunk size (4 minutes)

# WebSocket server URL (via SSH tunnel)
SERVER_URL = "ws://localhost:8765"

# Global queue for audio chunks and event loop reference
audio_queue = asyncio.Queue()
loop = None

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio data."""
    if status:
        print(status)
    # Put audio data into asyncio queue in a thread-safe way
    if loop is not None:
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

def calculate_rms(audio_data):
    """Calculate RMS (Root Mean Square) of audio data."""
    return np.sqrt(np.mean(audio_data ** 2))

def is_silence(audio_data, threshold=SILENCE_THRESHOLD):
    """Check if audio chunk is silence."""
    rms = calculate_rms(audio_data)
    return rms < threshold

async def send_audio(websocket):
    """Send audio chunks to the server, buffering and detecting silence."""
    audio_buffer = []
    silence_duration = 0.0
    
    while True:
        audio_chunk = await audio_queue.get()
        chunk_duration = len(audio_chunk) / SAMPLE_RATE
        
        # Check if this chunk is silence
        is_silent = is_silence(audio_chunk)
        
        if is_silent:
            silence_duration += chunk_duration
        else:
            # Reset silence counter when sound is detected
            silence_duration = 0.0
        
        # Always add non-silent chunks to buffer
        # Also add initial silent chunks (to capture quiet speech)
        if not is_silent or len(audio_buffer) == 0:
            audio_buffer.append(audio_chunk)
        
        # Calculate current buffer duration
        buffer_duration = sum(len(chunk) for chunk in audio_buffer) / SAMPLE_RATE
        
        # Send chunk if:
        # 1. We have enough content AND detected sufficient silence
        # 2. OR buffer exceeds max duration (prevent message size issues)
        should_send = False
        reason = ""
        
        if silence_duration >= SILENCE_DURATION_THRESHOLD and buffer_duration >= MIN_CHUNK_DURATION:
            should_send = True
            reason = f"silence detected ({silence_duration:.1f}s)"
        elif buffer_duration >= MAX_CHUNK_DURATION:
            should_send = True
            reason = f"max duration reached ({buffer_duration:.1f}s)"
        
        if should_send and audio_buffer:
            print(f"\n🔇 Trigger: {reason}, sending {buffer_duration:.1f}s of audio for transcription...")
            # Concatenate buffer and send as one chunk
            audio_data = np.concatenate(audio_buffer)
            await websocket.send(audio_data.tobytes())
            
            # Clear buffer and reset
            audio_buffer.clear()
            silence_duration = 0.0

async def receive_transcriptions(websocket):
    """Receive transcription results from the server."""
    async for message in websocket:
        data = json.loads(message)
        if data.get("type") == "transcription":
            # Handle both old format (text) and new format (segments)
            if "segments" in data:
                # New format with speaker diarization
                segments = data["segments"]
                duration = data.get("duration", 0)
                print(f"\n{'='*60}")
                print(f"=== Transcription ({duration:.1f}s of audio) ===")
                print(f"{'='*60}")
                for seg in segments:
                    speaker = seg.get("speaker", "UNKNOWN")
                    text = seg.get("text", "")
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    print(f"[{speaker}] ({start:.1f}s-{end:.1f}s): {text}")
                print(f"{'='*60}\n")
            else:
                # Old format (backward compatibility)
                print(f"Transcription: {data['text']}")
        elif data.get("type") == "ack":
            print(f"Server status: {data['status']}")

async def main():
    """Main client function."""
    global loop
    loop = asyncio.get_running_loop()
    
    print(f"Connecting to server at {SERVER_URL}...")
    
    try:
        # Increase max_size to 16MB and configure heartbeat for long processing
        # ping_interval=20s, ping_timeout=300s (5 minutes) to handle long transcriptions
        async with websockets.connect(
            SERVER_URL, 
            max_size=16 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=300
        ) as websocket:
            print("Connected to server!")
            
            # Send configuration handshake
            config = {
                "type": "config",
                "sample_rate": SAMPLE_RATE,
                "dtype": "float32",
                "channels": 1
            }
            await websocket.send(json.dumps(config))
            print("Configuration sent")
            
            # Start audio stream
            with sd.InputStream(
                callback=audio_callback, 
                channels=1, 
                samplerate=SAMPLE_RATE, 
                blocksize=BUFFER_SIZE
            ):
                print("Listening... Press Ctrl+C to stop.")
                print("\nNote: Audio is buffered and transcribed in chunks.")
                print("Transcription triggers after 60 seconds of silence.")
                print("This provides more accurate results than real-time transcription.\n")
                
                # Run send and receive tasks concurrently
                await asyncio.gather(
                    send_audio(websocket),
                    receive_transcriptions(websocket)
                )
    
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
        print("\nMake sure:")
        print("1. The server is running on the remote machine")
        print("2. SSH tunnel is active: ssh -L 8765:localhost:8765 user@server-hostname")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
