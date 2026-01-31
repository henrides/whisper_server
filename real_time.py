#!/usr/bin/env python3
import asyncio
import json
import sounddevice as sd
import numpy as np
import websockets

# Audio parameters
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024

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

async def send_audio(websocket):
    """Send audio chunks to the server."""
    while True:
        audio_chunk = await audio_queue.get()
        # Convert numpy array to bytes and send as binary
        await websocket.send(audio_chunk.tobytes())

async def receive_transcriptions(websocket):
    """Receive transcription results from the server."""
    async for message in websocket:
        data = json.loads(message)
        if data.get("type") == "transcription":
            # Handle both old format (text) and new format (segments)
            if "segments" in data:
                # New format with speaker diarization
                segments = data["segments"]
                print("\n=== Transcription ===")
                for seg in segments:
                    speaker = seg.get("speaker", "UNKNOWN")
                    text = seg.get("text", "")
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    print(f"[{speaker}] ({start:.1f}s-{end:.1f}s): {text}")
                print("====================\n")
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
        async with websockets.connect(SERVER_URL) as websocket:
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
