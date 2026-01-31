#!/usr/bin/env python3
import asyncio
import json
import numpy as np
import whisper
import websockets

# Load the Whisper model once at startup
print("Loading Whisper model...")
model = whisper.load_model("tiny.en")
print("Model loaded successfully!")

# Audio parameters
SAMPLE_RATE = 16000
CHUNK_BUFFER_SIZE = 50  # Number of chunks to accumulate before transcribing

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
                    print(f"Transcribing {len(audio_buffer)} chunks...")
                    
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Run transcription in thread to avoid blocking event loop
                    result = await asyncio.to_thread(
                        model.transcribe, 
                        audio_data.flatten(), 
                        language="en"
                    )
                    
                    transcription = result['text'].strip()
                    if transcription:
                        print(f"Transcription: {transcription}")
                        # Send transcription result back to client
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": transcription
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
    print("Starting WebSocket server on localhost:8765...")
    async with websockets.serve(handle_client, "localhost", 8765):
        print("Server ready! Waiting for connections...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
