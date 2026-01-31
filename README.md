# Whisper Audio Transcription - Client/Server

WebSocket-based client-server system for real-time audio transcription using OpenAI's Whisper model. The client captures microphone audio and streams it to a remote server for transcription over an encrypted SSH tunnel.

## Architecture

- **Client** (`real_time.py`): Captures microphone audio with `sounddevice` and streams it via WebSocket
- **Server** (`whisper_server.py`): Receives audio streams, runs Whisper transcription, and sends results back
- **Transport**: WebSocket over SSH tunnel for encrypted communication

## Setup

### Server Setup (Powerful Machine)

1. Clone the repository:
```bash
git clone <repository-url>
cd whisper_test
```

2. Create virtual environment and install dependencies:
```bash
python3 -m venv whisper_env
source whisper_env/bin/activate
pip install -r requirements-server.txt
```

3. Run the server:
```bash
python whisper_server.py
```

The server will listen on `localhost:8765` and load the Whisper model on startup.

### Client Setup (Local Machine)

1. Clone the repository:
```bash
git clone <repository-url>
cd whisper_test
```

2. Create virtual environment and install dependencies:
```bash
python3 -m venv whisper_env
source whisper_env/bin/activate
pip install -r requirements-client.txt
```

3. Establish SSH tunnel to the server:
```bash
ssh -L 8765:localhost:8765 user@server-hostname
```

3. In a separate terminal, run the client:
```bash
source whisper_env/bin/activate
python real_time.py
```

The client will connect through the SSH tunnel and start streaming audio.

## Usage

Once both client and server are running:
- Speak into your microphone
- The client streams audio to the server every ~1.5-3 seconds
- Transcription results appear on the client terminal
- Press `Ctrl+C` to stop

## Requirements

- Python 3.8+
- Microphone access on client machine
- Network connectivity between client and server
- SSH access to server for tunnel setup

## Configuration

### Audio Parameters

Both client and server use:
- Sample rate: 16kHz
- Buffer size: 1024 samples
- Channels: 1 (mono)

### Whisper Model

The server uses the `tiny.en` model by default (fastest, English-only). To change:

Edit `whisper_server.py`:
```python
model = whisper.load_model("tiny.en")  # Options: tiny.en, base.en, small.en, medium.en, large
```

### Transcription Buffer

The server transcribes after accumulating 50 audio chunks (~1.5-3 seconds). To adjust:

Edit `whisper_server.py`:
```python
CHUNK_BUFFER_SIZE = 50  # Increase for longer context, decrease for faster response
```

## Troubleshooting

**"WebSocket error" on client:**
- Verify SSH tunnel is active: `ssh -L 8765:localhost:8765 user@server`
- Check server is running: `python whisper_server.py`
- Ensure port 8765 is not blocked by firewall

**No audio captured:**
- Check microphone permissions
- Verify microphone is connected: `python -c "import sounddevice as sd; print(sd.query_devices())"`

**Slow transcription:**
- Use a smaller Whisper model (`tiny.en` or `base.en`)
- Reduce `CHUNK_BUFFER_SIZE` for more frequent transcription
- Ensure server has GPU support (CUDA) for faster processing

## License

This project uses OpenAI's Whisper model. See [openai/whisper](https://github.com/openai/whisper) for model details and licensing.
