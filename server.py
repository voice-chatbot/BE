# server.py
import os
import wave
from livekit import api, rtc
import asyncio
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO
import requests
import numpy as np
import io
import google.generativeai as genai
from collections import deque
from pydub import AudioSegment
import json

load_dotenv()

app = Flask(__name__)
# Configure CORS with specific origin
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize ElevenLabs client
elevenlabs = ElevenLabs(
    api_key=os.getenv('ELEVENLABS_API_KEY'),
)

transcription_buffer = deque(maxlen=5)

async def text_to_speech(text):
    audio_generator = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # Convert generator to bytes
    audio_bytes = b''.join(chunk for chunk in audio_generator)
    print(f"Audio bytes length: {len(audio_bytes)}")
    return audio_bytes
    
async def process_with_llm(text):
    print(f"Processing with LLM: {text}")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content(text)
    print(f"LLM Response: {response.text}")
    return response.text

async def process_audio_stream(track: rtc.Track, room: rtc.Room):
    # Initialize audio stream from LiveKit track
    audio_stream = rtc.AudioStream(track)
    buffer = []  # Buffer to store audio chunks
    MIN_AUDIO_LENGTH = 3.0  # Minimum audio length in seconds
    SAMPLE_RATE = 48000  # LiveKit audio sample rate
    is_processing = False  # Flag to control processing state
    
    try:
        # Step 1: LiveKit sends audio frames continuously
        async for event in audio_stream:
            try:
                # Skip processing if we're already processing a response
                if is_processing:
                    continue
                    
                # Get raw audio frame from LiveKit
                audio_frame = event.frame
                
                # Step 2: Convert raw PCM data to numpy array and add to buffer
                # LiveKit sends audio as 16-bit PCM data
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
                buffer.append(audio_data)
                
                # Calculate total audio length in seconds
                total_samples = sum(len(chunk) for chunk in buffer)
                audio_length = total_samples / SAMPLE_RATE
                
                # Step 3: Process when we have enough audio (3 seconds)
                if audio_length >= MIN_AUDIO_LENGTH:
                    is_processing = True  # Set processing flag
                    try:
                        # Combine all audio chunks into one array
                        combined_audio = np.concatenate(buffer)
                        
                        # Step 4: Create WAV file in memory with proper format
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono audio (1 channel)
                            wav_file.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                            wav_file.setframerate(SAMPLE_RATE)  # 48kHz sample rate
                            wav_file.writeframes(combined_audio.tobytes())
                        
                        # Reset buffer position to start
                        wav_buffer.seek(0)
                        
                        # Step 5: Send to ElevenLabs for transcription
                        transcription = elevenlabs.speech_to_text.convert(
                            file=wav_buffer,
                            model_id="scribe_v1",
                            tag_audio_events=True,
                            language_code="eng",
                            diarize=True,
                        )
                        
                        # Extract text from transcription
                        if hasattr(transcription, 'text'):
                            text = transcription.text
                            print(f"Transcription: {text}")
                            
                            # Add to transcription buffer
                            transcription_buffer.append(text)
                            
                            # Process with LLM
                            llm_response = await process_with_llm(text)
                            if llm_response:
                                print(f"LLM Response: {llm_response}")
                                
                                # Convert LLM response to speech
                                audio_response = await text_to_speech(llm_response)
                                if audio_response:
                                    print("Successfully generated speech response")
                                    
                                    # Convert MP3 to PCM format for LiveKit
                                    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_response))
                                    
                                    # Convert to PCM format
                                    pcm_data = audio_segment.raw_data
                                    
                                    # Create audio source with proper parameters
                                    SAMPLE_RATE = 48000  # LiveKit standard
                                    NUM_CHANNELS = 1  # Mono audio
                                    SAMPLES_PER_CHANNEL = 480  # 10ms at 48kHz
                                    
                                    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
                                    audio_track = rtc.LocalAudioTrack.create_audio_track(
                                        name="bot-response",
                                        source=source
                                    )
                                    
                                    # Publish the track to the room using the passed room instance
                                    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
                                    await room.local_participant.publish_track(audio_track, options)
                                    
                                    # Convert audio data to numpy array
                                    audio_data = np.frombuffer(pcm_data, dtype=np.int16)
                                    
                                    # Process audio in chunks
                                    chunk_size = SAMPLES_PER_CHANNEL
                                    for i in range(0, len(audio_data), chunk_size):
                                        chunk = audio_data[i:i + chunk_size]
                                        if len(chunk) < chunk_size:
                                            # Pad the last chunk if needed
                                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                                        
                                        # Create audio frame
                                        frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, chunk_size)
                                        frame_data = np.frombuffer(frame.data, dtype=np.int16)
                                        np.copyto(frame_data, chunk)
                                        
                                        # Send frame to the track
                                        await source.capture_frame(frame)
                                    
                                    # Unpublish the track after sending
                                    await room.local_participant.unpublish_track(audio_track)
                    except Exception as e:
                        error_message = f"Error processing audio: {str(e)}"
                        print(f"Error processing audio: {error_message}")
                    finally:
                        # Reset processing flag and buffer after completion
                        is_processing = False
                        buffer = []
                        
            except Exception as e:
                print(f"Error processing audio frame: {str(e)}")
                is_processing = False  # Reset processing flag on error
                buffer = []  # Clear buffer on error
    finally:
        await audio_stream.aclose()

def generate_backend_token():
    token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
        .with_identity("backend-bot") \
        .with_name("backend-bot") \
        .with_grants(api.VideoGrants(
            room_join=True,
            can_publish=True, 
            room="my-room",
            can_subscribe=True,
        ))
    return token.to_jwt()

@app.route('/getToken')
def getToken():
    try:
        token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
            .with_identity("user") \
            .with_name("user-bot") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room="my-room",
                can_subscribe=True,
                can_publish=True,
            ))
        return {"token": token.to_jwt()}
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/health')
def health_check():
    return {"status": "healthy"}, 200

@app.errorhandler(404)
def not_found(error):
    return {"error": "Resource not found"}, 404

@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500

async def handle_audio_stream():
    room = rtc.Room()
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        print(f"Publication: {publication}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_audio_stream(track, room))

    try:
        await room.connect("wss://chat-e7jp6qc0.livekit.cloud", generate_backend_token())
        await asyncio.Event().wait()
    except Exception as e:
        print(f"Error in audio stream: {e}")

async def main():
    await handle_audio_stream()

if __name__ == '__main__':
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start the audio stream in a separate task
    loop.create_task(main())
    
    # Run the Flask app in the same event loop
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    
    loop.run_until_complete(serve(app, config))