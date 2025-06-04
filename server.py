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
# import google.generativeai as genai
from collections import deque

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize ElevenLabs client
elevenlabs = ElevenLabs(
    api_key="sk_53fe88903983940a8899c41d896eeaaf9fec626b941fcdd1",  # Replace this with your actual API key
)
# audio_url = (
#     "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
# )
# response = requests.get(audio_url)
# audio_data = BytesIO(response.content)


transcription_buffer = deque(maxlen=5)

async def process_with_llm(text):
    print(f"Processing with LLM: {text}")

async def process_audio_stream(track: rtc.Track):
    # Initialize audio stream from LiveKit track
    audio_stream = rtc.AudioStream(track)
    buffer = []  # Buffer to store audio chunks
    MIN_AUDIO_LENGTH = 3.0  # Minimum audio length in seconds
    SAMPLE_RATE = 48000  # LiveKit audio sample rate
    
    try:
        # Step 1: LiveKit sends audio frames continuously
        async for event in audio_stream:
            try:
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
                    
                    try:
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
                        
                        # Step 6: Clear the buffer after successful transcription
                        # This allows us to start collecting new audio
                        buffer = []
                    except Exception as e:
                        print(f"Error in ElevenLabs API call: {str(e)}")
                        # Keep the buffer in case of error
            except Exception as e:
                print(f"Error processing audio frame: {str(e)}")
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

async def handle_audio_stream():
    room = rtc.Room()
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        print(f"Publication: {publication}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_audio_stream(track))

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