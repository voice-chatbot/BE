# Chatbot Backend

This is the backend service for the chatbot application, built with Flask and LiveKit.

## Prerequisites

- Python 3.8 or higher
- LiveKit account and API credentials
- ElevenLabs API key
- Google AI API key

## Environment Variables

Create a `.env` file with the following variables:

```env
# LiveKit Configuration
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Frontend URL for CORS
FRONTEND_URL=your_frontend_url
```

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python server.py
```

## API Endpoints

- `GET /getToken`: Get LiveKit token for client connection
