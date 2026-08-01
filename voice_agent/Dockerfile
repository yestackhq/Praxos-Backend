# The Praxos voice tutor worker.
#
# Deployed as its OWN Railway service with this directory as the build context —
# it shares no code with the API (agent.py imports nothing from lms_app, only
# httpx + livekit), and keeping the contexts separate means the repo-root
# railway.json/Dockerfile, which describe the API, cannot be applied to it.
#
# It publishes no public port and needs no domain: it dials out to LiveKit and
# talks to the API over Railway's private network.
FROM python:3.11-slim

WORKDIR /app

# libsndfile/ffmpeg back the audio codecs the plugins load at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/voice_agent/
RUN touch /app/voice_agent/__init__.py

# Downloads the Silero VAD weights at build time so the first learner does not
# pay for the fetch (and a cold start cannot fail on a network blip).
RUN python -m voice_agent.agent download-files || true

CMD ["python", "-m", "voice_agent.agent", "start"]
