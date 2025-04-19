#!/usr/bin/env python3
"""
WebSocket server for real-time streaming speech recognition.
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR, StreamingTranscriptionResult, PARAMETER_PRESETS
from speech_to_text.utils.audio_utils import audio_bytes_to_array

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Streaming Speech Recognition API",
    description="Real-time streaming speech recognition using Whisper.cpp",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
settings = {
    "model_path": "",
    "language": "en",
    "n_threads": 4,
    "sample_rate": 16000,
    "chunk_size_ms": 1000,
    # Add new parameters with default values
    "temperature": 0.0,
    "initial_prompt": None,
    "max_tokens": 0,
    "no_context": False,
    "single_segment": True,
    "preset": None
}

# Active clients
active_clients: Dict[str, StreamingWhisperASR] = {}

class TranscriptionRequest(BaseModel):
    """Request for transcription configuration."""
    language: str = Field(default="en", description="Language code")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    # Add new fields for advanced parameters
    temperature: Optional[float] = Field(default=None, description="Controls creativity in transcription (higher = more creative)")
    initial_prompt: Optional[str] = Field(default=None, description="Provides context to guide the transcription")
    max_tokens: Optional[int] = Field(default=None, description="Limits the number of tokens per segment")
    no_context: Optional[bool] = Field(default=None, description="Controls whether to use previous transcription as context")
    single_segment: Optional[bool] = Field(default=None, description="Enabled for better streaming performance")
    preset: Optional[str] = Field(default=None, description="Parameter preset to use")

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("Starting streaming speech recognition server")
    
    # Check if model path is set
    if not settings["model_path"] or not os.path.isfile(settings["model_path"]):
        logger.error(f"Model file not found: {settings['model_path']}")
        sys.exit(1)
    
    logger.info(f"Using model: {settings['model_path']}")
    logger.info(f"Default language: {settings['language']}")
    logger.info(f"Using {settings['n_threads']} CPU threads")
    
    # Log available parameter presets
    logger.info(f"Available parameter presets: {list(PARAMETER_PRESETS.keys())}")

@app.websocket("/ws/transcribe")
async def transcribe_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription.
    
    Audio format:
    - 16-bit PCM
    - Sample rate: 16kHz (configurable)
    - Mono
    """
    await websocket.accept()
    
    # Generate client ID
    client_id = f"client_{int(time.time() * 1000)}"
    
    try:
        # Receive configuration
        config_data = await websocket.receive_text()
        config = TranscriptionRequest.parse_raw(config_data)
        
        logger.info(f"Client {client_id} connected with config: {config}")
        
        # Build parameters dictionary with only provided values
        params = {}
        if config.temperature is not None:
            params['temperature'] = config.temperature
        if config.initial_prompt is not None:
            params['initial_prompt'] = config.initial_prompt
        if config.max_tokens is not None:
            params['max_tokens'] = config.max_tokens
        if config.no_context is not None:
            params['no_context'] = config.no_context
        if config.single_segment is not None:
            params['single_segment'] = config.single_segment
        if config.preset is not None:
            params['preset'] = config.preset
            
        # Create ASR instance for this client
        asr = StreamingWhisperASR(
            model_path=settings["model_path"],
            language=config.language,
            n_threads=settings["n_threads"],
            sample_rate=config.sample_rate,
            chunk_size_ms=settings["chunk_size_ms"],
            **params  # Include any parameters that were provided
        )
        
        # Store in active clients
        active_clients[client_id] = asr
        
        # Send ready message
        await websocket.send_json({
            "status": "ready",
            "message": "Ready to receive audio"
        })
        
        # Define result callback
        async def send_result(result: StreamingTranscriptionResult):
            await websocket.send_json({
                "type": "transcription",
                "text": result.text,
                "is_final": result.is_final,
                "confidence": result.confidence,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "chunk_id": result.chunk_id
            })
        
        # Process incoming audio
        while True:
            # Receive audio data
            try:
                data = await websocket.receive_bytes()
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break
            
            # Check for end of stream
            if not data or len(data) == 0:
                logger.info(f"Client {client_id} sent empty data, ending stream")
                break
            
            # Convert audio bytes to numpy array
            audio, sr = audio_bytes_to_array(
                data,
                sample_width=2,  # 16-bit PCM
                channels=1,       # Mono
                sample_rate=config.sample_rate,
                normalize=True
            )
            
            # Process audio chunk
            try:
                await asr.process_audio_chunk(audio, callback=send_result)
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                continue
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Clean up
        if client_id in active_clients:
            try:
                # Get final transcript
                final_text, duration = await active_clients[client_id].stop_streaming()
                # Send final result if available
                if final_text:
                    try:
                        await websocket.send_json({
                            "type": "final",
                            "text": final_text,
                            "duration": duration
                        })
                    except:
                        pass
            except:
                pass
            
            # Remove client
            del active_clients[client_id]
        
        try:
            await websocket.close()
        except:
            pass

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Streaming Speech Recognition API",
        "version": "0.1.0",
        "endpoints": {
            "websocket": "/ws/transcribe"
        },
        "available_presets": list(PARAMETER_PRESETS.keys())
    }

# Add simple HTML test page
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)

# Create simple HTML test page with additional parameters
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Recognition Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #status { margin: 10px 0; padding: 10px; background-color: #f0f0f0; }
        #transcript { margin: 10px 0; padding: 10px; min-height: 200px; border: 1px solid #ccc; }
        button { padding: 10px; margin: 5px; }
        .settings { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
        .setting-group { margin-bottom: 10px; }
        label { display: inline-block; width: 150px; }
    </style>
</head>
<body>
    <h1>Real-time Speech Recognition Test</h1>
    
    <div class="settings">
        <h3>Settings</h3>
        <div class="setting-group">
            <label for="language">Language:</label>
            <select id="language">
                <option value="en" selected>English</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="es">Spanish</option>
                <option value="it">Italian</option>
                <option value="ja">Japanese</option>
                <option value="zh">Chinese</option>
                <option value="auto">Auto Detect</option>
            </select>
        </div>
        
        <div class="setting-group">
            <label for="preset">Parameter Preset:</label>
            <select id="preset">
                <option value="" selected>None (Custom)</option>
                <option value="default">Default</option>
                <option value="creative">Creative</option>
                <option value="structured">Structured</option>
                <option value="technical">Technical</option>
                <option value="meeting">Meeting</option>
            </select>
        </div>
        
        <div class="setting-group">
            <label for="temperature">Temperature:</label>
            <input type="range" id="temperature" min="0" max="1" step="0.1" value="0">
            <span id="temperature-value">0.0</span>
        </div>
        
        <div class="setting-group">
            <label for="initial-prompt">Initial Prompt:</label>
            <input type="text" id="initial-prompt" style="width: 300px">
        </div>
        
        <div class="setting-group">
            <label for="max-tokens">Max Tokens:</label>
            <input type="number" id="max-tokens" min="0" max="500" value="0">
            <span>(0 = no limit)</span>
        </div>
        
        <div class="setting-group">
            <label for="no-context">No Context:</label>
            <input type="checkbox" id="no-context">
        </div>
        
        <div class="setting-group">
            <label for="single-segment">Single Segment:</label>
            <input type="checkbox" id="single-segment" checked>
        </div>
    </div>
    
    <div>
        <button id="startBtn">Start Recording</button>
        <button id="stopBtn" disabled>Stop Recording</button>
    </div>
    <div id="status">Status: Idle</div>
    <div>
        <h3>Transcript:</h3>
        <div id="transcript"></div>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const transcriptDiv = document.getElementById('transcript');
        const temperatureSlider = document.getElementById('temperature');
        const temperatureValue = document.getElementById('temperature-value');
        const presetSelect = document.getElementById('preset');
        
        // Update temperature value display
        temperatureSlider.addEventListener('input', () => {
            temperatureValue.textContent = temperatureSlider.value;
        });
        
        // Handle preset selection
        presetSelect.addEventListener('change', () => {
            if (presetSelect.value === '') {
                // Custom settings - do nothing
                return;
            }
            
            // Preset values will be applied on the server
            // Just disable individual settings when a preset is selected
            const isCustom = presetSelect.value === '';
            document.getElementById('temperature').disabled = !isCustom;
            document.getElementById('initial-prompt').disabled = !isCustom;
            document.getElementById('max-tokens').disabled = !isCustom;
            document.getElementById('no-context').disabled = !isCustom;
            document.getElementById('single-segment').disabled = !isCustom;
        });
        
        let socket;
        let mediaRecorder;
        let audioContext;
        let stream;
        
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        
        async function startRecording() {
            try {
                statusDiv.textContent = 'Status: Connecting...';
                
                // Connect to WebSocket
                const serverUrl = window.location.hostname;
                const serverPort = window.location.port;
                const wsUrl = `ws://${serverUrl}:${serverPort}/ws/transcribe`;
                
                socket = new WebSocket(wsUrl);
                
                socket.onopen = async () => {
                    // Get configuration from UI
                    const config = {
                        language: document.getElementById('language').value,
                        sample_rate: 16000
                    };
                    
                    // Add advanced parameters if selected
                    if (presetSelect.value) {
                        config.preset = presetSelect.value;
                    } else {
                        // Only add these if no preset is selected
                        const temperature = parseFloat(temperatureSlider.value);
                        config.temperature = temperature;
                        
                        const initialPrompt = document.getElementById('initial-prompt').value;
                        if (initialPrompt) {
                            config.initial_prompt = initialPrompt;
                        }
                        
                        const maxTokens = parseInt(document.getElementById('max-tokens').value);
                        config.max_tokens = maxTokens;
                        
                        config.no_context = document.getElementById('no-context').checked;
                        config.single_segment = document.getElementById('single-segment').checked;
                    }
                    
                    // Send configuration
                    socket.send(JSON.stringify(config));
                    
                    // Wait for ready message
                    socket.onmessage = async (event) => {
                        const data = JSON.parse(event.data);
                        
                        if (data.status === 'ready') {
                            // Start audio recording
                            statusDiv.textContent = 'Status: Recording...';
                            transcriptDiv.textContent = '';
                            
                            // Get user media
                            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                            
                            // Create audio context
                            audioContext = new AudioContext({
                                sampleRate: 16000
                            });
                            
                            // Create source node
                            const source = audioContext.createMediaStreamSource(stream);
                            
                            // Create processor node
                            const processor = audioContext.createScriptProcessor(4096, 1, 1);
                            
                            // Connect nodes
                            source.connect(processor);
                            processor.connect(audioContext.destination);
                            
                            // Process audio data
                            processor.onaudioprocess = (e) => {
                                const inputData = e.inputBuffer.getChannelData(0);
                                
                                // Convert to 16-bit PCM
                                const pcmData = new Int16Array(inputData.length);
                                for (let i = 0; i < inputData.length; i++) {
                                    pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                                }
                                
                                // Send to WebSocket if connected
                                if (socket.readyState === WebSocket.OPEN) {
                                    socket.send(pcmData.buffer);
                                }
                            };
                            
                            // Update button state
                            startBtn.disabled = true;
                            stopBtn.disabled = false;
                            
                            // Update message handler
                            socket.onmessage = handleMessage;
                        }
                    };
                };
                
                socket.onerror = (error) => {
                    statusDiv.textContent = `Status: Error - ${error}`;
                    console.error('WebSocket error:', error);
                };
                
                socket.onclose = () => {
                    statusDiv.textContent = 'Status: Disconnected';
                    stopRecording();
                };
            } catch (error) {
                statusDiv.textContent = `Status: Error - ${error.message}`;
                console.error('Error starting recording:', error);
            }
        }
        
        function stopRecording() {
            // Stop media tracks
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            // Close audio context
            if (audioContext) {
                audioContext.close();
            }
            
            // Close WebSocket
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.close();
            }
            
            // Update button state
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            statusDiv.textContent = 'Status: Idle';
        }
        
        function handleMessage(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcription') {
                // Update transcript
                if (data.text) {
                    const p = document.createElement('p');
                    p.textContent = `[${data.start_time.toFixed(1)}s-${data.end_time.toFixed(1)}s] ${data.text}`;
                    transcriptDiv.appendChild(p);
                    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                }
            } else if (data.type === 'final') {
                statusDiv.textContent = `Status: Completed (${data.duration.toFixed(1)}s)`;
            } else if (data.type === 'error') {
                statusDiv.textContent = `Status: Error - ${data.message}`;
                console.error('Transcription error:', data.message);
            }
        }
    </script>
</body>
</html>
"""

# Write HTML file
with open(os.path.join(static_dir, "index.html"), "w") as f:
    f.write(html_content)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add control endpoint to change parameters at runtime
@app.post("/update_params/{client_id}")
async def update_client_params(
    client_id: str,
    temperature: Optional[float] = None,
    initial_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    no_context: Optional[bool] = None,
    single_segment: Optional[bool] = None,
    preset: Optional[str] = None
):
    """Update parameters for an active client."""
    if client_id not in active_clients:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    try:
        asr = active_clients[client_id]
        
        if preset:
            asr.set_parameter_preset(preset)
        else:
            asr.update_parameters(
                temperature=temperature,
                initial_prompt=initial_prompt,
                max_tokens=max_tokens,
                no_context=no_context,
                single_segment=single_segment
            )
        
        return {"status": "success", "message": "Parameters updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameters: {str(e)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Streaming Speech Recognition Server')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to Whisper model file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port')
    parser.add_argument('--language', type=str, default='en',
                        help='Default language code')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of CPU threads to use')
    # Add new parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling (higher=more creative)')
    parser.add_argument('--initial-prompt', type=str, default=None,
                        help='Initial prompt to guide transcription')
    parser.add_argument('--max-tokens', type=int, default=0,
                        help='Maximum tokens per segment (0=no limit)')
    parser.add_argument('--no-context', action='store_true',
                        help='Do not use past transcription as context')
    parser.add_argument('--single-segment', action='store_true', default=True,
                        help='Force single segment output (useful for streaming)')
    parser.add_argument('--preset', type=str, choices=list(PARAMETER_PRESETS.keys()),
                        help='Parameter preset to use')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Update global settings
    settings["model_path"] = args.model
    settings["language"] = args.language
    settings["n_threads"] = args.threads
    settings["temperature"] = args.temperature
    settings["initial_prompt"] = args.initial_prompt
    settings["max_tokens"] = args.max_tokens
    settings["no_context"] = args.no_context
    settings["single_segment"] = args.single_segment
    settings["preset"] = args.preset
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()