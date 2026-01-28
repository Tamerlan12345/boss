import { SimliClient } from 'https://esm.sh/simli-client@latest';

let audioContext;
let ws;
let isConnected = false;
let avatar = null;

const connectBtn = document.getElementById('connectBtn');
connectBtn.disabled = false; // Enabled by default now, as we don't check avatars
const disconnectBtn = document.getElementById('disconnectBtn');
const statusDiv = document.getElementById('status');
const logsDiv = document.getElementById('logs');
const videoElement = document.getElementById('simli-video');
const audioElement = document.getElementById('simli-audio');

function log(message) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    logsDiv.prepend(div);
}

class SimliAvatar {
    constructor(videoEl, audioEl) {
        this.videoElement = videoEl;
        this.audioElement = audioEl;
        this.simliClient = new SimliClient();
        this.config = null;
    }

    async initialize() {
        try {
            log("Fetching Simli config...");
            const resp = await fetch('/simli/config');
            if (!resp.ok) throw new Error("Failed to load config");
            this.config = await resp.json();

            if (!this.config.apiKey || !this.config.faceID) {
                throw new Error("Missing Simli API Key or Face ID");
            }

            const simliConfig = {
                apiKey: this.config.apiKey,
                faceID: this.config.faceID,
                handleSilence: true,
                videoRef: this.videoElement,
                audioRef: this.audioElement,
                enableConsoleLogs: true,
            };

            log("Initializing Simli Client...");
            this.simliClient.Initialize(simliConfig);

            this.simliClient.on("connected", () => {
                log("✅ Simli WebRTC Connected");
                statusDiv.textContent = "Simli Connected";
            });

            this.simliClient.on("failed", () => {
                log("❌ Simli Connection Failed");
                statusDiv.textContent = "Connection Failed";
            });

            this.simliClient.on("disconnected", () => {
                log("⚠️ Simli Disconnected");
                statusDiv.textContent = "Disconnected";
            });

        } catch (e) {
            log(`Simli Init Error: ${e.message}`);
            throw e;
        }
    }

    start() {
        log("Starting Simli Session...");
        this.simliClient.start();
    }

    speak(audioData) {
        if (this.simliClient) {
            // audioData is Uint8Array (PCM16 16kHz)
            this.simliClient.sendAudioData(audioData);
        }
    }

    close() {
        if (this.simliClient) {
            this.simliClient.close();
        }
    }
}

connectBtn.onclick = async () => {
    try {
        statusDiv.textContent = 'Connecting...';
        connectBtn.disabled = true;

        // 1. Start Simli Avatar
        avatar = new SimliAvatar(videoElement, audioElement);
        await avatar.initialize();
        avatar.start();

        // 2. Initialize AudioContext for Microphone Input
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        log(`AudioContext created. Sample Rate: ${audioContext.sampleRate}`);

        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(audioContext, 'pcm-processor');

        source.connect(processor);
        processor.connect(audioContext.destination);

        // 3. Connect to Backend WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);
        ws.binaryType = 'arraybuffer'; // Expect binary audio back

        ws.onopen = () => {
            isConnected = true;
            statusDiv.textContent = 'System Connected';
            disconnectBtn.disabled = false;
            log('WebSocket connected');
        };

        ws.onclose = () => {
            isConnected = false;
            statusDiv.textContent = 'Disconnected';
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            log('WebSocket closed');

            processor.disconnect();
            source.disconnect();
            if (audioContext) audioContext.close();
            if (avatar) avatar.close();
        };

        ws.onerror = (e) => {
            log(`WebSocket error: ${e}`);
        };

        ws.onmessage = async (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Received Audio from TTS
                // log(`Received Audio Chunk: ${event.data.byteLength} bytes`);
                const uint8 = new Uint8Array(event.data);
                if (avatar) {
                    avatar.speak(uint8);
                }
            } else {
                log(`Text: ${event.data}`);
            }
        };

        // Send Microphone Audio to Backend
        processor.port.onmessage = (event) => {
            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };

    } catch (e) {
        log(`Error: ${e.message}`);
        statusDiv.textContent = 'Error';
        connectBtn.disabled = false;
        console.error(e);
    }
};

disconnectBtn.onclick = () => {
    if (ws) ws.close();
    if (avatar) avatar.close();
};
