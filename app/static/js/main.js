let audioContext;
let ws;
let isConnected = false;
let avatar = null;

const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const statusDiv = document.getElementById('status');
const logsDiv = document.getElementById('logs');
const videoElement = document.getElementById('heygen-video');

function log(message) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    logsDiv.prepend(div);
}

// HeyGen Avatar Logic
const AVATAR_ID = 'da4a68297f26487a95078864c39c55b5'; // Male Avatar (Tyler)
const VOICE_ID = '132a2651478f44b2a8bb7492c34cb623'; // Male Voice

class HeyGenAvatar {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.peerConnection = null;
        this.sessionId = null;
        this.token = null;
    }

    async getToken() {
        const response = await fetch('/token', { method: 'POST' });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        return data.data.token;
    }

    async createSession(token) {
        const response = await fetch('https://api.heygen.com/v1/streaming.new', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                quality: 'medium',
                avatar_name: AVATAR_ID,
                voice: { voice_id: VOICE_ID }
            })
        });
        const data = await response.json();
        if (!data.data) throw new Error('Failed to create session');
        return data.data;
    }

    async startStreaming() {
        try {
            log('Starting HeyGen Avatar...');
            this.token = await this.getToken();
            const sessionData = await this.createSession(this.token);
            this.sessionId = sessionData.session_id;
            const { sdp: serverSdp, ice_servers2: iceServers } = sessionData;

            this.peerConnection = new RTCPeerConnection({ iceServers: iceServers });

            this.peerConnection.ontrack = (event) => {
                if (event.track.kind === 'video' && this.videoElement.srcObject !== event.streams[0]) {
                    this.videoElement.srcObject = event.streams[0];
                    log('Video stream received');
                }
            };

            this.peerConnection.onicecandidate = async (event) => {
                if (event.candidate) {
                    await fetch('https://api.heygen.com/v1/streaming.ice', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${this.token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            session_id: this.sessionId,
                            candidate: event.candidate
                        })
                    });
                }
            };

            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(serverSdp));
            const localSdp = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(localSdp);

            await fetch('https://api.heygen.com/v1/streaming.start', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    sdp: localSdp
                })
            });

            log('HeyGen Session Started');

        } catch (e) {
            log(`HeyGen Error: ${e.message}`);
            throw e;
        }
    }

    async speak(text) {
        if (!this.sessionId) return;
        try {
            await fetch('https://api.heygen.com/v1/streaming.task', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text
                })
            });
            log(`Avatar speaking: ${text}`);
        } catch (e) {
            log(`Speak Error: ${e.message}`);
        }
    }

    async stopSession() {
        if (!this.sessionId || !this.token) return;
        try {
            await fetch('https://api.heygen.com/v1/streaming.stop', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            log('HeyGen Session Stopped');
        } catch (e) {
            log(`Stop Session Error: ${e.message}`);
        }
    }

    close() {
        if (this.peerConnection) this.peerConnection.close();
        this.stopSession();
    }
}

connectBtn.onclick = async () => {
    try {
        statusDiv.textContent = 'Connecting...';

        // Start Avatar
        avatar = new HeyGenAvatar(videoElement);
        await avatar.startStreaming();

        // Initialize AudioContext for Microphone Input
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        log(`AudioContext created. Sample Rate: ${audioContext.sampleRate}`);

        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(audioContext, 'pcm-processor');

        source.connect(processor);
        processor.connect(audioContext.destination); // Required for AudioWorklet to run, but output is silenced in processor

        // Check protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = () => {
            isConnected = true;
            statusDiv.textContent = 'Connected';
            connectBtn.disabled = true;
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
            // Receive text from Gemini
            log(`Gemini: ${event.data}`);
            if (avatar) {
                avatar.speak(event.data);
            }
        };

        processor.port.onmessage = (event) => {
            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };

    } catch (e) {
        log(`Error: ${e.message}`);
        statusDiv.textContent = 'Error';
        console.error(e);
    }
};

disconnectBtn.onclick = () => {
    if (ws) ws.close();
};

