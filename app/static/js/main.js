import { SimliClient } from 'https://esm.sh/simli-client@latest';

let audioContext;
let ws;
let isConnected = false;
let avatar = null;
let animationId;
let analyser;
let dataArray;

const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const muteBtn = document.getElementById('muteBtn');
const summaryBtn = document.getElementById('summaryBtn');
const introBtn = document.getElementById('introBtn');
const activeToggleBtn = document.getElementById('activeToggleBtn');
const statusDiv = document.getElementById('status');
const logsDiv = document.getElementById('logs');
const videoElement = document.getElementById('simli-video');

// Detect Mode
const path = window.location.pathname;
let mode = 'default';
if (path.includes('/panel')) mode = 'panel';
if (path.includes('/speaker')) mode = 'speaker';

if (mode === 'panel') {
    muteBtn.style.display = 'inline-block';
    let isActive = true;
    muteBtn.onclick = () => {
        isActive = !isActive;
        muteBtn.textContent = isActive ? "DOS: ACTIVE" : "DOS: MUTED";
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "mute_toggle", enabled: isActive }));
        }
    };
}

if (mode === 'speaker') {
    summaryBtn.onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "trigger_summary" }));
            summaryBtn.disabled = true;
            summaryBtn.textContent = "GENERATING...";
        }
    };

    introBtn.onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "trigger_introduce" }));
        }
    };

    let isDosActive = false;
    activeToggleBtn.onclick = () => {
        isDosActive = !isDosActive;
        activeToggleBtn.textContent = isDosActive ? "Режим: АКТИВНЫЙ" : "Режим: ПАССИВНЫЙ";
        activeToggleBtn.style.backgroundColor = isDosActive ? "var(--acid-green)" : "";
        activeToggleBtn.style.color = isDosActive ? "#000" : "";

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "toggle_active", enabled: isDosActive }));
        }
    };
}
const audioElement = document.getElementById('simli-audio');
const avatarFrame = document.getElementById('avatarFrame');
const canvas = document.getElementById('audio-visualizer');
const canvasCtx = canvas.getContext('2d');

function appendToLog(role, text) {
    const div = document.createElement('div');
    div.className = `log-entry role-${role.toLowerCase()}`;

    const timestamp = new Date().toLocaleTimeString();
    const roleLabel = role.toUpperCase();

    const timestampSpan = document.createElement('span');
    timestampSpan.className = 'timestamp';
    timestampSpan.textContent = `[${timestamp}] `;

    const roleSpan = document.createElement('span');
    roleSpan.className = 'role-label';
    roleSpan.textContent = `${roleLabel}> `;

    const textNode = document.createTextNode(text);

    div.appendChild(timestampSpan);
    div.appendChild(roleSpan);
    div.appendChild(textNode);

    logsDiv.appendChild(div);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

// Visualizer
function drawVisualizer() {
    if (!isConnected) return;

    animationId = requestAnimationFrame(drawVisualizer);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = '#111';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    for(let i = 0; i < dataArray.length; i++) {
        barHeight = dataArray[i] / 4; // Scale down
        canvasCtx.fillStyle = `rgb(${barHeight + 100}, 50, 255)`; // Purpleish
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
    }
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
            appendToLog('SYSTEM', "Fetching Simli config...");
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
                // !!! КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ НИЖЕ !!!
                // Без этой строчки P2P соединение падает с ошибкой Ice Servers Required
                iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
            };

            appendToLog('SYSTEM', "Initializing Simli Client...");
            this.simliClient.Initialize(simliConfig);

            this.simliClient.on("connected", () => {
                appendToLog('SYSTEM', "✅ Simli WebRTC Connected");
                statusDiv.textContent = "SYSTEM: ONLINE";
                statusDiv.style.color = "var(--acid-green)";
            });

            this.simliClient.on("failed", () => {
                appendToLog('SYSTEM', "❌ Simli Connection Failed");
                statusDiv.textContent = "SYSTEM: ERROR";
                statusDiv.style.color = "red";
            });

            this.simliClient.on("disconnected", () => {
                appendToLog('SYSTEM', "⚠️ Simli Disconnected");
                statusDiv.textContent = "SYSTEM: DISCONNECTED";
                statusDiv.style.color = "var(--text-color)";
            });

        } catch (e) {
            appendToLog('SYSTEM', `Simli Init Error: ${e.message}`);
            throw e;
        }
    }

    start() {
        appendToLog('SYSTEM', "Starting Simli Session...");
        
        // Configuration is now handled in Initialize
        this.simliClient.start();
    }

    speak(audioData) {
        if (this.simliClient) {
            this.simliClient.sendAudioData(audioData);
            // Flash frame
            avatarFrame.classList.add('speaking');
            setTimeout(() => avatarFrame.classList.remove('speaking'), 200);
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
        statusDiv.textContent = 'INITIATING HANDSHAKE...';
        connectBtn.disabled = true;

        // 1. Start Simli Avatar
        avatar = new SimliAvatar(videoElement, audioElement);
        await avatar.initialize();
        avatar.start();

        // 2. Initialize AudioContext for Microphone Input and Visualizer
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);

        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(audioContext, 'pcm-processor');

        source.connect(analyser); // Visualizer
        source.connect(processor);
        processor.connect(audioContext.destination);

        drawVisualizer();

        // 3. Connect to Backend WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws?mode=${mode}`);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            isConnected = true;
            if (mode === 'speaker') {
                summaryBtn.style.display = 'inline-block';
                introBtn.style.display = 'inline-block';
                activeToggleBtn.style.display = 'inline-block';

                // Initialize in Passive Mode
                ws.send(JSON.stringify({ type: "toggle_active", enabled: false }));
            }
            statusDiv.textContent = 'SYSTEM: CONNECTED';
            statusDiv.style.color = "var(--acid-green)";
            disconnectBtn.disabled = false;
            appendToLog('SYSTEM', 'Neural Uplink Established.');
        };

        ws.onclose = () => {
            isConnected = false;
            statusDiv.textContent = 'SYSTEM: OFFLINE';
            statusDiv.style.color = "var(--text-color)";
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            appendToLog('SYSTEM', 'Neural Uplink Terminated.');

            processor.disconnect();
            source.disconnect();
            if (audioContext) audioContext.close();
            if (avatar) avatar.close();
            cancelAnimationFrame(animationId);
        };

        ws.onerror = (e) => {
            appendToLog('SYSTEM', `WebSocket error: ${e}`);
        };

        ws.onmessage = async (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Audio
                const uint8 = new Uint8Array(event.data);
                if (avatar) {
                    avatar.speak(uint8);
                }
            } else {
                // Text / JSON
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'log') {
                        appendToLog(msg.role, msg.text);
                    } else if (msg.type === 'summary_done') {
                        summaryBtn.disabled = false;
                        summaryBtn.textContent = "GENERATE SUMMARY";
                        appendToLog('SYSTEM', "Summary generation complete.");
                    } else {
                        appendToLog('UNKNOWN', event.data);
                    }
                } catch (e) {
                    // Fallback for plain text if any
                    appendToLog('INFO', event.data);
                }
            }
        };

        // Send Microphone Audio to Backend
        processor.port.onmessage = (event) => {
            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };

    } catch (e) {
        appendToLog('ERROR', `Initialization Error: ${e.message}`);
        statusDiv.textContent = 'SYSTEM: ERROR';
        statusDiv.style.color = "red";
        connectBtn.disabled = false;
        console.error(e);
    }
};

disconnectBtn.onclick = () => {
    // Terminate Session Completely
    if (ws) ws.close();
    if (avatar) avatar.close();
    if (audioContext) {
        audioContext.close();
    }
    statusDiv.textContent = 'SESSION TERMINATED';
    statusDiv.style.color = "var(--text-color)";
    appendToLog('SYSTEM', 'Session Terminated by User.');
};
