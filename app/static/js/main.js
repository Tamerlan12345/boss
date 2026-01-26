let audioContext;
let ws;
let isConnected = false;
let nextStartTime = 0;

const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const statusDiv = document.getElementById('status');
const logsDiv = document.getElementById('logs');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');

function log(message) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    logsDiv.prepend(div);
}

connectBtn.onclick = async () => {
    try {
        statusDiv.textContent = 'Connecting...';

        // Initialize AudioContext
        // Try to force 16kHz to match Gemini requirement, otherwise we might need resampling
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        log(`AudioContext created. Sample Rate: ${audioContext.sampleRate}`);

        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(audioContext, 'pcm-processor');

        source.connect(processor);
        processor.connect(audioContext.destination);

        // Check protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);
        ws.binaryType = 'arraybuffer';

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
        };

        ws.onerror = (e) => {
            log(`WebSocket error: ${e}`);
        };

        ws.onmessage = async (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Play audio
                playAudioChunk(event.data);
            } else {
                log(`Received text: ${event.data}`);
            }
        };

        processor.port.onmessage = (event) => {
            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);

                // Visualization for User Audio (Blue)
                const int16Data = new Int16Array(event.data);
                const float32Data = new Float32Array(int16Data.length);
                for (let i = 0; i < int16Data.length; i++) {
                    float32Data[i] = int16Data[i] / 32768.0;
                }
                drawVisualizer(float32Data, '#00ccff');
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

function playAudioChunk(arrayBuffer) {
    // Gemini usually sends 24kHz PCM (Mono).
    const int16Data = new Int16Array(arrayBuffer);
    const float32Data = new Float32Array(int16Data.length);

    for (let i = 0; i < int16Data.length; i++) {
        float32Data[i] = int16Data[i] / 32768.0;
    }

    // Create buffer. 24000 is the default for Gemini Live
    const buffer = audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);

    const currentTime = audioContext.currentTime;
    if (nextStartTime < currentTime) {
        nextStartTime = currentTime;
    }
    source.start(nextStartTime);
    nextStartTime += buffer.duration;

    // Visualize (AI Green)
    drawVisualizer(float32Data, '#00ff00');
}

function drawVisualizer(data, color) {
    // Fade out previous frame
    canvasCtx.fillStyle = 'rgba(51, 51, 51, 0.3)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = color;
    canvasCtx.beginPath();

    const step = Math.ceil(data.length / canvas.width);
    const amp = canvas.height / 2;

    for (let i = 0; i < canvas.width; i++) {
        let min = 1.0;
        let max = -1.0;

        for (let j = 0; j < step; j++) {
            const val = data[i * step + j];
            if (val < min) min = val;
            if (val > max) max = val;
        }

        canvasCtx.moveTo(i, amp + min * amp);
        canvasCtx.lineTo(i, amp + max * amp);
    }
    canvasCtx.stroke();
}
