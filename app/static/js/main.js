let audioContext;
let ws;
let isConnected = false;
let nextStartTime = 0;
let analyser;
let mouthCanvas;
let mouthCtx;
let animationId;

const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const statusDiv = document.getElementById('status');
const logsDiv = document.getElementById('logs');

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

        // Setup Analyser for Lip-Sync
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        analyser.smoothingTimeConstant = 0.5;
        analyser.connect(audioContext.destination);

        mouthCanvas = document.getElementById('mouth-canvas');
        mouthCtx = mouthCanvas.getContext('2d');

        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(audioContext, 'pcm-processor');

        source.connect(processor);
        // User audio doesn't go to analyser (we don't want Dos to lip-sync user audio)
        // But user audio goes to destination?
        // Wait, processor.connect(audioContext.destination) in original code.
        // If we connect processor to destination, we hear ourselves?
        // In audio-processor.js, it silences output. So it's fine.
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
            animateMouth();
        };

        ws.onclose = () => {
            isConnected = false;
            statusDiv.textContent = 'Disconnected';
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            log('WebSocket closed');
            if (animationId) cancelAnimationFrame(animationId);
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
    // Connect to analyser for lip-sync, which is connected to destination
    source.connect(analyser);

    const currentTime = audioContext.currentTime;
    if (nextStartTime < currentTime) {
        nextStartTime = currentTime;
    }
    source.start(nextStartTime);
    nextStartTime += buffer.duration;

}

function animateMouth() {
    if (!isConnected) return;
    animationId = requestAnimationFrame(animateMouth);

    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    let sum = 0;
    for(let i = 0; i < bufferLength; i++) {
        // value is 128 for silence (0 amplitude)
        const v = (dataArray[i] - 128) / 128;
        sum += v * v;
    }
    const rms = Math.sqrt(sum / bufferLength);

    drawMouth(rms);
}

function drawMouth(amplitude) {
    if (!mouthCtx) return;

    mouthCtx.clearRect(0, 0, mouthCanvas.width, mouthCanvas.height);

    const centerX = 100;
    const centerY = 145;
    const width = 50;

    // Scale amplitude
    // Amplitude is typically small for speech, need boost
    const sensitivity = 5.0;
    let openHeight = Math.max(2, amplitude * 100 * sensitivity);
    if (openHeight > 60) openHeight = 60; // Max open

    mouthCtx.fillStyle = '#333'; // Inside of mouth
    mouthCtx.beginPath();

    if (amplitude < 0.01) {
        // Closed mouth
        mouthCtx.moveTo(centerX - width/2, centerY);
        mouthCtx.lineTo(centerX + width/2, centerY);
        mouthCtx.strokeStyle = '#333';
        mouthCtx.lineWidth = 3;
        mouthCtx.stroke();
    } else {
        // Open mouth
        mouthCtx.ellipse(centerX, centerY, width/2, openHeight/2, 0, 0, 2 * Math.PI);
        mouthCtx.fill();
    }
}
