let audioContext;
let ws;
let isConnected = false;
let avatar = null;

const connectBtn = document.getElementById('connectBtn');
connectBtn.disabled = true; // Ensure disabled initially
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
let activeAvatarId = '7b888024-f8c9-4205-95e1-78ce01497bda';
let activeVoiceId = '';

async function checkAvailability() {
    try {
        log('Checking Avatar availability...');

        // Check Avatars
        const avatarsResp = await fetch('/avatars');
        if (avatarsResp.ok) {
            const avatarsData = await avatarsResp.json();
            const avatars = avatarsData.data ? avatarsData.data.avatars : (avatarsData.avatars || []);

            console.log("Full Avatar List:", avatars);

            if (avatars.length > 0) {
                // ЛОГИКА АВТОВЫБОРА:
                // Если ID пустой или такого ID нет в списке — берем ПЕРВЫЙ доступный
                let selectedAvatar = avatars.find(a => a.avatar_id === activeAvatarId);

                if (!selectedAvatar) {
                    selectedAvatar = avatars[0]; // Берем самый первый из списка
                    activeAvatarId = selectedAvatar.avatar_id; // Сохраняем его ID
                    log(`⚠️ Auto-selected Avatar: ${selectedAvatar.name}`);
                } else {
                    log(`✅ Found requested Avatar: ${selectedAvatar.name}`);
                }

                // --- НОВАЯ ЛОГИКА ГОЛОСА ---
                // Если у аватара нет голоса по умолчанию, ставим стандартный английский
                if (selectedAvatar.default_voice_id) {
                    activeVoiceId = selectedAvatar.default_voice_id;
                } else {
                    activeVoiceId = "2d5b0e6cf361460aa7fc47e3eee4ba54"; // Стандартный мужской голос (или выберите женский)
                    log("Avatar has no default voice. Using fallback voice.");
                }
                // ---------------------------

                log(`Using Avatar: ${selectedAvatar.name} | Voice: ${activeVoiceId}`);

                // Установка картинки (Poster)
                const imageUrl = selectedAvatar.preview_image_url || selectedAvatar.thumbnail_url;
                if (imageUrl) {
                    videoElement.poster = imageUrl;
                }

                // Разблокировка кнопки
                connectBtn.disabled = false;
                statusDiv.textContent = 'Ready to Connect';
            } else {
                log("❌ ERROR: No Interactive Avatars found in your HeyGen account!");
                statusDiv.textContent = 'No Avatars Found';
                // Подсказка пользователю
                log("Please go to HeyGen Labs -> Interactive Avatar and create/select one.");
            }
        } else {
            log('Failed to fetch avatars list');
        }

        // Check Voices
        const voicesResp = await fetch('/voices');
        if (voicesResp.ok) {
            const voicesData = await voicesResp.json();
            const voices = voicesData.data ? voicesData.data.voices : (voicesData.voices || []);

            const voice = voices.find(v => v.voice_id === activeVoiceId);
            if (voice) {
                log(`Voice ${activeVoiceId} is available: ${voice.name}`);
            } else {
                log(`WARNING: Default Voice ${activeVoiceId} not found!`);
                if (voices.length > 0) {
                    // Try to find an English voice or just take the first one
                    const englishVoice = voices.find(v => v.language === 'English' || v.name.includes('English'));
                    if (englishVoice) {
                         activeVoiceId = englishVoice.voice_id;
                         log(`Switched to available voice: ${activeVoiceId} (${englishVoice.name})`);
                    } else {
                         activeVoiceId = voices[0].voice_id;
                         log(`Switched to first available voice: ${activeVoiceId} (${voices[0].name})`);
                    }
                }
            }
        } else {
            log('Failed to fetch voices list');
        }

    } catch (e) {
        log(`Availability check failed: ${e.message}`);
    }
}

// Check availability on load
checkAvailability();

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
        log(`Creating session with Avatar ID: ${activeAvatarId}`);
        // ИЗМЕНЕНО: Запрос на наш сервер, а не на api.heygen.com
        const response = await fetch('/heygen/session/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                token: token, // Передаем токен серверу
                quality: 'medium',
                avatar_name: activeAvatarId,
                voice_id: activeVoiceId
            })
        });
        const data = await response.json();
        if (!data.data) {
            console.error("Server Error:", data);
            // Превращаем объект ошибки в строку, чтобы прочитать её на экране
            throw new Error(JSON.stringify(data.error) || 'Failed to create session');
        }
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
                    await fetch('/heygen/ice', { // ИЗМЕНЕНО
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            token: this.token,
                            session_id: this.sessionId,
                            candidate: event.candidate
                        })
                    });
                }
            };

            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(serverSdp));
            const localSdp = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(localSdp);

            await fetch('/heygen/session/start', { // ИЗМЕНЕНО
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    token: this.token,
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
            await fetch('/heygen/task', { // ИЗМЕНЕНО
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    token: this.token,
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
            await fetch('/heygen/session/stop', { // ИЗМЕНЕНО
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    token: this.token,
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
