from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import struct
from string import Template
import time
import uvicorn
import torch
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer 

app = FastAPI()

template = """
<html><head><meta charset="utf-8">
<title>Whisper test.</title></head>
<body>
 <h1>Текст</h1>
 <form action="#" id="form" method="POST">
  <textarea id="txt" name="text"></textarea>
  <button id="submit-button" disabled type="submit">Отправить</button>
 </form>
 <div id="messages"></div>
 <script>
    const ws = new WebSocket("$ws_address")
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        const form = document.getElementById("form");
        const txt = document.getElementById("txt");
        const button = document.getElementById("submit-button")
        button.removeAttribute("disabled");
                    
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const value = txt.value;
            ws.send(value);
        });

        ws.onmessage = (evt) => {
            playByteArray(evt.data);
        }
    }

    const context = new AudioContext();

    function playByteArray(arrayBuffer) {
        const bufferView = new Uint8Array(arrayBuffer);
        context.decodeAudioData(arrayBuffer, play);
    }

    // Play the loaded file
    function play(buf) {
        // Create a source node from the buffer
        const source = context.createBufferSource();
        source.buffer = buf;
        // Connect to the final output node (the speakers)
        source.connect(context.destination);
        // Play immediately
        source.start(0);
    }
 </script>
</body>
"""

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# # List available 🐸TTS models and choose the first one
model_name = TTS().list_models()[0]
# # Init TTS
# model_name = "tts_models/multilingual/multi-dataset/bark"
tts = TTS(model_name).to(device)
tts.speakers
print("speakers", tts.speakers)

print("languages", tts.languages)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        F_NAME="output.wav"
        # wav = np.array(tts.tts(data, speaker=tts.speakers[0], language=tts.languages[0]))
        # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

        # writer = BytesIO()
        # scipy.io.wavfile.write(writer, 1, wav_norm.astype(np.int16))
        # await websocket.send_bytes(writer.getvalue())
        tts.tts_to_file(data, speaker=tts.speakers[0], language=tts.languages[0], emotion="Happy", file_path=F_NAME)
        with open(F_NAME, 'rb') as file:
            data = file.read()
            await websocket.send_bytes(data)


@app.get("/", response_class=HTMLResponse)
async def root():
    values = {
        "ws_address": "ws://localhost:8000/ws"
    }
    return Template(template).substitute(values)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
