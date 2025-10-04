# 🗣️ My Own ElevenLabs — Free & Local Voice Cloning

A free, open-source **AI voice cloning** and **text-to-speech (TTS)** setup using **OpenVoice + MeloTTS**.
No subscriptions. No credit limits. No servers.
Just **your voice, your rules, running locally** on your own PC.

Built for **content creators, podcasters, and researchers** who want full control over their AI voice tools.

---

## 🚀 Features

* 🧠 **Voice Cloning** — Generate speech that sounds like your recorded voice
* 🎧 **Text-to-Speech (TTS)** — Convert any text into natural speech
* 💻 **Offline & Private** — Runs fully on your PC, no cloud required
* 🌎 **Multilingual Support** — English, Spanish, French, Chinese, Japanese, Korean
* 🎨 **Open Source & Free** — Build your own ElevenLabs-style system at no cost
* ⚙️ **Gradio Interface** — Simple web UI for testing voices instantly

---

## 📦 Requirements

* Windows 10 / 11
* Python 3.10+
* Git
* FFmpeg
* (Optional) NVIDIA GPU with CUDA 12.1 for faster processing

---

## 🛠 Full Setup Guide (Windows)

> Root Directory: `..\OpenVoice\`

---

### 1️⃣ Clone OpenVoice Repository

```powershell
git clone https://github.com/myshell-ai/OpenVoice.git .
```

✅ `..\OpenVoice\` now contains the OpenVoice repo files.

---

### 2️⃣ Create & Activate Virtual Environment

```powershell
py -3.10 -m venv openVoice_venv
openVoice_venv\Scripts\Activate.ps1
```

Your PowerShell prompt should now look like this:

```
(openVoice_venv)
```

---

### 3️⃣ Install Dependencies

#### 3.1 Install PyTorch (CUDA 12.1)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only users:

```powershell
pip install torch torchvision torchaudio
```

#### 3.2 Install OpenVoice + Extras

```powershell
pip install -e .
pip install gradio soundfile
```

#### 3.3 Install MeloTTS

```powershell
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

---

### 4️⃣ Download Model Checkpoints

#### 4.1 OpenVoice Converter (V2)

📥 [Hugging Face – myshell-ai](https://huggingface.co/myshell-ai)

Place into:

```
..\OpenVoice\checkpoints_v2\converter\
```

#### 4.2 MeloTTS Models

📥 [Hugging Face – myshell-ai](https://huggingface.co/myshell-ai)

Place into:

```
..\OpenVoice\checkpoints_v2\melo\en\
```

---

### ✅ Final Folder Structure

```
..\OpenVoice\
 ├── checkpoints_v2\
 │   ├── converter\
 │   │   ├── config.json
 │   │   └── checkpoint.pth
 │   └── melo\
 │       ├── en\*.pth
 ├── samples\
 │   └── sample_1.wav
 ├── openVoice_venv\
 └── ov.py
```

---

### 5️⃣ Install NLTK Data

```powershell
python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng punkt
```

---

### 6️⃣ Install FFmpeg

1. Download from: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the ZIP file (e.g. `ffmpeg-2025-win64-gpl`).
3. Move it to a folder like `C:\ffmpeg\`.
4. Add `C:\ffmpeg\bin` to your **Windows PATH**.

✅ Test with:

```powershell
ffmpeg -version
```

---

## 💻 Run the App

Start the Gradio interface:

```powershell
python ov.py
```

Then open your browser and visit:

```
http://127.0.0.1:7860
```

You’ll see two tabs:

* 🎙️ **Cloned Voice** (OpenVoice + MeloTTS)
* 🗣️ **Original Voice** (MeloTTS default)

---

## 🧩 Example Code (Main Script)

```python
import torch
import gradio as gr
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

converter = ToneColorConverter(
    config_path="checkpoints_v2/converter/config.json",
    device=device
)
converter.load_ckpt("checkpoints_v2/converter/checkpoint.pth")

reference_speaker, *_ = se_extractor.get_se("samples/demo_speaker1.mp3", converter)

def generate_cloned_audio(text, language="EN"):
    output_path = "output/openvoice_v2_out.wav"
    tmp_path = "output/tmp_base.wav"
    tts_model = TTS(language=language, device=device)
    tts_model.tts_to_file(text=text, speaker_id=1, output_path=tmp_path)
    src_se = torch.zeros_like(reference_speaker).to(device)
    converter.convert(tmp_path, src_se, reference_speaker, output_path=output_path)
    return output_path

def generate_original_audio(text, language="EN"):
    output_path = "output/melo_tts_out.wav"
    tts_model = TTS(language=language, device=device)
    tts_model.tts_to_file(text=text, speaker_id=1, output_path=output_path)
    return output_path

# Gradio Interface
with gr.Blocks(title="OpenVoice + MeloTTS Demo") as demo:
    with gr.Tab("Cloned Voice"):
        gr.Interface(fn=generate_cloned_audio, inputs=gr.Textbox(label="Enter text"), outputs=gr.Audio(label="Cloned Voice"))
    with gr.Tab("Original MeloTTS Voice"):
        gr.Interface(fn=generate_original_audio, inputs=gr.Textbox(label="Enter text"), outputs=gr.Audio(label="Original Voice"))

if __name__ == "__main__":
    demo.launch()
```

---

## 🧠 Tips

* Record a **clean 1–3 minute sample** of your own voice and save it as `samples/yourname.mp3`.
* Edit this line in the script:

  ```python
  reference_speaker, *_ = se_extractor.get_se("samples/yourname.mp3", converter)
  ```
* Use that as your **personal cloned voice**.

---

## 💡 Credits

* 🧩 [OpenVoice](https://github.com/myshell-ai/OpenVoice) — Voice conversion engine
* 🔊 [MeloTTS](https://github.com/myshell-ai/MeloTTS) — Text-to-speech model
* 🧰 [Gradio](https://gradio.app/) — Web interface
* 🎛️ [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) — Audio processing

---

## ⚖️ License

This project is **free for research and personal use**.
For commercial use, please review the individual licenses of **OpenVoice** and **MeloTTS** on their GitHub pages.

---

## 💬 Contribute

Found a bug or idea for improvement?
Open an issue or pull request — let’s make open-source TTS better together!

---

## ⭐ Support the Project

If this helps you:

* Give it a ⭐ on GitHub
* Share your results online
* Tag the project to inspire others