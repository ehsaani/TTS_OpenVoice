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
        gr.Interface(
            fn=generate_cloned_audio,
            inputs=gr.Textbox(label="Enter text"),
            outputs=gr.Audio(label="Cloned Voice")
        )
    with gr.Tab("Original MeloTTS Voice"):
        gr.Interface(
            fn=generate_original_audio,
            inputs=gr.Textbox(label="Enter text"),
            outputs=gr.Audio(label="Original Voice")
        )

if __name__ == "__main__":
    demo.launch()
