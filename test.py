import sys
sys.path.append('./bark-voice-cloning-HuBERT-quantizer')
import os
from scipy.io.wavfile import write as write_wav
import numpy as np
import torch
import torchaudio
from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, load_codec_model
from encodec.utils import convert_audio
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert

print("Successfully imported all libs")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = True if device == 'cuda' else False
model = load_codec_model(use_gpu=use_gpu)

preload_models (
    text_use_gpu = use_gpu,
    text_use_small = False,
    coarse_use_gpu = use_gpu,
    coarse_use_small = False,
    fine_use_gpu = use_gpu,
    fine_use_small = False,
    codec_use_gpu = use_gpu,
    force_reload = False
)

hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

# Load the HuBERT modeldevice = 'cuda' if torch.cuda.is_available() else 'cpu'
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

# Load the CustomTokenizer model
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth', map_location=device).to(device)



text_prompt = "A ladybug with mismatched spots landed on a plump blueberry, its tiny legs tickling the fruit's fuzzy skin. Suddenly, a fluffy bumble bee, bigger than the ladybug itself, buzzed by, its pollen-dusted body casting a comical shadow. [laughs]"
audio_filepath = 'samples/temp.mp3'

if not os.path.isfile(audio_filepath):
  raise ValueError(f"Audio file not exists ({audio_filepath})")

wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)

semantic_vectors = hubert_model.forward(wav, input_sample_hz = model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()

voice_filename = 'output.npz'

current_path = os.getcwd()
voice_name = os.path.join(current_path, voice_filename)

np.savez(voice_name, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

# simple generation
audio_array = generate_audio(text_prompt, history_prompt = voice_name, text_temp=0.7, waveform_temp=0.7)

# save audio
filepath = "output/out1.wav" # change this to your desired output path
write_wav(filepath, SAMPLE_RATE, audio_array)