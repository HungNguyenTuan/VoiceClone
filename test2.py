from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch
import sys
import platform
import numpy as np

import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_codec_model(use_gpu = True)
filename = 'samples/temp.mp3'
wav, sr = torchaudio.load(filename)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0).to(device)
text_prompt = "A ladybug with mismatched spots landed on a plump blueberry, its tiny legs tickling the fruit's fuzzy skin. Suddenly, a fluffy bumble bee, bigger than the ladybug itself, buzzed by, its pollen-dusted body casting a comical shadow. [laughs]"

with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
seconds = wav.shape[-1] / model.sample_rate
semantic_tokens = generate_text_semantic(text_prompt, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)
codes = codes.cpu().numpy()
import numpy as np
voice_name = 'temp'
output_path = 'voice/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

