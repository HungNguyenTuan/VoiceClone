import sys
sys.path.append('./bark-voice-cloning-HuBERT-quantizer')
from pydub import AudioSegment

input = "./samples/temp.mp3"
audio_segment = AudioSegment.from_mp3(input)

# Export as wav
audio_segment.export('samples/temp.wav', format='wav')