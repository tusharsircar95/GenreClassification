import librosa
import numpy as np

def split_signal(signal,chunk_ratio=0.3, offset_ratio=0.5):
    splits = []
    n = signal.shape[0]
    chunk_size =  int(n * chunk_ratio)
    offset_size = int(chunk_size * offset_ratio)
    for i in range(0,n-chunk_size+1,offset_size):
      splits.append(signal[i:(i+chunk_size)])
    return splits



def convert_signal_to_spec(signal, n_fft=1024, hop_length=512):
    spec = librosa.feature.melspectrogram(signal, n_fft = n_fft, hop_length = hop_length)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec

def load_song(path):
    signal,sr = librosa.load(path)
    return signal
