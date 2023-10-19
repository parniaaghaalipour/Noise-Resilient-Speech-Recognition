import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchaudio.transforms import Spectrogram
import torchaudio
import random
from abc import ABC, abstractmethod

# Load audio and normalize 
def load_audio(path):
    # Load waveform from the file
    wave, sr = torchaudio.load(path)
    
    # Normalize waveform into [-1, 1]
    wave = wave / wave.abs().max()
    
    # Average multiple channels if exists to reduce audio into mono channel
    if wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    
    # Remove the channel dimension
    return wave.squeeze()

# Parent class for creating custom audio parser
class AudioParser(ABC):
    @abstractmethod
    def parse_transcript(self, transcript_path):
        # The abstract method for parsing the text transcript, it will be implemented in the child class
        pass

    @abstractmethod
    def parse_audio(self, audio_path):
        # The abstract method for loading the audio data, it will implemented in the child class
        pass

# Noise Injection to the audio for data augmentation
class NoiseInjection():
    def __init__(self, path, sample_rate, noise_levels=(0, 0.5)):
        # Sampling rate of noise
        self.sample_rate = sample_rate
        # Amplitude scale of noise to be injected
        self.noise_levels = noise_levels
        
        # Load noise waveform from the given path
        noise_path = os.path.join(path)
        self.noises = [load_audio(noise_path)]

    def inject(self, data):
        # Choose a random noise sample
        noise = random.choice(self.noises)
        
        # Randomly choose a starting point in noise waveform to ensure random overlay
        start = np.random.randint(0, len(noise) - len(data))
        noise = noise[start:start+len(data)]
        
        # Randomly decide an amplitude scale for the noise chosen
        noise_level = np.random.uniform(*self.noise_levels)
        
        # Combine noise with the original audio data
        data = (1.0 - noise_level) * data + noise_level * noise
        return data

# Converts audio waveform to a spectrogram
class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize = False):
        super(SpectrogramParser, self).__init__()
        
        # Window size for the STFT
        self.window_size = audio_conf['window_size']
        # Stride for the STFT
        self.window_stride = audio_conf['window_stride']
        # Sample rate of the incoming signal
        self.sample_rate = audio_conf['sample_rate']
        
        # Window function for STFT
        self.window = torch.hann_window(window_size).cuda()
        
        # Whether to normalize the spectrogram
        self.normalize = normalize

    def parse_audio(self, audio_path):
        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        D = self.spectrogram(y)
        
        # Calculate the magnitude of the spectrogram
        spect, phase = torchaudio.functional.magphase(D)
        spect = spect.log1p()
        
        # Normalize spectrogram
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
            
        return spect

# Prepare the dataset
class SpectrogramDataset(Dataset):
    def __init__(self, audio_parser, manifest_filepath, labels):
        # List of tuples containing absolute path of audio and corresponding label
        self.data = []
        # Audio parser to convert audio waveform to spectrogram
        self.audio_parser = audio_parser
        self.labels = labels
        # Parse all lines in the manifest file
        with open(manifest_filepath, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                assert len(items) == 2, f"Incorrect format in the manifest file: {line}"
                self.data.append(tuple(items))

    def __getitem__(self, index):
        # Fetch audio and label path from the data list
        audio_path, transcript_path = self.data[index]
        # Load audio and construct spectrogram
        audio = self.audio_parser.parse_audio(audio_path)
        # Parse transcript
        transcript = self.audio_parser.parse_transcript(transcript_path)
        return audio, transcript

    def __len__(self):
        return len(self.data)
