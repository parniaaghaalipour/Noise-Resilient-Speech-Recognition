## Noisy Speech Recognition Data Loader

This Python code is designed to load and preprocess data for a Neural Network model. This model is specifically intended to perform speech recognition tasks on noisy audio data.

### Function Overview
Here's a brief overview of what each class/function in the repository does:

- **load_audio**: This function accepts a file path and returns the normalized mono channel waveform for the audio file at that path.

- **AudioParser**: This is an abstract base class that creates custom audio parsers. It includes abstract methods for parsing both the audio data and its associated transcript, to be implemented by subclasses.

- **NoiseInjection**: This class is used to randomly inject noise into the audio for data augmentation purposes. The injected noise is randomly selected from a list of noise waveforms.

- **SpectrogramParser**: This class is a subclass of AudioParser that converts the loaded waveform into a spectrogram. It also provides the capability to normalize these spectrograms.

- **SpectrogramDataset**: This class is a subclass of PyTorch's Dataset class, designed specifically for spectrogram data. It uses the audio parser to convert audio files and transcripts into spectrograms.

### Usage

```python
audio_conf = {'window_size': 0.02, 'window_stride': 0.01, 'sample_rate': 16000}
audio_parser = SpectrogramParser(audio_conf)

speech_dataset = SpectrogramDataset(audio_parser, 'audio_manifest.csv', labels) 
data_loader = DataLoader(speech_dataset, batch_size=16, shuffle=True)
```

In this example, an instance of *SpectrogramParser* is created with a specific audio configuration. A *SpectrogramDataset* is then prepared using this parser, along with a manifest file containing file paths for the audio files and their transcripts. The audio dataset is then wrapped with a DataLoader, which makes it easier to iterate over the data in minibatches.

### Setup

You would need to have `torchaudio`, `torch` and `numpy` installed in the Python environment to use this script. Use pip to install the required package.

```shell
pip install torchaudio torch numpy
```

### Note
Paths provided in the manifest file should be absolute paths towards every audio and its corresponding transcription file.

### Contributions
Feel free to contribute to this project by creating a fork and submitting a pull request. For significant changes, please open an issue first to discuss the proposed changes. Since this is a typical data loader for a deep learning project in PyTorch, any significant improvement will be appreciated.
