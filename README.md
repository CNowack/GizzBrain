# GizzBrain
**Automate the ID3 tags on your live music files!**

## Overview
GizzBrain is an AI-driven live music classification and annotation tool designed to autonomously identify and tag live music recordings. Developed for use with live recordings by King Gizzard & The Lizard Wizard sourced from the Internet Archive but can be trained on any artist(s).

GizzBrain ingests raw, unannotated MP3 files, and embeds them into 2D audio spectrograms (basically an image) to train a CNN that predicts the correct song title and metadata (working on concert venue). It automates the process of organizing vast libraries of live recordings by updating the ID3 tags of the original audio files.

## AI/ML Architecture

* **Convolutional Neural Network (CNN):** At the core of GizzBrain is a custom PyTorch-based CNN (`AudioClassifier`). It utilizes 2D convolutional layers and max pooling to extract local acoustic patterns and temporal dependencies from audio files.
* **Mel Spectrogram Encoding:** Raw audio is not fed directly into the model. Instead, `librosa` is used to convert 5-second audio chunks into Mel Spectrograms. This 2D matrix representation captures frequency, time, and decibel volume, effectively translating the audio classification task into a specialized computer vision problem.
* **Anti-Leakage Data Splitting:** To ensure the model generalizes to *new* live performances rather than simply memorizing specific recordings, GizzBrain utilizes a strict file-level training and validation split. All chunks from a specific MP3 file are isolated to either the training or validation set, preventing data leakage across acoustic environments.
    - several improvements are planned to increase prediction accuracy including adding a transformer layer to pick the *iconic* song moments that are most indicative of a specific title. Also build in multiple predictions per title to allow for sick multi-song jams.
* **Just-in-Time Data Loading:** Optomizes memory, the `GizzDataset` class processes audio into PyTorch tensors dynamically during the training loop.
* **Hardware Acceleration:** The model dynamically routes tensor computations to the optimal available hardware, supporting NVIDIA CUDA, Apple Silicon (MPS), and AMD DirectML.

## Features
* **Directory Scanning:** Recursively scans given directories to identify MP3 files and extracts existing metadata or filename hints.
* **Parquet Storage:** Compiles library metadata into high-performance `.parquet` formats utilizing `pyarrow` and `pandas`.
* **Automated ID3 Tagging:** Uses `mutagen` to write AI-predicted metadata (Title, Artist, Album) directly to the MP3 files.
* **Command Line Interface:** Fully featured CLI for execution of scanning and training workflows.

## Installation

Ensure you have Python 3.10 or higher installed. You can install GizzBrain and its dependencies using `pip`:
```
git clone [https://github.com/cnowack/gizzbrain.git](https://github.com/cnowack/gizzbrain.git)
cd gizzbrain
pip install .
```

Key Dependencies: `torch`, `torchaudio`, `librosa`, `pandas`, `pyarrow`, `numpy`, `mutagen`.

## Usage
GizzBrain is operated via a Command Line Interface (CLI) with two primary workflows: scan and train.

1. Scanning a Library
Analyze a directory of raw MP3s to extract filename parameters and generate a baseline dataset for the neural network.


```gizzbrain scan /path/to/mp3/directory --output library.parquet```

2. Training the Neural Network

    Train the PyTorch CNN on the compiled dataset. This process targets the top 10 most frequent songs, chunks the audio into 5-second tensors, and fits the model.

```
gizzbrain train --data library.parquet --epochs 15 --batch-size 32
Upon completion, the trained weights are saved to gizzbrain_weights.pt.
```

License & Authors
Developed by Cam Nowack.