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

## V1 Summary

### What V1 Is
GizzBrain V1 is a complete, working proof-of-concept for AI-powered live music tagging. Given a directory of untagged (or poorly tagged) live MP3s, it learns to recognize songs from audio fingerprints alone — no lyrics, no metadata hints — and writes the correct ID3 tags back to the files automatically.

The first target corpus is King Gizzard & The Lizard Wizard live recordings from the Internet Archive, though the pipeline is artist-agnostic by design.

### Pipeline at a Glance
```
Raw MP3 directory
      │
      ▼
 gizzbrain scan          # Recursively finds MP3s, parses filenames,
      │                  # extracts existing ID3 tags, writes library.parquet
      ▼
 library.parquet          # Columnar metadata store (song title → file path)
      │
      ▼
 gizzbrain train          # Chunks audio into 5-sec Mel Spectrograms,
      │                   # trains CNN, saves gizzbrain_weights.pt
      ▼
 gizzbrain_weights.pt     # Trained model (~13 MB)
      │
      ▼
 Auto-tagged MP3s         # ID3 Title / Artist / Album written back to files
```

### Architecture
| Layer | Technology | Role |
|---|---|---|
| CLI | `argparse` | Entry point; dispatches `scan` and `train` subcommands |
| Scanning | `mutagen`, `pandas` | Recursive MP3 discovery, regex filename parsing, Parquet output |
| Audio Encoding | `librosa`, `torchaudio` | Converts 5-sec audio chunks → 128-bin Mel Spectrograms (log power) |
| Dataset | `torch.utils.data.Dataset` | Just-in-time tensor loading to keep memory footprint low |
| Model | PyTorch CNN | Conv2d → MaxPool → Flatten → Linear; cross-entropy loss, SGD |
| Hardware | `torch` device detection | Auto-selects CUDA, Apple MPS, AMD DirectML, or CPU |
| Tagging | `mutagen` | Writes predicted Title/Artist/Album back to MP3 ID3 tags |

### V1 Scope & Constraints
- **Vanguard test:** Training targets the **top 10 most frequent songs** in the library — enough to validate the approach without requiring a massive labeled dataset.
- **File-level train/val split (80/20):** All chunks from a given MP3 stay in one split, preventing the model from learning recording-specific artifacts rather than song structure.
- **Minimal CNN:** Single convolutional layer with 32 filters. Intentionally simple — the goal is a working baseline, not peak accuracy.
- **Per-batch normalization:** Applied each batch to prevent NaN loss from amplitude spikes common in live recordings.

### Planned V2 Improvements
- Transformer layer to focus on the most *iconic* moments of each song (choruses, signature riffs) for better discrimination
- Multi-prediction voting per song title to handle extended jams and segues
- Venue/show classification as a second output head
- Larger training corpus beyond the top-10 vanguard set

### Module Map
```
gizzbrain/
├── cli.py       — Command dispatch, workflow orchestration
├── tagger.py    — File scanning, filename parsing, ID3 read/write, Parquet I/O
├── encoder.py   — Audio → Mel Spectrogram, GizzDataset, chunk indexing
└── model.py     — AudioClassifier CNN, training loop, hardware detection
```

License & Authors
Developed by Cam Nowack.