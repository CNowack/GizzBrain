 GizzBrain Evaluation Report V1
 ==============================================================
    Split: val  |  Files: 10  |  Chunks: 2082

    Chunk Accuracy :   47.8%   (995 / 2082 correct)
    File Accuracy  :   70.0%   (7 / 10 correctly tagged)

    Per-Song Breakdown:
    Song                              Files  Correct    Acc   Avg Conf
    --------------------------------------------------------------
    2.02 Killer Year                      1        0     0%      74.6%
    Dreams                                1        1   100%      89.0%
    Extinction                            1        0     0%      73.6%
    Gilgamesh                             1        1   100%      93.0%
    Kepler-22b                            1        1   100%      83.8%
    Perihelion                            1        1   100%      83.7%
    Set                                   1        1   100%      85.7%
    Shanghai                              1        1   100%      87.5%
    Superposition                         1        0     0%      86.0%
    The Bitter Boogie                     1        1   100%      79.6%

    Top Confusions (file level):
    "2.02 Killer Year"  →  "Perihelion"  (1 file)
    "Extinction"  →  "Kepler-22b"  (1 file)
    "Superposition"  →  "Kepler-22b"  (1 file)

GizzBrain V1.1
==============================================================
### Changes:
* 30 second trim of beginning and end of each song
## Training Report

    (gizzbrain) PS C:\Users\TheDankTower\Code\GizzBrain> gizzbrain train --data data/metadata/library.parquet --epochs 15 --batch-size 32
    Loading metadata from data/metadata/library.parquet...
    Vanguard Test: Targeting Top 10 songs: ['Kepler-22b', 'Gilgamesh', 'Extinction', 'Set', 'The Bitter Boogie', 'Perihelion', 'Superposition', '2.02 Killer Year', 'Dreams', 'Shanghai']
    Split completed. Training MP3s: 19 | Validation MP3s: 10
    Chunking training audio (this may take a few minutes)...
    Chunking validation audio...
    Hardware allocated: AMD DirectML
    Starting training for 15 epochs on privateuseone:0...
    Epoch 1/15 | Train Loss: 5.7915 | Train Acc: 39.85% | Val Loss: 1.7405 | Val Acc: 46.45%
    Epoch 2/15 | Train Loss: 0.5200 | Train Acc: 82.95% | Val Loss: 1.8112 | Val Acc: 57.06%
    Epoch 3/15 | Train Loss: 0.1819 | Train Acc: 94.56% | Val Loss: 1.9161 | Val Acc: 60.13%
    Epoch 4/15 | Train Loss: 0.0769 | Train Acc: 98.13% | Val Loss: 2.1678 | Val Acc: 59.13%
    Epoch 5/15 | Train Loss: 0.0506 | Train Acc: 98.76% | Val Loss: 2.6092 | Val Acc: 58.26%
    Epoch 6/15 | Train Loss: 0.0305 | Train Acc: 99.28% | Val Loss: 2.3696 | Val Acc: 61.67%
    Epoch 7/15 | Train Loss: 0.0050 | Train Acc: 100.00% | Val Loss: 2.3655 | Val Acc: 62.39%
    Epoch 8/15 | Train Loss: 0.0025 | Train Acc: 100.00% | Val Loss: 2.4690 | Val Acc: 61.77%
    Epoch 9/15 | Train Loss: 0.0018 | Train Acc: 100.00% | Val Loss: 2.5187 | Val Acc: 62.39%
    Epoch 10/15 | Train Loss: 0.0014 | Train Acc: 100.00% | Val Loss: 2.5728 | Val Acc: 61.77%
    Epoch 11/15 | Train Loss: 0.0012 | Train Acc: 100.00% | Val Loss: 2.6187 | Val Acc: 61.67%
    Epoch 12/15 | Train Loss: 0.0011 | Train Acc: 100.00% | Val Loss: 2.6708 | Val Acc: 61.14%
    Epoch 13/15 | Train Loss: 0.0010 | Train Acc: 100.00% | Val Loss: 2.6812 | Val Acc: 61.48%
    Epoch 14/15 | Train Loss: 0.0009 | Train Acc: 100.00% | Val Loss: 2.7202 | Val Acc: 61.58%
    Epoch 15/15 | Train Loss: 0.0008 | Train Acc: 100.00% | Val Loss: 2.7481 | Val Acc: 61.38%
    Training complete. Weights saved to gizzbrain_weights.pt | Label map saved to label_map.json

## Evaluation Report
    Split: val  |  Files: 10  |  Chunks: 2082

    Chunk Accuracy :   61.4%   (1278 / 2082 correct)
    File Accuracy  :  100.0%   (10 / 10 correctly tagged)

    Per-Song Breakdown:
    Song                              Files  Correct    Acc   Avg Conf
    --------------------------------------------------------------
    2.02 Killer Year                      1        1   100%      85.1%
    Dreams                                1        1   100%      93.4%
    Extinction                            1        1   100%      86.2%
    Gilgamesh                             1        1   100%      97.9%
    Kepler-22b                            1        1   100%      90.7%
    Perihelion                            1        1   100%      90.3%
    Set                                   1        1   100%      92.7%
    Shanghai                              1        1   100%      91.3%
    Superposition                         1        1   100%      91.9%
    The Bitter Boogie                     1        1   100%      86.4%

    No confusions — all files correctly tagged!