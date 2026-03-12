# gizzbrain
A fun application of using a neural net to guess and annotate mp3 songs.

Purpose:
Identify and annotate live music available on the Internet Archive. 

Goals:
* Mine file name to store for help in classifying post prediction
* Identify song title
* Identify artist
* Identify album (hoping there is some differences in tappings and ambient noise to classify album)
* Create Manual validation and option to correct predictions

Packages:
* Mutagen (ID3 tag editing)
* Librosa (for Mel Spectrograms)
* TorchAudio (PyTorch)

Test/Train Data Strategy:
1. Download 500(?) songs
2. Annotate with Mutagen (artist, title, and alubum)
3. Enocde mp3 data as Mel Spectrogram with Librosa (2D matrix, captures frequency, time and volume)
4. Convert matricies to tensors
6. Train net on spectrograms (will start with one artist)


Prediction Workflow:
1. Mine file name for hints at tags
2. 2D Encode
3. Predict title, album, maybe artist
4. Compare to tag hints
5. Save the file with new tags