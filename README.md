## audio_classifier

The algorithm works as follows:

- Go through audio data as wav-file in chunks of 16k
- For each chunk calculate the mel spectogram
- Classify the mel spectogram with a vision transformer

The main challenge here was to get the same mel spectogram in Python and Java Script so that the model works
on both platforms.

## Models

| Model         | accuracy | Model size |
| ------------- | -------- | ---------- |
| resnet18      | 0.667979 | 45M        |
| resnet50      | 0.612205 | 98M        |
| squeezenet1_0 | 0.673228 | 4.9M       |
| squeezenet1_1 | 0.690289 | 4.9M       |

accuracy is taken from training on 100 files per category for 5 epochs
