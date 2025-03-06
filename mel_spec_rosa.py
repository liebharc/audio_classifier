import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_file = 'your_audio_file.wav'
y, sr = librosa.load(audio_file)

# Calculate the mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

# Convert the mel spectrogram to decibels for better visualization
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot and save the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')

# Save the plot as a PNG file
plt.savefig('librosa_spectrogram.png')

# Close the plot to free resources
plt.close()
