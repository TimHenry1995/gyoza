import os, numpy as np, librosa
import matplotlib.pyplot as plt
#import pywt
from gyoza.utilities import file_management as fm

if __name__ == "__main__":
    # Read a sample file
    filename = os.path.join("src","gyoza","sample_files","63.wav")
    fm.load()
    # Read an audio file using librosa
    y, sr = librosa.load(filename)
    y = y[:len(y)//100]
    D = librosa.stft(y)
    y_hat = librosa.istft(D)


    level = 10
    #coeffs = pywt.wavedec(y, 'db6', level=level)
    #y_hat = pywt.waverec(coeffs, 'db6')
    
    plt.figure()
    plt.plot(y)
    plt.plot(y_hat)
    plt.show()
    
    plt.subplots(nrows=level+1)
    for l in range(level):
        plt.subplot(level, 1, level-l)
        #plt.imshow(coeffs[l][np.newaxis,:], aspect='auto')
        plt.axis('off')

    
    plt.tight_layout()
    plt.show()

    """

    wavelet = 'morl' # wavelet type: morlet
    sr = 8000 # sampling frequency: 8KHz
    widths = np.arange(1, 64) # scales for morlet wavelet 
    print("These are the scales that we are using: ", widths)
    dt = 1/sr # timestep difference

    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    print("These are the frequencies that re associated with the scales: ", frequencies)

    # Compute continuous wavelet transform of the audio numpy array
    wavelet_coeffs, freqs = pywt.cwt(y, widths, wavelet = wavelet, sampling_period=dt)
    print("Shape of wavelet transform: ", wavelet_coeffs.shape)

    # Display the scalogram. We will display a small part of scalogram because the length of scalogram is too big.
    plt.imshow(wavelet_coeffs[:,:400], cmap='coolwarm')
    plt.xlabel("Time")
    plt.ylabel("Scales")
    plt.yticks(widths[0::11])
    plt.title("Scalogram")
    plt.show()"""