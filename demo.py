import librosa
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(AUDIO_PATH, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    signal, sample_rate = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)

    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish],
                                    sample_rate, 
                                    n_mfcc=num_mfcc, 
                                    n_fft=n_fft, 
                                    hop_length=hop_length)
        mfcc = mfcc.T

    return mfcc

def predict(model, X):
    
    # mapping labels to targets
    target = ['PsyTrance','Bigroom House','pop','metel','disco','blues','reggae','Trap',
              'Hardstyle','classical','rock','hiphop','Bass House','Dubstep','country',
              'Future Bass','Slap House','Bounce','Future House','jazz']
    
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ..., np.newaxis]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)[0]
    prediction = format(target[predicted_index])
    return prediction

def OK():

    AUDIO_PATH = format(entry.get())
    prediction = save_mfcc(AUDIO_PATH=AUDIO_PATH)[..., np.newaxis][0]
    msg = predict(model, prediction)
    messagebox.showinfo(title="Genre", message="file :\n" + AUDIO_PATH + "\n\nGenre is : \n" + msg)

if __name__ == "__main__":

    # load model
    model = load_model("Music_Gnere_Classifier_TOGETHER.h5")
    
    # 視窗
    window = tk.Tk()
    window.title('Music Genre Classifier')
    window.geometry("500x300+500+200")

    # 標示文字
    label = tk.Label(window,
                    text="Please enter your file path here:",
                    font=("Arial", 20))
    label.pack()

    # 輸入欄位
    entry = tk.Entry(window,    # 輸入欄位所在視窗
                    width=50)   # 輸入欄位寬度
    entry.pack()

    # 按鈕
    button = tk.Button(window, text="OK", command=OK)
    button.pack()
    
    window.mainloop()