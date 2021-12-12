import librosa
import numpy as np
from tensorflow.keras.models import load_model

# load model
model = load_model('Music_Gnere_Classifier_TOGETHER.h5')

x = input("Drag your file here:")
AUDIO_PATH = x
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    
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
                    mfcc = mfcc.T[np.newaxis]

    return mfcc

def predict(model, X):
    
    # mapping labels to targets
    target = ['PsyTrance','Bigroom House','pop','metel','disco','blues','reggae','Trap',
              'Hardstyle','classical','rock','hiphop','Bass House','Dubstep','country',
              'Future Bass','Slap House','Bounce','Future House','jazz']
    
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)[0]
    
    print("預測結果為: {}".format(target[predicted_index]))

if __name__ == "__main__":

    # pick  sample to predict
    prediction = save_mfcc()[..., np.newaxis][0]

    # predict sample
    predict(model, prediction)