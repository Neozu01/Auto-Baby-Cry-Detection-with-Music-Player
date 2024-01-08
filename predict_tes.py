import pickle
import librosa
import numpy as np
import os

def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features
'''
def predict_label(file_path, model_path="Model_DT.pkl"):
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Extract features from the audio file
    features = features_extractor(file_path)

    # Make prediction
    prediction = model.predict([features])[0]

    # Define a dictionary to map class labels to class names
    class_label_mapping = {
        0: "Silence",
        1: "Belly Pain",
        2: "Burping",
        3: "Discomfort",
        4: "Hungry",
        5: "Tired"
    }

    # Get the predicted class name
    predicted_label = class_label_mapping.get(prediction, "Unknown")

    # Extract the true class label from the file path
    true_label = file_path.split("/")[-2].capitalize()

    # Print the predicted label and the real label
    print(f"Predicted Label: {predicted_label}")
    print(f"Real Label: {true_label}")

# Example usage:
#file_to_predict = "D:/dataset/tangisan_bayi/tired/7A22229D-06C2-4AAA-9674-DE5DF1906B3A-1436891957-1.1-m-72-ti.wav"
file_to_predict = "D:/dataset/tangisan_bayi/discomfort/7b0e160e-0505-459e-8ecb-304d7afae9d2-1437486974312-1.7-m-04-dc.wav"
predict_label(file_to_predict)
'''
folder_path = "D:/dataset/tangisan_bayi/belly_pain/"
model_path = "Model_DT1.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Skip if the file is not a regular file (e.g., it's a directory)
    if not os.path.isfile(file_path):
        continue

    # Extract features from the audio file
    features = features_extractor(file_path)

    # Make prediction
    prediction = model.predict([features])[0]

    # Define a dictionary to map class labels to class names
    class_label_mapping = {
        0: "Silence",
        1: "Belly Pain",
        2: "Burping",
        3: "Discomfort",
        4: "Hungry",
        5: "Tired"
    }

    # Get the predicted class name
    predicted_label = class_label_mapping.get(prediction, "Unknown")

    # Extract the true class label from the file path
    true_label = os.path.basename(os.path.dirname(file_path)).capitalize()

    # Print the predicted label and the real label
    print(f"File: {file_name}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Real Label: {true_label}")
    print("-" * 20)

#Hasil Tes Model_KNN.pkl
#belly_pain 16/30 = 53%
#burping 9/36 = 25%
#discomfort 27/30 = 90%
#hungry 21/30 = 70%
#silence 32/32 = 100%
#tired 24/28 =86%
#Total 70,6%

#Hasil Tes Model_DT.pkl
#belly_pain 11/30 = 37%
#burping 28/36 = 78%
#discomfort 23/30 = 77%
#hungry 29/30 = 97%
#silence 31/32 = 97%
#tired 16/28 = 57%
#Total 74%

