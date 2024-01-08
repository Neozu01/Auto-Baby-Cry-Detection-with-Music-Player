import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import librosa
import os
import numpy as np
from tqdm import tqdm
import pickle

def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def train_model():
    extracted_features = []

    belly_pain_dir = "D:/dataset/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/audio/belly_pain/"
    for file in os.scandir(belly_pain_dir):
        final_class_label = 1
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_0")

    burping_dir = "D:/dataset/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/audio/burping/"
    for file in os.scandir(burping_dir):
        final_class_label = 2
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_1")

    discomfort_dir = "D:/dataset/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/audio/discomfort/"
    for file in os.scandir(discomfort_dir):
        final_class_label = 3
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_2")
    
    hungry_dir = "D:/dataset/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/audio/hungry/"
    for file in os.scandir(hungry_dir):
        final_class_label = 4
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_3")
    
    tired_dir = "D:/dataset/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/audio/tired/"
    for file in os.scandir(tired_dir):
        final_class_label = 5
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_4")

    silent_dir = "D:/dataset/tangisan_bayi/silence/"
    for file in os.scandir(silent_dir):
        final_class_label = 0
        data = features_extractor(file)
        extracted_features.append([data, final_class_label])
        print("Done_5")


    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

    X = np.array(extracted_features_df['feature'].tolist())
    Y = np.array(extracted_features_df['class'].tolist())

    # Use RandomOverSampler to handle class imbalance
    oversample = RandomOverSampler(sampling_strategy='auto', random_state=69)
    X_resampled, Y_resampled = oversample.fit_resample(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.25, random_state=2)

    #model = KNeighborsClassifier()
    model = RandomForestClassifier()
    #model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    file = open("Model_RF.pkl", "wb")
    pickle.dump(model, file)
    print("Training Success")
    print("Accuracy :", score)

if __name__ == "__main__":
    train_model()
