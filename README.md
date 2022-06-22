# Emotion Classifier
**Emotion_Classifier** is a web app that recognises 4 different emotions (Happy, Angry, Sad, Neutral ) in an audio file. There is an abuse detection model the detects abuse in an audio. We are extracting various features of an audio such as **MFCC**, **MelSpectrogram**, **chroma_stft**, **chrom_cqt** etc. to classify the audio emotion and type of audio(Abusive or not). We are using libraries like sklearn, librosa, padsas, numpy etc. for the training and testing of the model. We have trained the model on Google Collab and saved weights and biases of two different model using pickel library. In backend we have used this pickel file for model inferencing file and used it in our app.
  
  
Finding a suitable dataset for the training of the model was very challlenging as we prioritize hindi language. The dataset for emotion recognition was collected from <a href="https://www.kaggle.com/datasets/bitlord/urdu-language-speech-dataset"> **Kaggle**</a>. For abuse detection we are using the recent <a href = "https://drive.google.com/drive/folders/1geQ4PlXGsNCvPQDT3tKztvAu817PB5TP"> **ADIMA** </a> by <a href = "https://sharechat.com/research/adima"> ShareChat </a>.
    
    
This <a href = "https://arxiv.org/pdf/2202.07991.pdf"> Paper </a> contains state of art deep learning approach for abuse detection but this could be computationally very heavy and hence for better user experience we have implemented our Machine Learning based approach for the same.


The web app is built using **StreamLit**.

<hr>

## Instructions to run our web app:
 * clone this repo
 * pip install -r requirements.txt
 * change to main folder directory 
 * In terminal run this command:
    * streamlit run app.py
 <hr>

## Live WebAPP is being hosted using streamlit cloud based share support:-

*  https://share.streamlit.io/gaurav7888/emotion_classifier/main/app.py

### Live demo of the web app:
*    https://youtu.be/w0oLvqrHkok

<hr>

## Mobile App

*  "https://github.com/Gaurav7888/Emotion_Classifier/blob/main/_Emotion_detection_15653441.apk"

### Live demo of the mobile app:
*    https://youtu.be/3QQZnerBKz0




