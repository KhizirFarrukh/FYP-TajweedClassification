import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display

from pydub import AudioSegment

from flask import Flask, render_template, redirect, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename

# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn import metrics

app = Flask(__name__)
app.config["SECRET_KEY"] = 'secret_key'
app.config["UPLOAD_FOLDER"] = 'static/files'
app.config["OCSVM_MODEL_PATH"] = os.path.join(os.path.dirname(os.getcwd()),'model','one_class_svm_model.sav')

class UploadFileForm(FlaskForm):
  file = FileField("File", validators=[InputRequired()])
  submit = SubmitField("Generate Results")

@app.route('/', methods=['GET','POST'])
def home():
  form = UploadFileForm()
  if form.validate_on_submit():
    file = form.file.data
    file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config["UPLOAD_FOLDER"],secure_filename(file.filename)))
    return redirect("/result?file="+secure_filename(file.filename))
  return render_template('index.html', form=form)

@app.route('/result')
def result():
  args = request.args
  filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config["UPLOAD_FOLDER"],args.get("file"))

  wav_audio_file = GetWavFormatFile(filename)

  spectrogram_image_file = CreateSpectrogram(wav_audio_file)

  result = SpectrogramClassifier(spectrogram_image_file)

  return render_template('result.html', value=result+" Qalqalah Recitation")

def GetWavFormatFile(audio_file):
  base_name, ext = os.path.splitext(audio_file)
  wav_file = base_name + ".wav"

  if ext.lower() == ".wav":
    return audio_file
  else:
    try:
      audio = AudioSegment.from_file(audio_file, format=ext[1:])
      audio.export(wav_file, format="wav")
      print("File converted to WAV format.")
      return wav_file
    except Exception as e:
      print("Error converting file to WAV format:", str(e))
      return None

def CreateSpectrogram(audio_file):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

  y, sr = librosa.load(audio_file)
  ms = librosa.feature.melspectrogram(y=y, sr=sr)
  log_ms = librosa.power_to_db(ms, ref=np.max)
  librosa.display.specshow(log_ms, sr=sr)

  image_file = audio_file.replace('.wav', '.png')
  fig.savefig(image_file)
  plt.close(fig)

  return image_file

def SpectrogramClassifier(spectrogram_image_file):
  X = []

  image = plt.imread(spectrogram_image_file)
  X.append(image)
  
  X = np.array(X)
  X = X.reshape(X.shape[0], -1)

  ocsvm_model = pickle.load(open(app.config["OCSVM_MODEL_PATH"],'rb'))

  y_pred = ocsvm_model.predict(X)
  result = ["Bad" if i==-1 else "Good" for i in y_pred]

  return result[0]

if __name__ == "__main__":
  app.run(debug=True)