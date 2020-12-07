import warnings
warnings.filterwarnings("ignore")
import os
import keras
import cv2, spacy, numpy as np
from keras.models import model_from_json,load_model
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')

VQA_model_file_name      = 'VQA_MODEL.json'
VQA_weights_file_name   = 'VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'FULL_labelencoder_trainval.pkl_01.npy'
model_new = 'model_new.hdf5'

def get_image_features(im,model_vgg):
    image_features = np.zeros((1, 4096))
    image_features[0,:] = model_vgg.predict(im)[0]
    return image_features

def get_question_features(question):
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

import cv2
from flask import Flask
from flask import request   
from flask import jsonify
from flask import render_template

app = Flask(__name__,template_folder='template')  
@app.route('/')
def index():
    return render_template('UI.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':

        file = request.files['image']
        filename = file.filename  
        print(filename)
        filepath = os.path.join("static/images/", filename)
        file.save(filepath)
        #print(filepath)
        K.clear_session()
        model_vgg = load_model(model_new)
        im = cv2.imread(filepath)
        #print(im)
        im = cv2.resize(im,(224,224))
        #print(im)
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        image_features = get_image_features(im,model_vgg)
        #print(image_features.shape) 
  
        question = request.form['text']
        question_features = get_question_features(question)
        
        model_vqa = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
        y_output = model_vqa.predict([question_features, image_features])
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        m = np.load(label_encoder_file_name)
        labels_list=[]
        for label in reversed(np.argsort(y_output)[0,-5:]):

            response = str(round(y_output[0,label]*100,2)).zfill(5)+ "% ," + m[label]
            labels_list.append(response)
   
            
    return render_template('predict.html', labels=labels_list,filepath=filepath,question=question)

if __name__=='__main__':

   app.run(debug=True)