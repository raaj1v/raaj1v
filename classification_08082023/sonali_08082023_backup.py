import csv
import os
from datetime import datetime
from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from preprocessing import *


app = FastAPI()

model=load_model(r"model/cleandata_basic_30L_2.h5")
multilabel_binarizer = joblib.load(r"classes/cleandata_Updated_mlb_All_Product_2.pkl")
labels = multilabel_binarizer.classes_
tokenizer = joblib.load(r"tokenizer/cleandata_Updated_tokenizer_30L_2.pkl")
full_form_data=pd.read_csv(r"shortCodesProduct.csv")
full_form_data=full_form_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)



def log_predictions(request: Request, input_sentence, predictions):
    log_file = "prediction_log.csv"
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Date Time", "IP Address", "Input Sentence", "Predicted Labels"])

        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip_address = request.client.host

        writer.writerow([date_time, ip_address, input_sentence, predictions])

def predict_op_th_main(new_sentence, threshold):
    new_sentence=new_sentence.lower()
    new_sentence=replace_short_with_full(new_sentence, full_form_data)
    new_sentence = cleanText(new_sentence)
    new_sentence = preprocess_text(new_sentence)
    new_sentence_sequences = tokenizer.texts_to_sequences([new_sentence])
    new_sentence_padded = pad_sequences(new_sentence_sequences, maxlen=512)
    predicted_domain = model.predict(new_sentence_padded)
    result = [{label,  float(prob)} for label, prob in zip(labels, predicted_domain[0]) if prob >= threshold]
    print(type(result))
    return result

def Predict_Labels(new_sentence):
    threshold = 0.3
    output = predict_op_th_main(new_sentence, threshold)
    if not output:
        threshold = 0.1
        output = predict_op_th_main(new_sentence, threshold)
    return output

class InputData(BaseModel):
    sentence: str

@app.post("/prediction3")
def predict_sentence(request: Request, data: InputData):
    output = Predict_Labels(data.sentence)
    log_predictions(request, data.sentence, output)
    return output

