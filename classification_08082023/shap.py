# sunil.k@eptlltd.onmicrosoft.com
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
import shap

app = FastAPI()

model = load_model(r"model/cleandata_basic_30L_2.h5")
multilabel_binarizer = joblib.load(r"classes/cleandata_Updated_mlb_All_Product_2.pkl")
labels = multilabel_binarizer.classes_
tokenizer = joblib.load(r"tokenizer/cleandata_Updated_tokenizer_30L_2.pkl")
full_form_data = pd.read_csv(r"shortCodesProduct.csv")
full_form_data = full_form_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def predict_op_th_main(new_sentence, threshold):
    new_sentence = new_sentence.lower()
    new_sentence = replace_short_with_full(new_sentence, full_form_data)
    new_sentence = cleanText(new_sentence)
    new_sentence = preprocess_text(new_sentence)
    new_sentence_sequences = tokenizer.texts_to_sequences([new_sentence])
    new_sentence_padded = pad_sequences(new_sentence_sequences, maxlen=512)
    predicted_domain = model.predict(new_sentence_padded)
    result = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(labels, predicted_domain[0])
        if prob >= threshold
    ]
    return result

@app.get("/predict/")
async def predict_text(request: Request, sentence: str, threshold: float = 0.1):
    result = predict_op_th_main(sentence, threshold)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, tokenizer)
    
    # Preprocess the sentence for SHAP interpretation
    preprocessed_sentence = preprocess_text(sentence)
    input_data = tokenizer.texts_to_sequences([preprocessed_sentence])
    input_data_padded = pad_sequences(input_data, maxlen=512)
    
    # Calculate SHAP values
    shap_values = explainer(input_data_padded)
    
    # Summarize SHAP values for each label
    shap_summary = {}
    for idx, label in enumerate(labels):
        shap_values_label = shap_values[0][idx]
        shap_summary[label] = shap_values_label.tolist()
    
    return {"predictions": result, "shap_summary": shap_summary}
