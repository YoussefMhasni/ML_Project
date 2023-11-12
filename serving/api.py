import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
global model

# Loading the pickled model from a file
model=joblib.load(open("./artifacts/best_model.pkl", 'rb'))
app = FastAPI()

class ImageData(BaseModel):
    data: list
class Feedback(BaseModel):
    data : list
    predicted_value: int
    real_value: int
class Result(BaseModel):
    prediction:int

@app.post("/feedback")
async def feedback(fb: Feedback):
    try:
        file_path = os.path.join(os.getcwd(), 'data', 'prod_data.csv')
        df = pd.read_csv(file_path)
        data = np.append(fb.data, [fb.real_value, fb.predicted_value])
        if len(data) == len(df.columns):
            new_row = pd.DataFrame([data], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(str(file_path), index=False)
        else:
            print("Number of elements in 'data' does not match the number of columns in the DataFrame.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/")
async def upload_image(image: ImageData):
    try:
        img_array = image.data
        data=np.array(img_array).astype('uint8')
        prediction = model.predict(data)
        result = prediction[0]
        return Result(prediction=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
