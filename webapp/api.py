import streamlit as st
import requests
from PIL import Image
import os
import sys
sys.path.append(str(os.getcwd()))
from scripts.functions import encoder

st.title("Anomaly Detection Web Application ")
st.write("Authors : Achour Simoud, Youssef M'hasni, Mohamed Amine Elkaout, Oussama Abdelmoula")
st.write(" ")
st.subheader("In this Anomaly Detection Project, we've utilized CIFAR-10 image data. The anomaly in this context is identified as instances belonging to the 'Airplane' class.")

# Display authors' names in the top-right corner

def webapp():
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image.save("image.png")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=False,width=300)
            img_array=encoder("image.png")
            response = requests.post(f"http://127.0.0.1:8000/predict/",json={"data": img_array.tolist()})
            result = response.json().get("prediction")
            st.write(f"Prediction: {result}")
            real_value=st.text_input(label="Enter 1 for True or 0 for False for the last prediction")
        if st.button('Feedback'):
            st.write(f"Feedback received: {real_value}")
            requests.post(f"http://127.0.0.1:8000/feedback/",json={"data": img_array.tolist(),"predicted_value": int(result), "real_value": int(real_value)})
            image_filename = "image.png"
            img_path = os.path.join(os.getcwd(), image_filename)
            if os.path.exists(img_path):
                os.remove(img_path)

if __name__ == "__main__":
    webapp()


