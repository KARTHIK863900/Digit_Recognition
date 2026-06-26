import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload Digit Image",type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    img=np.array(image)

    img=255-img

    img=img/255.0

    img=img.reshape(1,28,28,1)

    prediction=model.predict(img)

    digit=np.argmax(prediction)

    st.image(image,width=150)

    st.success(f"Predicted Digit : {digit}")
