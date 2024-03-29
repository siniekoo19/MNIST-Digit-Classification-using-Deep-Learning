import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO


st.title("Digit Prediction Webapp :smiley:")


# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the uploaded file as a byte stream
    bytes_data = uploaded_file.read()  # bytes_data is the byte representation of the image

    # Display the uploaded image
    st.image(bytes_data, width=250)



if st.button("Predict"):

    # Load your model
    model = tf.keras.models.load_model('my_model.h5')

    # Convert bytes_data to numerical data
    image = Image.open(BytesIO(bytes_data))

    # Preprocess the image (e.g., resize, convert to grayscale, etc.)
    image = image.resize((28, 28))  # Resize to the input size of your model
    image = np.array(image) 
    image = image[:,:,0] # Convert image to numpy array
    

    # Normalize the data
    normalized_data = tf.keras.utils.normalize(image)
    normalized_data = normalized_data.reshape(1, 28, 28, 1)  # Add batch dimension

    # Make prediction
    pred = model.predict(normalized_data)

    # Display the image
    # st.image(image, width=600)

    # Display prediction
    st.markdown(f"## :blue[This digit is probably : ] <span style='font-size:36px'>{np.argmax(pred)}</span>", unsafe_allow_html=True)
