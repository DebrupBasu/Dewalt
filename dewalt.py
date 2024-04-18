import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the JSON file containing product information
with open('C:\\Users\\Debrup Basu\\Downloads\\dewalt.json') as f:
    data = json.load(f)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title('Dewalt Power Tool Image Recognition')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the uploaded image
    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    # Get the feature vector for the uploaded image
    features = model.predict(img_array)
    # Find the most similar product
    max_similarity = 0
    most_similar_product = None
    for dictionary in data:
        for product in dictionary["PowerTools"]:
            for product_image_path in product["ImagePaths"]:
             try:
                    
                   product_image = Image.open(product_image_path)
                   product_image = product_image.resize((128, 128))
                   product_image_array = np.expand_dims(np.array(product_image), axis=0)
                   product_image_array = preprocess_input(product_image_array)
                   product_features = model.predict(product_image_array)
                   similarity = cosine_similarity(features, product_features)[0][0]
                   if similarity > max_similarity:
                     max_similarity = similarity
                     most_similar_product = product
             except UnidentifiedImageError:
                   pass
    # Display the most similar product information
    st.write('Most Similar Product:')
    st.write('Name:', most_similar_product["Name"])
    st.write('Features:', most_similar_product["Features"])
    st.write('Applications:', most_similar_product["Applications"])