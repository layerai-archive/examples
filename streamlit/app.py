import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import layer
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

fig = plt.figure()

st.header("Predict the type of food")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(image)
        st.write(predictions)
        st.pyplot(fig)


def predict(image):
    model = layer.get_model('layer/image-classification/models/food-vision').get_train()
    test_image = image.resize((300, 300))
    test_image = img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['chicken_curry', 'oysters', 'tuna_tartare', 'pho', 'fried_rice', 'hot_and_sour_soup',
                   'seaweed_salad', 'baklava', 'eggs_benedict', 'panna_cotta', 'onion_rings', 'lasagna', 'foie_gras',
                   'churros', 'donuts', 'spring_rolls', 'gyoza', 'ice_cream', 'dumplings', 'ceviche''ramen', 'nachos',
                   'greek_salad', 'scallops', 'chocolate_mousse', 'grilled_cheese_sandwich', 'cheesecake', 'steak',
                   'hummus', 'bread_pudding', 'frozen_yogurt', 'falafel', 'paella', 'pulled_pork_sandwich', 'bibimbap',
                   'risotto', 'macarons', 'garlic_bread', 'beef_carpaccio', 'red_velvet_cake', 'ravioli', 'waffles',
                   'grilled_salmon', 'tacos', 'lobster_bisque', 'sushi', 'clam_chowder', 'sashimi', 'french_onion_soup',
                   'french_fries', 'tiramisu', 'takoyaki', 'chicken_quesadilla', 'chicken_wings', 'pizza', 'pork_chop',
                   'crab_cakes', 'cannoli', 'beignets', 'miso_soup', 'mussels', 'strawberry_shortcake', 'caprese_salad',
                   'gnocchi', 'deviled_eggs', 'macaroni_and_cheese', 'fish_and_chips', 'beef_tartare', 'guacamole',
                   'hamburger', 'club_sandwich', 'edamame', 'cheese_plate', 'peking_duck', 'fried_calamari',
                   'prime_rib', 'caesar_salad', 'beet_salad', 'lobster_roll_sandwich', 'pancakes', 'samosa',
                   'french_toast', 'omelette', 'croque_madame', 'creme_brulee', 'filet_mignon', 'poutine', 'apple_pie',
                   'spaghetti_bolognese', 'bruschetta', 'cup_cakes', 'pad_thai', 'huevos_rancheros', 'baby_back_ribs',
                   'chocolate_cake', 'carrot_cake', 'hot_dog', 'spaghetti_carbonara', 'breakfast_burrito',
                   'shrimp_and_grits', 'escargots']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
        'Tomato Healthy': 0,
        'Tomato Septoria Leaf Spot': 0,
        'Tomato Bacterial Spot': 0,
        'Tomato Blight': 0,
        'Cabbage Healthy': 0,
        'Tomato Spider Mite': 0,
        'Tomato Leaf Mold': 0,
        'Tomato_Yellow Leaf Curl Virus': 0,
        'Soy_Frogeye_Leaf_Spot': 0,
        'Soy_Downy_Mildew': 0,
        'Maize_Ravi_Corn_Rust': 0,
        'Maize_Healthy': 0,
        'Maize_Grey_Leaf_Spot': 0,
        'Maize_Lethal_Necrosis': 0,
        'Soy_Healthy': 0,
        'Cabbage Black Rot': 0
    }
    result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} percent confidence."
    return result


if __name__ == "__main__":
    main()
