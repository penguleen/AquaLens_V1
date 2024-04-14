import cv2
import numpy as np
import tensorflow as tf

# Use weights from the trained classification model, for first process of fish labels.
def classification(tested_fish, image_cat):
    # height and width of the image
    height, width = 135, 240

    # Import trained classification model
    reconstructed_model = tf.keras.models.load_model("./keras/__5class_custom_model_30.keras")
    class_predictions_list = []

    # From the converted images, resize them and classify each fish labels to each image.
    for item in tested_fish:
        sample_image = cv2.imread(str(item))
        sample_image_resized = cv2.resize(sample_image, (height, width))
        sample_image = np.expand_dims(sample_image_resized, axis=0)

        # Assign fish name from classification model.
        model_pred = reconstructed_model.predict(sample_image)

        predicted_class=image_cat[np.argmax(model_pred)]
        print("The predicted category is", predicted_class)

        class_predictions_list.append(predicted_class)

    # Store predicted fish names into 'class.txt' file outside the loop
    with open("class.txt", "w") as classification_file:
        for predicted_class in class_predictions_list:
            class_name = str(predicted_class) + '\n'  # only 1 class: 0:fish, 5 variables
            classification_file.write(class_name)
