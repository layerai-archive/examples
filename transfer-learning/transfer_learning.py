import layer
from layer.decorators import model, pip_requirements, fabric

layer.login("https://development.layer.co/")
layer.init("transfer-learning")


@pip_requirements(packages=["wget", "tensorflow", "keras"])
@fabric("f-medium")
@model("xception")
def train():
    import tensorflow as tf
    import wget
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import zipfile
    wget.download("https://namespace.co.ke/ml/train.zip")
    with zipfile.ZipFile('train.zip', 'r') as zip_ref:
        zip_ref.extractall('train')
    base_dir = 'train/train'
    filenames = os.listdir(base_dir)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append("dog")
        else:
            categories.append("cat")

    df = pd.DataFrame({'filename': filenames, 'category': categories})
    train_datagen = ImageDataGenerator(validation_split=0.2)
    validation_gen = ImageDataGenerator(validation_split=0.2)
    image_size = (150, 150)
    training_set = train_datagen.flow_from_dataframe(df, base_dir,
                                                     seed=101,
                                                     target_size=image_size,
                                                     batch_size=32,
                                                     x_col='filename',
                                                     y_col='category',
                                                     subset="training",
                                                     class_mode='binary')
    validation_set = validation_gen.flow_from_dataframe(df, base_dir,
                                                        target_size=image_size,
                                                        batch_size=32,
                                                        x_col='filename',
                                                        y_col='category',
                                                        subset="validation",
                                                        class_mode='binary')
    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape=(150, 150, 3))
    data_augmentation = keras.Sequential([keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1)])
    x = data_augmentation(inputs)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=keras.metrics.BinaryAccuracy())
    model.fit(training_set, epochs=3, validation_data=validation_set)
    base_model.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=keras.metrics.BinaryAccuracy())
    callbacks = [EarlyStopping(patience=5)]
    history = model.fit(training_set, epochs=3, validation_data=validation_set, callbacks=callbacks)
    metrics_df = pd.DataFrame(history.history)
    layer.log({"Metrics DataFrame": metrics_df})
    loss, accuracy = model.evaluate(validation_set)
    print('Accuracy on test dataset:', accuracy)
    metrics_df[["loss", "val_loss"]].plot()
    layer.log({"Loss plot": plt.gcf()})
    metrics_df[["binary_accuracy", "val_binary_accuracy"]].plot()
    layer.log({"Accuracy plot": plt.gcf()})
    return model


train()

# Run the project on Layer Infra
# layer.run([train])


