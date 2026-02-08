import tensorflow as tf
import numpy as np

LABEL_MAP = {0:'IM', 1:'NP', 2:'PR'}

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_clip(model, clip):
    x = np.expand_dims(clip, axis=0)
    prob = model.predict(x, verbose=0)[0]
    label_idx = np.argmax(prob)
    return LABEL_MAP[label_idx], prob