import os
import argparse
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import tensorflow_hub as hub
from PIL import Image
import json

IMAGE_SIZE = 224

def load_json_map(path):
    with open(path, 'r') as f:
        flower_map = json.load(f)
        
    return flower_map

class InvalidModelException(Exception):
    pass

class InvalidFileException(Exception):
    pass

def process_image(image_path):
    image = Image.open(image_path)
    numpy_image = np.asarray(image)
    numpy_image = tf.cast(numpy_image, tf.float32)
    numpy_image = tf.image.resize(numpy_image, (IMAGE_SIZE, IMAGE_SIZE))
    numpy_image /= IMAGE_SIZE
    return np.expand_dims(numpy_image, axis=0)

def predict(image, model, top_K):
    result = model.predict(image)
    probs, classes = tf.nn.top_k(result, top_K)
    return probs, classes

def main():
    parser = ArgumentParser()
    parser.add_argument("model", nargs=1)
    parser.add_argument("file_path", nargs=1)
    parser.add_argument('-k', '--top_k', default=1, nargs=1)
    parser.add_argument('-c', '--category_names', nargs=1)
    args = parser.parse_args()
    
    top_k = args.top_k[0]
    
    try:
        model_path = os.path.join(os.getcwd(), args.model[0])
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    except InvalidModelException as e:
        raise InvalidModelException(e)
    
    try:
        file_path = args.file_path[0]
        image = process_image(file_path)
    except InvalidFileException as e:
        raise InvalidFileException(e)
        
    probs, classes = predict(image, model, int(top_k))
    
    if not args.category_names:
        print(classes.numpy().tolist()[0])
    
    else:
        json_path = os.path.join(os.getcwd(), args.category_names[0])
        json_map = load_json_map(json_path)
        mapped_categories = map(
            lambda flower: json_map[str(int(flower) + 1)], classes.numpy().tolist()[0]
        )
        print(list(mapped_categories))
                                 
         
                                
                               
    
    

    
if __name__ == "__main__":
    main()