import os
from keras.models import load_model
from keras import backend as K
import skimage.transform as sk_tf
import skimage.io as sk_io
import numpy as np
import pickle
import csv


def cnn_pred(model_name='DenseNet121',
             path_raw='prediction/raw_images',
             input_shape=(224, 224, 3), nb_pred=3):

    path_raw = path_raw.rstrip(r'/ +')
    path_parent = os.path.join(*path_raw.split('/')[:-1])
    path_prediction = os.path.join(path_parent, 'prediction_cnn.csv')
    path_dict_pred = os.path.join(path_parent, 'prediction_cnn.p')

    path_folder = os.path.join('training/models/cnn', model_name)
    path_classes = os.path.join(path_folder, 'classes.p')
    path_model = os.path.join(path_folder,  'model.h5')

    if not os.path.exists(path_raw):
        print(f'ERROR : folder not found: {path_raw}')
        return

    print(f'Prediction folder : {path_raw}')
    print(f'Loading model: {path_model}')
    K.clear_session()
    model = load_model(path_model)
    classes = pickle.load(open(path_classes, 'rb'))

    images = sorted(os.listdir(path_raw))
    n_images = len(images)

    predictions = []
    dict_pred = {}
    for i, image_name in enumerate(images):
        print(f'Image {i + 1}/{n_images} : {image_name}')
        image_path = os.path.join(path_raw, image_name)
        image_rgb = sk_io.imread(image_path)
        resized = sk_tf.resize(image_rgb, input_shape)
        dim_4 = np.expand_dims(resized, axis=0)
        prediction = model.predict(dim_4)
        predictions.append(prediction[0])
        dict_pred[image_name] = prediction[0]

    with open(path_prediction, 'w') as csv_file:
        writer = csv.writer(csv_file)
        columns = ['image_names']
        for i in range(1, nb_pred + 1):
            columns += ['prediction_' + str(i), 'score_' + str(i)]
        writer.writerow(columns)

        for i, pred in enumerate(predictions):
            row = [images[i]]
            idx_pred = np.argsort(pred)[-nb_pred:][::-1]
            for i in idx_pred:
                row += [classes[i], round(pred[i], 4)]
            writer.writerow(row)
    print(f'Prediction successful, results in {path_prediction}')
    pickle.dump(dict_pred, open(path_dict_pred, 'wb'))
    print(f'(dictionary saved to {path_dict_pred})')
