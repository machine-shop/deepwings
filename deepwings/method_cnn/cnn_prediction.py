import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import skimage.transform as sk_tf
import skimage.io as sk_io
import numpy as np
import csv


names_species = ['lasioglossum_acuminatum',  # 0
                 'bombus_bimaculata',  # 1
                 'ceratina_calcarata',  # 2
                 'osmia_coloradensis',  # 3
                 'lasioglossum_coriaceum',  # 4
                 'bombus_griseocolis',  # 5
                 'bombus_impatiens',  # 6
                 'lasioglossum_leucozonium',  # 7
                 'osmia_lignaria',  # 8
                 'lasioglossum_mawispb',  # 9
                 'lasioglossum_nymphaerum',  # 10
                 'lasioglossum_pilosum',  # 11
                 'osmia_pusilla',  # 12
                 'osmia_ribifloris',  # 13
                 'lasioglossum_rohweri',  # 14
                 'agapostemon_sericeus',  # 15
                 'osmia_texana',  # 16
                 'agapostemon_texanus',  # 17
                 'agapostemon_virescens',  # 18
                 'lasioglossum_zephyrum',  # 19
                 'lasioglossum_zonolum']  # 20


def cnn_pred(category='species', nb_pred=3, path_pred='prediction/raw_images'):

    if not os.path.exists(path_pred):
        print('ERROR : folder not found: ' + path_pred)
        return 0

    if category == 'species':
        path_model = 'method_cnn/models/VGG16_2nd_method_dataug_110epft20ep.h5'
        model = load_model(path_model)
    else:
        print('Only species classification is available in the CNN model')
        return 0

    images = sorted(os.listdir(path_pred))
    number_images = len(images)
    predictions = []
    for image_name in images:
        image_path = os.path.join(path_pred, image_name)
        image_rgb = sk_io.imread(image_path)
        resized = sk_tf.resize(image_rgb, (150, 150))
        dim_4 = np.expand_dims(resized, axis=0)
        prediction = model.predict(dim_4)
        predictions.append(prediction)

    # print(predictions)

    # pred_datagen = ImageDataGenerator(rescale=1./255)
    # pred_generator = pred_datagen.flow_from_directory(path_pred,
                                                      # target_size=(150, 150),
                                                      # batch_size=number_images,
                                                      # class_mode=None,
                                                      # shuffle=False)
    # predictions = model.predict_generator(pred_generator)

    with open('prediction/prediction_cnn.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        columns = ['image_names']
        for i in range(1, nb_pred + 1):
            columns += ['prediction_' + str(i), 'score_' + str(i)]
        writer.writerow(columns)

        for i, pred in enumerate(predictions):
            pred = pred[0]
            row = [images[i]]
            for i in range(nb_pred):
                idx_max = np.argmax(pred)
                row += [names_species[idx_max], round(pred[idx_max], 4)]
                pred[idx_max] = 0
            writer.writerow(row)
    print('Prediction successful, results in '
          'deepwings/prediction/prediction_cnn.csv')
