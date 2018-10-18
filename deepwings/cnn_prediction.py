#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
import numpy as np
import pandas as pd
import csv

names_species = ['lasioglossum_acuminatum', #0
                     'bombus_bimaculata', #1
                     'ceratina_calcarata', #2
                     'osmia_coloradensis', #3
                     'lasioglossum_coriaceum', #4
                     'bombus_griseocolis', #5
                     'bombus_impatiens', #6
                         'lasioglossum_leucozonium', #7
                     'osmia_lignaria', #8
                     'lasioglossum_mawispb', #9
                     'lasioglossum_nymphaerum',#10
                     'lasioglossum_pilosum', #11
                     'osmia_pusilla', #12
                     'osmia_ribifloris', #13
                     'lasioglossum_rohweri', #14
                     'agapostemon_sericeus', #15
                     'osmia_texana', #16
                     'agapostemon_texanus', #17
                     'agapostemon_virescens', #18
                     'lasioglossum_zephyrum',
                     'lasioglossum_zonolum']

def cnn_pred(category='species', nb_pred=3, path_pred='prediction/raw_images'):
    
    if not os.path.exists(path_pred):
        print('ERROR : folder not found: ' + path_pred)
        return 0
    
    images = sorted(os.listdir(path_pred+'/test/'))
    number_images = len(images)
    
    if category=='species':
        model = load_model('method_cnn/models/VGG16_2nd_method_dataug_110epft20ep.h5')

    pred_datagen = ImageDataGenerator(rescale=1./255)
    pred_generator = pred_datagen.flow_from_directory(path_pred,
                                                   target_size=(150, 150),
                                                   batch_size=number_images,
                                                   class_mode=None, 
                                                   shuffle=False)
    predictions = model.predict_generator(pred_generator)
    
    if os.path.exists('prediction/prediction_cnn.csv'):
        os.remove('prediction/prediction_cnn.csv')
    
    with open('prediction/prediction_cnn.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        
        columns = ['image_names'] 
        for i in range(1, nb_pred +1):
            columns += ['prediction_' + str(i), 'score_' + str(i)]
        writer.writerow(columns)
        
        for i, pred in enumerate(predictions):
            row = [images[i]]
            for i in range(nb_pred):
                idx_max = np.argmax(pred)
                row += [names_species[idx_max], round(pred[idx_max], 4)]
                pred[idx_max] = 0
            writer.writerow(row)
    print("Prediction successful, results in 'deepwings/prediction/prediction_cnn.csv' ")

    
    
   
    
            
        
    
        
        
        