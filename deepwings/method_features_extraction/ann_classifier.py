import csv
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.externals import joblib

# from method_features_extraction import features_extractor as fe

# species_6cells = ['MAWIspB', 'acuminatum', 'bimaculata', 'calcarata',
# 'coloradensis', 'seocolis', 'impatiens', 'leucozonium',
# 'lignaria', 'nymphaerum', 'pilosum', 'pusilla',
# 'ribifloris', 'rohweri', 'texana', 'texanus', 'virescens',
# 'zephyrum', 'zonulum']

# species_7cells = ['MAWIspB', 'acuminatum', 'bimaculata', 'calcarata',
# 'coloradensis', 'coriaceum', 'griseocolis', 'impatiens',
# 'leucozonium', 'nymphareum', 'lignaria', 'nymphaerum',
# 'pilosum', 'pusilla', 'ribifloris', 'rohweri', 'sericeus',
# 'texana', 'texanus', 'virescens', 'zephyrum', 'zonulum']
#  MAWIspB acuminatum bimaculata calcarata coloradensis	coriaceum griseocolis
# impatiens leucozonium lignaria nymphaerum pilosum pusilla ribifloris rohweri
# sericeus texana	texanus	virescens zephyrum zonulum


def create_csv(parent_folder, category):
    csv_name = os.path.join(parent_folder, 'valid.csv')
    dataset = pd.read_csv(csv_name)
    # shuffle rows
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    data = dataset.iloc[:, :].values

    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data[:, 1:] = imputer.fit_transform(data[:, 1:])
    names = data[:, 0]

    category_list = []

    for name in names:
        if category == 'genus':
            category_name = name.split(' ')[1].lower()
        elif category == 'species':
            category_name = (name.split(' ')[1] + '_' +
                             name.split(' ')[2]).lower()
        if category_name not in category_list:
            category_list.append(category_name)
    category_list = sorted(category_list)

    new_csv_name = os.path.join(parent_folder, f'valid_encoded_{category}.csv')

    with open(new_csv_name, 'w') as new_csv_file:
        writer = csv.writer(new_csv_file)
        first_row = list(dataset) + category_list
        writer.writerow(first_row)
        for row in data:
            zeros = np.zeros(len(category_list))
            if category == 'genus':
                category_name = row[0].split(' ')[1].lower()
            elif category == 'species':
                category_name = (row[0].split(' ')[1] + '_' +
                                 row[0].split(' ')[2]).lower()

            idx_category = category_list.index(category_name)
            zeros[idx_category] = 1
            output = list(row) + list(zeros)
            writer.writerow(output)
    print(new_csv_name)
    return new_csv_name, len(category_list)


def ANN_classifier(input_dim, output_dim):
    print(f'input_dim : {input_dim}  output_dim: {output_dim}')
    classifier = Sequential()
    classifier.add(Dense(output_dim=input_dim, init='uniform',
                         activation='relu', input_dim=input_dim))
    classifier.add(Dropout(p=0.7))  # avoiding overfitting
    classifier.add(Dense(output_dim=output_dim, init='uniform',
                         activation='softmax'))

    return classifier


def train(category, test_size=0.2, n_epochs=300):
    if not 0 < test_size < 1:
        print('ERROR : test_size must be in ]0, 1[')
        print('To specify test_size : $ python pipeline -t ann -ts 0.3')
        return

    csv_name, nb_categories = create_csv('training', category)
    print(csv_name)
    dataset = pd.read_csv(csv_name)
    data = dataset.iloc[:, :].values
    X = data[:, 1:-nb_categories]
    Y = data[:, -nb_categories:]

    # Feature scaling : calculations
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaler_name = 'training/scaler.save'
    joblib.dump(scaler, scaler_name)

    classifier = ANN_classifier(input_dim=X.shape[1],
                                output_dim=Y.shape[1])

    # compiling the ANN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy'])

    history = classifier.fit(X, Y, batch_size=40, nb_epoch=n_epochs,
                             validation_split=test_size, shuffle=True)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_accuracy = round(np.mean(acc[-20:]), 4)
    test_accuracy = round(np.mean(val_acc[-20:]), 4)
    epochs = np.arange(n_epochs)
    print('test_size : ', test_size)
    print('Mean accuracies on last 20 epochs:')
    print('train_accuracy : ', train_accuracy)
    print('test_accuracy : ', test_accuracy)

    prefix = 'deepwings/method_features_extraction/models/'
    path_model = os.path.join(prefix, f'{category}_ann.h5')
    classifier.save(path_model)
    print(f'Model saved to {path_model}')

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].plot(epochs, acc, label='train')
    ax[0].plot(epochs, val_acc, label='test')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].plot(epochs, loss, label='train')
    ax[1].plot(epochs, val_loss, label='test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    fig.suptitle(f'{category}_ann.h5')

    path_fig = os.path.join(prefix, f'{category}_ann.png')
    fig.savefig(path_fig, dpi=fig.dpi)
    print(f'Figure saved to {path_fig}')


def predict(category, plot, n_descriptors, nb_pred=3):
    exist_csv_valid = os.path.exists('prediction/valid.csv')
    exist_csv_invalid = os.path.exists('prediction/invalid.csv')

    if not (exist_csv_valid and exist_csv_invalid):
        print('ERROR: no CSV files found, try extracting features before\
              predicting')
        return 1

    csv_valid_pred = 'prediction/valid.csv'
    path_scaler = 'training/scaler.save'
    path_encoded_csv = f'training/valid_encoded_{category}.csv'
    path_prediction = 'prediction/prediction_ann.csv'

    if not os.path.exists(path_scaler):
        print(f'{path_scaler} not found')
        print('Try training before predicting: python pipeline.py -t ann')
        return 1

    # if not os.path.exists(processed_csv):
        # print(processed_csv + ' not found')
        # print('Try training before predicting: python pipeline.py -t ann')
        # break

    if not os.path.exists(csv_valid_pred):
        print(f'{csv_valid_pred} not found')
        return 1

    dataset = pd.read_csv(csv_valid_pred)
    data = dataset.iloc[:, :].values
    X = data[:, 1:]
    scaler = joblib.load(path_scaler)
    X = scaler.transform(X)

    prefix = 'deepwings/method_features_extraction/models/'
    path_model = os.path.join(prefix, f'{category}_ann.h5')
    classifier = load_model(path_model)

    y = classifier.predict(X)

    # Finding the names of the classes
    encoded_csv = pd.read_csv(path_encoded_csv)
    columns = encoded_csv.columns
    cells_names = ['marg', '1st_med', '2nd_med', '2nd_cub',
                   '1st_sub', '2nd_sub', '3rd_sub']
    category_names = []
    for i, header in enumerate(columns[1:]):
        last_feature = True
        for name in cells_names:
            if name in header:
                last_feature = False
        if last_feature:
            category_names = columns[i+1:]
            break

    with open(path_prediction, 'w') as csv_file:
        writer = csv.writer(csv_file)

        # Header
        columns_pred = ['image_names']
        for i in range(1, nb_pred + 1):
            columns_pred += [f'prediction_{i}', f'score_{i}']
        writer.writerow(columns_pred)

        for i, pred in enumerate(y):
            row = [data[i, 0]]
            for i in range(nb_pred):
                idx_max = np.argmax(pred)
                row += [category_names[idx_max], round(pred[idx_max], 4)]
                pred[idx_max] = 0
            writer.writerow(row)

    print('Prediction successful, results in '
          'prediction/prediction_ann.csv')
