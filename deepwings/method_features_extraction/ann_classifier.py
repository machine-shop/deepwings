import csv
import pickle
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
# from keras import regularizers
from keras import optimizers
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.externals import joblib


def encod_csv(classes, category='species',
              path_valid='training/valid.csv',
              path_valid_encoded='training/valid_encoded_species.csv'):

    dataset = pd.read_csv(path_valid)

    data = dataset.iloc[:, :].values

    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data[:, 1:] = imputer.fit_transform(data[:, 1:])
    names = data[:, 0]

    embeddings = []
    for name in names:
        genus = name.split(' ')[1].lower()
        if category == 'genus':
            category_name = genus
        elif category == 'species':
            species = name.split(' ')[2].lower()
            category_name = f'{genus}_{species}'
        idx = classes.index(category_name)
        embedding = [0] * len(classes)
        embedding[idx] = 1
        embeddings.append(embedding)

    with open(path_valid_encoded, 'w') as new_csv_file:
        writer = csv.writer(new_csv_file)
        first_row = list(dataset) + classes
        writer.writerow(first_row)
        for row, embedding in zip(data, embeddings):
            new_row = list(row) + embedding
            writer.writerow(new_row)
    print(f'Valid data encoded to {path_valid_encoded}')


def ANN_classifier(input_dim, output_dim):
    print(f'input_dim : {input_dim}  output_dim: {output_dim}')
    classifier = Sequential()
    classifier.add(Dense(output_dim=64, init='uniform',
                         activation='relu', input_dim=input_dim))
    classifier.add(Dropout(p=0.7))  # avoiding overfitting
    classifier.add(Dense(output_dim=output_dim, init='uniform',
                         activation='softmax'))

    return classifier


def train(category, n_epochs=600):

    dict_info = pickle.load(open('training/info_train_test.p', 'rb'))
    classes = dict_info['classes']
    path_valid_encoded = f'training/valid_encoded_{category}.csv'

    encod_csv(classes=classes, category=category,
              path_valid_encoded=path_valid_encoded)

    dataset = pd.read_csv(path_valid_encoded)
    n_classes = len(classes)
    data = dataset.iloc[:, :].values

    # mask_train = [name in dict_info['train'] for name in data[:, 0]]
    mask_train = []
    for name in data[:, 0]:
        if name in dict_info['train']:
            mask_train.append(1)
        elif name in dict_info['test']:
            mask_train.append(0)
        else:
            print(f'Not found : {name}')
    mask_train = np.array(mask_train, dtype=bool)

    data_train = data[mask_train]
    data_test = data[~mask_train]

    X_train = data_train[:, 1:-n_classes]
    print(X_train[:10])
    Y_train = data_train[:, -n_classes:]

    X_test = data_test[:, 1:-n_classes]
    Y_test = data_test[:, -n_classes:]

    # Feature scaling : calculations
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = ANN_classifier(input_dim=X_train.shape[1],
                                output_dim=Y_train.shape[1])

    # compiling the ANN

    sgd = optimizers.SGD(lr=0.003, decay=1e-4, momentum=0.9, nesterov=True)
    classifier.compile(optimizer=sgd, loss='categorical_crossentropy',
                       metrics=['accuracy'])

    history = classifier.fit(X_train, Y_train, batch_size=40,
                             nb_epoch=n_epochs,
                             validation_data=(X_test, Y_test))
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_accuracy = round(np.mean(acc[-20:]), 4)
    test_accuracy = round(np.mean(val_acc[-20:]), 4)
    epochs = np.arange(n_epochs)
    print('test_size : ', dict_info['test_size'])
    print('Mean accuracies on last 20 epochs:')
    print('train_accuracy : ', train_accuracy)
    print('test_accuracy : ', test_accuracy)

    PREFIX = f'training/models/ann/{category}'
    os.makedirs(PREFIX, exist_ok=True)

    path_model = os.path.join(PREFIX, 'model.h5')
    path_fig = os.path.join(PREFIX, 'graph.png')
    path_scaler = os.path.join(PREFIX, 'scaler.save')
    path_classes = os.path.join(PREFIX, 'classes.p')

    classifier.save(path_model)
    print(f'Model saved to {path_model}')

    joblib.dump(scaler, path_scaler)
    print(f'Scaler saved to {path_scaler}')

    pickle.dump(dict_info['classes'], open(path_classes, 'wb'))
    print(f'Classes saved to {path_classes}')

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].plot(epochs, acc, label='train')
    ax[0].plot(epochs, val_acc, label='test')
    ax[0].legend(loc="lower right")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].plot(epochs, loss, label='train')
    ax[1].plot(epochs, val_loss, label='test')
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    fig.suptitle(f'{category}_ann.h5')

    fig.savefig(path_fig, dpi=fig.dpi)
    print(f'Figure saved to {path_fig}')


def predict(category='species', path_raw='prediction/raw_images',
            path_model=None, path_scaler=None,
            path_info='training/info_train_test.p',
            n_descriptors=15, nb_pred=3):

    path_raw = path_raw.rstrip(r'/ +')
    path_parent = os.path.join(*path_raw.split('/')[:-1])
    path_valid = os.path.join(path_parent, 'valid.csv')
    path_prediction = os.path.join(path_parent,
                                   f'prediction_ann_{category}.csv')
    path_classes = f'training/models/ann/{category}/classes.p'

    if not path_scaler:
        path_scaler = f'training/models/ann/{category}/scaler.save'
    if not path_model:
        path_model = f'training/models/ann/{category}/model.h5'

    if not os.path.exists(path_valid):
        print(f'ERROR: no CSV files found in {path_valid}, try extracting'
              ' features before predicting')
        return

    if not os.path.exists(path_scaler):
        print(f'{path_scaler} not found')
        print('Try training before predicting: python pipeline.py -t ann')
        return

    print(f'Prediction folder : {path_raw}')
    print(f'Model used : {path_model}')

    dataset = pd.read_csv(path_valid)
    data = dataset.iloc[:, :].values
    X = data[:, 1:]
    scaler = joblib.load(path_scaler)
    X = scaler.transform(X)

    classifier = load_model(path_model)

    y = classifier.predict(X)

    classes = pickle.load(open(path_classes, 'rb'))

    with open(path_prediction, 'w') as csv_file:
        writer = csv.writer(csv_file)

        # Header
        columns_pred = ['image_names']
        for i in range(1, nb_pred + 1):
            columns_pred += [f'prediction_{i}', f'score_{i}']
        writer.writerow(columns_pred)

        for i, pred in enumerate(y):
            row = [data[i, 0]]
            idx_pred = np.argsort(pred)[-nb_pred:][::-1]
            for i in idx_pred:
                row += [classes[i], round(pred[i], 4)]
            writer.writerow(row)

    print(f'Prediction successful, results in {path_prediction}')
