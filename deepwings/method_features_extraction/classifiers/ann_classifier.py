import csv
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import numpy as np
import os
import pandas as pd
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


def create_csv(folder_path, category, nb_cells):
    csv_name = folder_path + 'data_' + str(nb_cells) + 'cells.csv'
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

    new_csv_name = folder_path + category + '_' + csv_name.split('/')[-1]

    if os.path.exists(new_csv_name):
        os.remove(new_csv_name)

    with open(new_csv_name, 'a') as new_csv_file:
        writer = csv.writer(new_csv_file)
        first_row = list(dataset) + category_list
        writer.writerow(first_row)
        for row in data:
            zeros = np.zeros((len(category_list,)))
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
    classifier.add(Dropout(p=0.55))  # avoiding overfitting
    classifier.add(Dense(output_dim=output_dim, init='uniform',
                         activation='softmax'))

    return classifier


def train(category):
    train_accuracies = []
    test_accuracies = []
    for nb_cells in [6, 7]:
        csv_name, nb_categories = create_csv('training/', category, nb_cells)
        print(csv_name)
        dataset = pd.read_csv(csv_name)
        data = dataset.iloc[:, :].values
        X = data[:, 1:-nb_categories]
        # remove last column for Dummy variables trap
        Y = data[:, -nb_categories:]

        # Splitting the data
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
        # test_size=0.2, random_state=0)

        # Feature scaling : calculations
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        scaler_name = 'training/scaler_'+str(nb_cells) + 'cells.save'
        joblib.dump(scaler, scaler_name)

        classifier = ANN_classifier(input_dim=X.shape[1],
                                    output_dim=Y.shape[1])

        # compiling the ANN
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

        history = classifier.fit(X, Y, batch_size=40, nb_epoch=300,
                                 validation_split=0.2, shuffle=True)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        train_accuracies.append(round(np.mean(acc[-20:]), 4))
        test_accuracies.append(round(np.mean(val_acc[-20:]), 4))

        prefix = 'method_features_extraction/classifiers/models_ann/'
        classifier.save(prefix + category + '_' + str(nb_cells) +
                        'cells_ann_model.h5')

    print('train_accuracy_6cells : ', train_accuracies[0])
    print('test_accuracy_6cells : ', test_accuracies[0])
    print('train_accuracy_7cells : ', train_accuracies[1])
    print('test_accuracy_7cells : ', test_accuracies[1])


def predict(category, plot, n_descriptors, nb_pred=3):
    bool_6cells = os.path.exists('prediction/data_6cells.csv')
    bool_7cells = os.path.exists('prediction/data_7cells.csv')
    bool_inv = os.path.exists('prediction/invalid.csv')

    if not (bool_6cells or bool_7cells or bool_inv):
        print('ERROR: no CSV files found, try extracting features before\
              predicting')

        return 0

    if os.path.exists('prediction/prediction_ann.csv'):
        os.remove('prediction/prediction_ann.csv')

    for nb_cells in [6, 7]:
        csv_name = 'prediction/data_' + str(nb_cells) + 'cells.csv'
        scaler_name = 'training/scaler_'+str(nb_cells) + 'cells.save'
        processed_csv = ('training/' + category + '_data_' + str(nb_cells) +
                         'cells.csv')
        if not os.path.exists(scaler_name):
            print(scaler_name + ' not found')
            print('Try training before predicting: python pipeline.py -t ann')
            break

        if not os.path.exists(processed_csv):
            print(processed_csv + ' not found')
            print('Try training before predicting: python pipeline.py -t ann')
            break

        if not os.path.exists(csv_name):
            print(csv_name + ' not found')
            continue

        dataset = pd.read_csv(csv_name)
        data = dataset.iloc[:, :].values
        X = data[:, 1:]
        scaler = joblib.load(scaler_name)
        X = scaler.transform(X)
        prefix = 'method_features_extraction/classifiers/models_ann/'
        classifier = load_model(prefix + category + '_' + str(nb_cells) +
                                'cells_ann_model.h5')
        y = classifier.predict(X)
        processed_dataset = pd.read_csv(processed_csv)
        columns = processed_dataset.columns
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

        with open('prediction/prediction_ann.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            if nb_cells == 6:
                columns_pred = ['image_names']
                for i in range(1, nb_pred + 1):
                    columns_pred += ['prediction_' + str(i), 'score_' + str(i)]
                writer.writerow(columns_pred)

            for i, pred in enumerate(y):
                row = [data[i, 0]]
                for i in range(nb_pred):
                    idx_max = np.argmax(pred)
                    row += [category_names[idx_max], round(pred[idx_max], 4)]
                    pred[idx_max] = 0
                writer.writerow(row)

    print('Prediction successful, results in '
          'deepwings/prediction/prediction_ann.csv')
