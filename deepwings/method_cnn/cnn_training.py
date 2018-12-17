from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import pickle
import datetime
import os


def build_model(base='DenseNet121', n_class=21,
                weights_conv='imagenet', input_shape=(224, 224, 3)):

    now = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    model_name = f'{base}_{now}'

    model = Sequential(name=model_name)

    if base == 'DenseNet121':

        conv = DenseNet121(input_shape=input_shape, include_top=False,
                           weights=weights_conv)

        model.add(conv)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(n_class, activation='softmax'))

    else:
        print('Only DenseNet121 is supported for CNN')
        return

    return model


def train_model(model, path_train='training/sorted_species/train/',
                path_test='training/sorted_species/test/',
                output='training/models/cnn/',
                bs_train=20, bs_test=20,
                epochs=20,
                steps_per_epoch=100):

    if not os.path.exists(output):
        print(f'ERROR : Could not find the folder {output}')
        return

    print('\nParameters :')
    print(f'path_train : {path_train}')
    print(f'path_test : {path_test}')
    print(f'epochs : {epochs}')
    print(f'Batch size train : {bs_train}')
    print(f'Batch size test : {bs_test}')
    print(f'steps per epoch : {steps_per_epoch}\n')
    path_folder = os.path.join(output, model.name)
    os.makedirs(path_folder, exist_ok=True)
    path_model = os.path.join(path_folder, 'model.h5')
    path_classes = os.path.join(path_folder, 'classes.p')
    path_fig = os.path.join(path_folder, 'graph.png')
    path_history = os.path.join(path_folder, 'history.p')

    input_shape = model.layers[0].layers[0].get_config()['batch_input_shape']
    target_size = (input_shape[1], input_shape[2])

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       shear_range=0,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=False,
                                       fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
                                    path_train,
                                    target_size=target_size,
                                    batch_size=bs_train,
                                    class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
                                  path_test,
                                  target_size=target_size,
                                  batch_size=bs_test,
                                  class_mode='categorical')

    if 'DenseNet121' in model.name:

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-3, momentum=0.9, decay=0),
                      metrics=['acc'])

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      validation_data=test_generator,
                                      validation_steps=steps_per_epoch // 2,
                                      shuffle=True)

    else:
        print('ERROR : Only DenseNet121 training is supported')
        return

    model.save(path_model)
    print(f'\nModel saved to {path_model}')

    classes_idx = train_generator.class_indices
    classes = list(classes_idx.keys())
    pickle.dump(classes, open(path_classes, "wb"))
    print(f'Classes saved to {path_classes}')

    pickle.dump(history.history, open(path_history, "wb"))
    print(f'History saved to {path_history}')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].plot(acc, label='train')
    ax[0].plot(val_acc, label='test')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].plot(loss, label='train')
    ax[1].plot(val_loss, label='test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')

    fig.suptitle(model.name)

    fig.savefig(path_fig, dpi=fig.dpi)
    print(f'Figure saved to {path_fig}')
