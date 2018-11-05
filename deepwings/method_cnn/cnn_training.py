from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


def build_model(dense_layers=4, dropout=0.5, n_class=21,
                weights_conv='imagenet', start=300, input_shape=(150, 150, 3)):

    base = VGG16(weights=weights_conv, include_top=False,
                 input_shape=input_shape)
    model = Sequential()
    model.add(base)
    model.add(Flatten(input_shape=(None, 4, 4, 512)))
    for i in range(dense_layers-1):
        n_outputs = int(start/(2**i))
        model.add(Dense(n_outputs, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(n_class, activation='softmax'))
    print(model.layers[0].summary())
    print(model.summary())

    return model


def train_model(model, path_train='training/sorted_species/train/',
                path_test='training/sorted_species/test/',
                target_size=(150, 150), bs_train=20, bs_test=20,
                epochs1=1, epochs2=1, top_layers_ft=3):

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

    model.layers[0].trainable = False
    print(model.layers[0].summary())
    print(model.summary())

    print('Compiling ..')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    print('Model training ..')

    # history1 = model.fit_generator(train_generator,
    # steps_per_epoch=1,
    # epochs=epochs1,
    # validation_data=test_generator,
    # validation_steps=1)

    if epochs2 > 0:
        layers_inverted = []

        conv_layers = model.layers[0].layers
        counter = 0
        for i, layer in enumerate(reversed(conv_layers)):
            if ('filters' in layer.get_config().keys() and
                    counter < top_layers_ft):
                layer.trainable = True
                counter += 1
            else:
                layer.trainable = False
            layers_inverted.append(layer)

        for layer in layers_inverted:
            print('layer')
            print(layer.get_config())
        base = Sequential()
        for layer in reversed(layers_inverted):
            base.add(layer)
        print(base.summary())
        ft_model = Sequential()
        ft_model.add(base)
        for layer in model.layers[1:]:
            ft_model.add(layer)
        # print(model.layers[0].summary())
        ft_model.compile(loss='categorical_crossentropy',
                         optimizer=optimizers.RMSprop(lr=2e-5),
                         metrics=['acc'])
        ft_model.build()

        print(ft_model.summary())

        # history2 = model.fit_generator(train_generator,
        # steps_per_epoch=1,
        # epochs=epochs2,
        # validation_data=test_generator,
        # validation_steps=1)

    return model
