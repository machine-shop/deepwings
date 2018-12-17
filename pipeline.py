import argparse
import pickle
import os

import deepwings.utils as utils
import deepwings.method_cnn.cnn_prediction as cnnp
from deepwings.method_features_extraction import ann_classifier as annc
from deepwings.method_cnn import cnn_training as cnnt
from deepwings.method_features_extraction import features_extractor as fe


DESCRIPTION = """
Predict bee species from images of their wings.

Two different methods:
    - Convolutional Neural Network (based on DenseNet121)
    - Features extraction + classifier (ANN)
"""


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-l', '--list_categories',
                        action='store_true',
                        help='Display list of genera/species complying with\
                        min_images')
    parser.add_argument('-s', '--sort',
                        action='store_true',
                        help='Create subfolders train/test with the\
                        different genera/species')
    parser.add_argument('-c', '--category',
                        type=str,
                        help="Category to classify: 'genus' or 'species'",
                        default='species')
    parser.add_argument('-m', '--min_images',
                        type=int,
                        help='Minimum number of images per genus/species',
                        default=20)
    parser.add_argument('-rs', '--random_seed',
                        type=int,
                        help='Random seed used for train/test split',
                        default=1234)
    parser.add_argument('-restart', '--restart',
                        action='store_true',
                        help='Restart feature extraction from beginning,\
                        erasing the current csv files')
    parser.add_argument('-pred', '--model_prediction',
                        type=str,
                        help="Choose a model : 'cnn' or 'ann'")
    parser.add_argument('-e', '--extraction',
                        type=str,
                        help="Run feature extraction process for 'train' or\
                        'pred'")
    parser.add_argument('-fd', '--n_fourier_descriptors',
                        type=int,
                        help='Number of Fourier descriptors used for each\
                        cell',
                        default=15)
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help="If True, plots figures in valid_images/ or\
                        invalid_images/")
    parser.add_argument('-t', '--train',
                        type=str,
                        help="Choose a model : only 'ann' or 'cnn'",
                        required=False)
    parser.add_argument('-raw_train', '--path_raw_training',
                        type=str,
                        help='Input path for raw image used for training',
                        default='training/raw_images/')
    parser.add_argument('-ts', '--test_size',
                        type=float,
                        help='The ratio of dataset used for testing',
                        required=False,
                        default=0.3)
    parser.add_argument('-fp', '--folder_to_predict',
                        type=str,
                        help="Path to folder of images to predict",
                        default='prediction/raw_images')
    parser.add_argument('-cnn', '--name_cnn',
                        type=str,
                        help="Name of the CNN model",
                        default="DenseNet121")
    parser.add_argument('-raw_pred', '--raw_images_prediction',
                        type=str,
                        help="Path to the folder to predict",
                        default='prediction/raw_images')
    parser.add_argument('-pann', '--path_ann',
                        type=str,
                        help="Path to the ANN model")
    parser.add_argument('--n_epochs',
                        type=int,
                        help='Number of epochs for CNN training',
                        default=20)
    parser.add_argument('-bs_train', '--batch_size_train',
                        type=int,
                        help='Batch size for CNN training',
                        default=20)
    parser.add_argument('-bs_test', '--batch_size_test',
                        type=int,
                        help='Batch size for CNN validation',
                        default=20)
    parser.add_argument('--steps_epoch',
                        type=int,
                        help='Steps per epoch for CNN training',
                        default=100)

    args = parser.parse_args()

    if args.category not in ['genus', 'species']:
        print("ERROR: category must be 'genus' or 'species'")
        return
    if args.extraction not in [None, 'training', 'pred']:
        print("ERROR: extraction must be 'pred' or 'train'")
        return

    pipeline_process = []
    if args.list_categories:
        pipeline_process += ['list_categories']
    if args.sort:
        pipeline_process += ['sort']
    if args.extraction == 'pred':
        pipeline_process += ['extraction_pred']
    if args.extraction == 'training':
        pipeline_process += ['extraction_training']
    if args.train:
        pipeline_process += [f'train_{args.train}']
    if args.model_prediction:
        pipeline_process += [f'pred_{args.model_prediction}']

    if len(pipeline_process) == 0:
        print("No argument entered, type 'python pipeline.py -h' for"
              " further information")

    for step in pipeline_process:
        if step == 'list_categories':
            sorter = utils.Sorter(args.path_raw_training, args.category,
                                  args.min_images)
            sorter.filter_categories(verbose=True)

        if step == 'sort':
            sorter = utils.Sorter(args.path_raw_training, args.category,
                                  args.min_images, args.test_size,
                                  args.random_seed)
            sorter.filter_categories(verbose=True)
            sorter.train_test_split(verbose=True)
            sorter.create_subfolders('train')
            sorter.create_subfolders('test')
            sorter.pickle_train_test()

        elif step == 'extraction_training':
            dict_info = pickle.load(open('training/info_train_test.p', 'rb'))
            selected_images = dict_info['train'] + dict_info['test']
            paths_images = []
            for img_name in selected_images:
                path_img = os.path.join(args.path_raw_training, img_name)
                paths_images.append(path_img)

            fe.extract_pictures(paths_images=paths_images,
                                plot=args.plot,
                                n_descriptors=args.n_fourier_descriptors,
                                continue_csv=not(args.restart))

        elif step == 'extraction_pred':  # Features extraction
            paths_images = []
            for image_name in os.listdir(args.folder_to_predict):
                path_img = os.path.join(args.folder_to_predict, image_name)
                paths_images.append(path_img)

            fe.extract_pictures(paths_images=paths_images,
                                plot=args.plot,
                                n_descriptors=args.n_fourier_descriptors,
                                continue_csv=False)

        elif step == 'train_ann':
            annc.train(category=args.category)

        elif step == 'train_cnn':
            model = cnnt.build_model()
            cnnt.train_model(model,
                             epochs=args.n_epochs,
                             bs_train=args.batch_size_train,
                             bs_test=args.batch_size_test,
                             steps_per_epoch=args.steps_epoch)

        elif step == 'pred_ann':
            annc.predict(category=args.category,
                         path_raw=args.raw_images_prediction,
                         path_model=args.path_ann,
                         n_descriptors=args.n_fourier_descriptors)

        elif step == 'pred_cnn':
            cnnp.cnn_pred(model_name=args.name_cnn,
                          path_raw=args.raw_images_prediction)


if __name__ == "__main__":
    main()
