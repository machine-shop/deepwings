import argparse
import utils
import os

import method_cnn.cnn_prediction as cnnp
# import method_cnn.cnn_training as cnnt
from method_features_extraction.classifiers import ann_classifier as annc
from method_features_extraction import features_extractor as fe


"""
Predict bee species from images of their wings.

Two different methods:
    - Convolutional Neural Network (based on VGG16)
    - Features extraction + classifier (ANN)
"""


def main():
    parser = argparse.ArgumentParser(
        description='Script preprocess the bee wing raw images, extracts the \
        features from preprocessed bee wing image, and trains a Random Forest\
        Classifier using extracted features')
    parser.add_argument('-restart', '--restart',
                        action='store_true',
                        help='Restart feature extraction from beginning, \
                        erasing the current csv files')
    parser.add_argument('-c', '--category',
                        type=str,
                        help="Category to classify: 'genus' or 'species'",
                        required=False,
                        default='species')
    parser.add_argument('-m', '--min_images',
                        type=int,
                        help='Minimum number of images per category',
                        required=False,
                        default=20)
    parser.add_argument('-s', '--sort',
                        action='store_true',
                        help='Create subfolders train, test with the \
                        different categories')
    parser.add_argument('-pred', '--model_predicting',
                        type=str,
                        help="Choose a model : 'cnn', 'ann' ...",
                        required=False)
    parser.add_argument('-e', '--extraction',
                        type=str,
                        help="Run feature extraction process for 'train' or \
                        'pred' ",
                        default='none')
    parser.add_argument('-fd', '--n_fourier_descriptors',
                        type=int,
                        help='Number of Fourier descriptors used for each \
                        cell',
                        required=False,
                        default=15)
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help="If True, plots figures in valid_images/ or \
                        invalid_images/")
    parser.add_argument('-t', '--train',
                        type=str,
                        help="Choose a model : only 'ann' available for now",
                        required=False)
    parser.add_argument('-r', '--path_raw',
                        type=str,
                        help='The input path for raw image',
                        required=False,
                        default='./training/raw_images/')
    parser.add_argument('-ts', '--test_size',
                        type=float,
                        help='The ratio of dataset used for testing',
                        required=False,
                        default=0)
    args = parser.parse_args()

    if args.category not in ['genus', 'species']:
        print("ERROR: category must be 'genus' or 'species'")
        return 0
    if args.extraction not in ['none', 'train', 'pred']:
        print("ERROR: extraction must be 'pred' or 'train'")
        return 0

    pipeline_process = []
    if args.sort:
        pipeline_process += ['sort']
    if args.extraction == 'pred':
        pipeline_process += ['extraction_pred']
    if args.extraction == 'train':
        pipeline_process += ['extraction_train']
    if args.train:
        pipeline_process += ['train_' + args.train]
    if args.model_predicting:
        pipeline_process += ['pred_' + args.model_predicting]

    if len(pipeline_process) == 0:
        print('No argument entered: -s sorting  -e extraction  -t training\
              -ucsv update_csv')
        print("Type 'python pipeline.py -h' for further information")

    for step in pipeline_process:
        if step == 'sort':
            s = utils.Sorter(args.path_raw,
                             args.test_size,
                             args.min_images,
                             args.category)
            s.sort_images()
        elif step == 'extraction_train':

            # Generate a list of image paths respecting min images
            image_sorter = utils.Sorter(path=args.path_raw,
                                        test_size=0,
                                        min_images=args.min_images,
                                        category=args.category)
            paths_images = image_sorter.generate_image_paths()
            fe.extract_pictures(paths_images=paths_images,
                                plot=args.plot,
                                n_descriptors=args.n_fourier_descriptors,
                                continue_csv=not(args.restart))
        elif step == 'extraction_pred':  # Features extraction
            directory = 'prediction/raw_images/'
            paths_images = []
            for image_name in os.listdir(directory):
                paths_images.append(directory + image_name)
            fe.extract_pictures(paths_images=paths_images,
                                plot=args.plot,
                                n_descriptors=args.n_fourier_descriptors,
                                continue_csv=False)
        elif step == 'train_ann':
            annc.train(category=args.category)
        # In progresss :
        # elif step == 'train_cnn':
            # model = cnnt.build_model()
            # trained = cnnt.train_model(model)
        elif step == 'pred_ann':
            annc.predict(category=args.category,
                         plot=args.plot,
                         n_descriptors=args.n_fourier_descriptors)
        elif step == 'pred_cnn':
            cnnp.cnn_pred(category=args.category)


if __name__ == "__main__":
    main()
