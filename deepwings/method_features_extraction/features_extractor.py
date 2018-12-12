import csv
import os
import pandas as pd
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray

from method_features_extraction import image_processing as ip
import matplotlib.pyplot as plt


def process_and_extract(path_raw_image, plot=False, n_descriptors=15):
    """Process a wing_photo() object and extracts features from its cells
    if the wing_photo() object is valid

    Arguments
    ---------
    path_raw_image : str
        path of the image to process
    plot : bool
        if true, an explanatory figures is plotted
        in 'method_feature_extraction/explanatory_figures/'
    n_descriptors : int
        number of Fourier descriptors we want to extract for each cell

    Returns
    -------
    csv_name : str
        'data_6cells.csv' or 'data_7cells.csv' depending on
        the number of cells detected. If the image is invalid, returns None
    output : 2D list
        row containing the name of the picture and all extracted features.
        If the image is invalid, returns None
    """
    filename = path_raw_image.split('/')[-1]
    parent_folder = path_raw_image.split('/')[-3]

    image_rgb = imread(path_raw_image)
    img_gray = rgb2gray(image_rgb)

    binary = ip.block_binarization(img_gray, 20, 100)

    cleared_binary = ip.clear_binary(binary)
    cleared_binary = ndi.binary_erosion(cleared_binary, iterations=5)

    img_label, markers, distances, labels = ip.create_img_label(
                                               cleared_binary)

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(40, 20))
        plt.suptitle(filename, fontsize=16)

        ax[0, 0].set_title('Grayscale')
        ax[0, 0].imshow(img_gray)
        ax[0, 1].set_title('Blocks binarization')
        ax[0, 1].imshow(binary)
        ax[0, 2].set_title('Reconstructed and eroded')
        ax[0, 2].imshow(cleared_binary)
        ax[0, 3].set_title('Markers')
        ax[0, 3].imshow(markers)
        ax[0, 4].set_title('Distances')
        ax[0, 4].imshow(distances)
        ax[1, 0].set_title('Distances reversed')
        ax[1, 0].imshow(-distances)
        ax[1, 1].set_title('Labels')
        ax[1, 1].imshow(labels)
        ax[1, 2].set_title('Image label')
        ax[1, 2].imshow(img_label)
        ax[1, 3].set_title('Before region filtering')
        ax[1, 4].set_title('After region filtering')

    rs = ip.region_sorter(img_label, filename)

    rs.filter_area()
    rs.filter_weird_shapes()

    if plot:
        rs.plot(ax[1, 3], img_gray)

    rs.update_img_label()
    rs.find_neighbors()
    rs.filter_neighbors()

    rs.compute_eccentricities()
    rs.find_central_cell()

    rs.compute_angles()
    rs.filter_angles()

    rs.filter_annoying_cell()
    # rs.update_center()

    rs.identify_regions()
    rs.compute_fd(n_descriptors)

    if plot:
        rs.plot(ax[1, 4], img_gray)
        if rs.valid_image:
            path_plot = os.path.join(parent_folder, 'valid_images', filename)
        else:
            path_plot = os.path.join(parent_folder, 'invalid_images', filename)
        plt.savefig(path_plot)
        plt.close()

    nb_cells, output = rs.extract_features()

    return nb_cells, output


def extract_pictures(paths_images, plot, n_descriptors, continue_csv):
    """Extract features from pictures satisfying some criteria:
    We only keep pictures of bees having at least <min_images> per <category>

    Arguments
    ---------
    path_images : str
        Input raw images folder path of images to process
    plot : bool
        if true, explanatory figures will be plotted
        in 'method_feature_extraction/explanatory_figures/'
    n_descriptors : int
        number of Fourier descriptors we want to extract for each cell
    continue_csv : bool
        if true, continue extracting process from already existing csv_files
        if false, erase all csv_files and restart from scratch
    """
    parent_folder = paths_images[0].split('/')[-3]
    path_valid_images = os.path.join(parent_folder, 'valid_images')
    path_invalid_images = os.path.join(parent_folder, 'invalid_images')

    path_6cells_csv = os.path.join(parent_folder, 'data_6cells.csv')
    path_7cells_csv = os.path.join(parent_folder, 'data_7cells.csv')
    path_invalid_csv = os.path.join(parent_folder, 'invalid.csv')
    path_valid_csv = os.path.join(parent_folder, 'valid.csv')

    if plot:
        os.makedirs(path_valid_images, exist_ok=True)
        os.makedirs(path_invalid_images, exist_ok=True)

    already_extracted = []

    if continue_csv:
        try:
            paths_csv = [path_6cells_csv, path_7cells_csv, path_invalid_csv]
            for path_csv in paths_csv:
                dataset = pd.read_csv(path_csv)
                names = dataset.iloc[:, 0].values
                already_extracted += list(names)
            already_extracted = sorted(already_extracted)

        except OSError:
            print('Error: could not find csv files')

    else:
        # Creating the csv files and headers
        cells_ordered = ['marg', '1st_med', '2nd_med', '2nd_cub', '1st_sub',
                         '2nd_sub', '3rd_sub']

        # initializing invalid.csv
        with open(path_invalid_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['names'])

        # initializing valid.csv
        # Headers
        header = ['names']
        header += (
            [cell+'_area' for cell in cells_ordered] +
            [cell+'_ecc' for cell in cells_ordered] +
            [cell+'_ang' for cell in cells_ordered if cell != '1st_med'])

        for cell in cells_ordered:
            for i in range(2, n_descriptors + 2):
                header += [cell + '_fd' + str(i)]

        # Writing header to csv
        with open(path_valid_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

    n = len(paths_images)

    # Process and extract features for each image
    for i, path in enumerate(paths_images):

        print(f'### Image {i+1}/{n}')
        name_image = path.split('/')[-1]
        print(f'# Image name : {name_image}')

        if name_image in already_extracted:
            print('# Already extracted')
        else:
            nb_cells, output = process_and_extract(path, plot, n_descriptors)
            if not nb_cells:  # if the image is not valid
                print('# Invalid image')
                with open(path_invalid_csv, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([name_image])
            else:
                print('# Valid image')
                with open(path_valid_csv, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(output)
