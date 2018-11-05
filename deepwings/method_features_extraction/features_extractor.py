import csv
import os
import pandas as pd
from scipy import ndimage as ndi
from shutil import rmtree
from skimage.morphology import watershed
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import time
from skimage.segmentation import clear_border

from method_features_extraction import image_processing as ip
import matplotlib.pyplot as plt


class wing_photo():

    def __init__(self, file_path):
        self.path = file_path
        self.information = file_path.split('/')
        self.file_name = self.information[-1]
        splitted = self.file_name.split(' ')
        self.family = splitted[1]
        self.species = splitted[2]
        self.image = imread(file_path)[:, :, 0]

    def process_and_extract(self, plot, n_descriptors, folder_path):
        """Process a wing_photo() object and extracts features from its cells
        if the wing_photo() object is valid

        Arguments
        ---------
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

        min_area = 11000
        max_area = 300000
        image = imread(self.path)[:, :, 0]
        img_gray = rgb2gray(image)
        img_gray = resize(img_gray, (1600, 2000))

        t = time.time()
        binary = ip.block_binarization(img_gray, 20, 100)
        duration = round(time.time() - t, 4)
        print(f'# Block binarization lasted {duration}s')

        cleared_binary = ip.clear_binary(binary)
        cleared_binary = ndi.binary_erosion(cleared_binary, iterations=5)

        t = time.time()
        binary_structure = ndi.generate_binary_structure(2, 1)
        markers, _ = ndi.label(cleared_binary,
                               structure=binary_structure)
        markers, num_features = ip.filter_and_label(markers)
        if num_features > 255:
            print("### Warning : more than 255 regions detected")
        distances = ndi.distance_transform_edt(cleared_binary)
        labels = watershed(-distances, markers)
        labels[0, :] = 0
        labels[:, -1] = 0
        labels[:, 0] = 0
        img_label = clear_border(labels)
        duration = round(time.time() - t, 4)
        print(f'# Creating img label lasted {duration}s')

        if plot:
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(40, 20))
            plt.suptitle(self.file_name, fontsize=16)

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

        rs = ip.region_sorter(img_label, self.file_name)

        rs.filter_area(min_area, max_area)
        rs.filter_weird_shapes()
        if plot:
            rs.plot(img_gray, ax[1, 3])

        rs.find_neighbors()
        rs.filter_neighbors()
        rs.compute_eccentricities()

        rs.find_central_cell()
        rs.compute_angles()
        rs.filter_angles()

        rs.filter_annoying_cell()
        rs.update_center()

        rs.identify_regions()
        rs.compute_fd(n_descriptors)

        if plot:
            rs.plot(img_gray, ax[1, 4])
            if rs.valid_image:
                path_figure = folder_path + 'valid_images/'
            else:
                path_figure = folder_path + 'invalid_images/'
            plt.savefig(path_figure + self.file_name)
            plt.close()

        csv_name, output = rs.extract_features()

        return csv_name, output


def extract_pictures(folder_path, paths_images, plot, n_descriptors,
                     continue_csv):
    """Extract features from pictures satisfying some criteria:
    We only keep pictures of bees having at least <min_images> per <category>

    Arguments
    ---------
    folder_path : str
        path of the prediction or training folder
    category : str
        level of classification, must be 'genus' or 'species'
    min_images : int (default 20)
        min number of image per category
    plot : bool
        if true, explanatory figures will be plotted
        in 'method_feature_extraction/explanatory_figures/'
    n_descriptors : int
        number of Fourier descriptors we want to extract for each cell
    continue_csv : bool
        if true, continue extracting process from already existing csv_files
        if false, erase all csv_files and restart from scratch
    """
    already_extracted = []

    if continue_csv:
        try:
            if not os.path.exists(folder_path + "data_6cells.csv"):
                print("data_6cells.csv not found")
                return 0
            if not os.path.exists(folder_path + "data_7cells.csv"):
                print("data_7cells.csv not found")
                return 0
            if plot:
                if not os.path.exists(folder_path + 'valid_images/'):
                    os.makedirs(folder_path + 'valid_images/')
                if not os.path.exists(folder_path + 'invalid_images/'):
                    os.makedirs(folder_path + 'invalid_images/')

        except OSError:
            print('Error: Finding directories')

        for csv_name in ['data_6cells.csv', 'data_7cells.csv', 'invalid.csv']:
            dataset = pd.read_csv(folder_path + csv_name)
            names = dataset.iloc[:, 0].values
            already_extracted += list(names)
        already_extracted = sorted(already_extracted)

    else:
        try:
            if os.path.exists(folder_path + "data_6cells.csv"):
                os.remove(folder_path + "data_6cells.csv")
            if os.path.exists(folder_path + "data_7cells.csv"):
                os.remove(folder_path + "data_7cells.csv")
            if os.path.exists(folder_path + "invalid.csv"):
                os.remove(folder_path + "invalid.csv")
            if plot:
                if os.path.exists(folder_path + 'valid_images/'):
                    rmtree(folder_path + 'valid_images/')
                os.makedirs(folder_path + 'valid_images/')
                if os.path.exists(folder_path + 'invalid_images/'):
                    rmtree(folder_path + 'invalid_images/')
                os.makedirs(folder_path + 'invalid_images/')

        except OSError:
            print('Error: Creating directories ')

        # Creating the csv files and headers
        cells_ordered = ['marg', '1st_med', '2nd_med', '2nd_cub', '1st_sub',
                         '2nd_sub', '3rd_sub']

        # initializing invalid images
        with open(folder_path + 'invalid.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['names'])

        # initializing other csv files
        for csv_name in ['data_6cells.csv', 'data_7cells.csv']:
            if csv_name == 'data_6cells.csv':
                relevant_cells = cells_ordered[:-1]
            else:
                relevant_cells = cells_ordered

            # Headers
            header = ['names']
            header += (
                [cell+'_area' for cell in relevant_cells[:-1]] +
                [cell+'_ecc' for cell in relevant_cells] +
                [cell+'_ang' for cell in relevant_cells if cell != '1st_med'])

            for cell in relevant_cells:
                for i in range(2, n_descriptors + 2):
                    header += [cell + '_fd' + str(i)]

            # Writing header to csv
            with open(folder_path + csv_name, 'a') as csv_file:
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
            photo = wing_photo(path)
            nb_cells, output = photo.process_and_extract(plot,
                                                         n_descriptors,
                                                         folder_path)
            if not nb_cells:  # if the image is not valid
                print('# Invalid image')
                with open(folder_path + 'invalid.csv', 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([name_image])
            else:
                print('# Valid image')
                if nb_cells == 6:
                    csv_name = 'data_6cells.csv'
                else:
                    csv_name = 'data_7cells.csv'
                with open(folder_path + csv_name, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(output)
