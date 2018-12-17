import os
import numpy as np
import pickle
from shutil import copyfile, rmtree
from collections import defaultdict
import pandas as pd


class Sorter:

    def __init__(self, path_raw='training/raw_images', category='species',
                 min_images=20, test_size=0.3, random_seed=1234):

        self.path_raw = path_raw.strip(r' +/ +')
        self.path_parent = os.path.join(*self.path_raw.split('/')[:-1])
        self.category = category
        self.min_images = min_images
        self.path_sorted = os.path.join(self.path_parent,
                                        f'sorted_{category}')
        self.test_size = test_size
        self.random_seed = random_seed
        self.path_train = os.path.join(self.path_sorted, 'train')
        self.path_test = os.path.join(self.path_sorted, 'test')

    def filter_categories(self, verbose=False):
        """Returns a dictionary {group: list_images} of groups having at least
        self.min_images"""

        files = os.listdir(self.path_raw)
        dict_groups = defaultdict(list)
        for img_name in files:
            genus = img_name.split()[1].lower()
            if self.category == 'genus':
                group = genus
            else:
                species = img_name.split()[2].lower()
                group = f'{genus}_{species}'
            dict_groups[group].append(img_name)

        dict_groups = dict(dict_groups)
        groups = np.array(list(dict_groups.keys()))
        counts = np.array([len(dict_groups[group]) for group in groups])

        mask = (counts >= self.min_images)
        counts_filtered = counts[mask]
        groups_filtered = groups[mask]
        dict_filtered = {gr: dict_groups[gr] for gr in groups_filtered}
        n_groups = len(groups_filtered)
        n_groups_init = len(groups)

        print(f'\nAt least {self.min_images} images per {self.category}')
        if self.category == 'genus':
            print(f'Number of genera : {n_groups} '
                  f'(initial : {n_groups_init})\n')
        else:
            print(f'Number of species : {n_groups} '
                  f'(initial : {n_groups_init})\n')

        if verbose:
            series = pd.Series(counts_filtered, index=groups_filtered)

            print(series.sort_values())
            print('')
            print(f'TOTAL : {series.sum()}')

        self.dict_filtered = dict_filtered

    def train_test_split(self, verbose=False):
        """Split dict filtered in train and test with the same random seed"""

        dict_train = defaultdict(list)
        dict_test = defaultdict(list)
        n_tot_train = 0
        n_tot = 0

        for group, images in self.dict_filtered.items():
            images_sorted = np.array(sorted(images))
            n = len(images_sorted)
            n_train = int(n * (1 - self.test_size))
            np.random.seed(self.random_seed)
            np.random.shuffle(images_sorted)
            dict_train[group] += list(images_sorted[:n_train])
            dict_test[group] += list(images_sorted[n_train:])
            n_tot += n
            n_tot_train += n_train

        test_size_exp = round((n_tot - n_tot_train)/n_tot, 4)

        if verbose:
            print(f'\nRandom seed : {self.random_seed}')
            print(f'Input test size : {self.test_size}')
            print(f'Experimental test size : {test_size_exp}\n')

        self.dict_train = dict(dict_train)
        self.dict_test = dict(dict_test)

    def create_subfolders(self, train_test):
        """Create subfolders corresponding to train/[categories] and
        test/[categories]"""

        if train_test == 'train':
            path_set = self.path_train
            if os.path.exists(path_set):
                rmtree(path_set)
            os.makedirs(path_set, exist_ok=True)
            dict_set = self.dict_train
        else:
            path_set = self.path_test
            if os.path.exists(path_set):
                rmtree(path_set)
            os.makedirs(path_set, exist_ok=True)
            dict_set = self.dict_test

        for group, img_names in dict_set.items():
            path_group = os.path.join(path_set, group)
            os.makedirs(path_group, exist_ok=True)
            for img_name in img_names:
                path_origin = os.path.join(self.path_raw, img_name)
                path_dest = os.path.join(path_group, img_name)
                copyfile(path_origin, path_dest)

        if train_test == 'train':
            print(f'Train {self.category} subfolders created to {path_set}')
        else:
            print(f'Test {self.category} subfolders created to {path_set}')

    def pickle_train_test(self):
        """Save train/test image names, parameters and classes to be reused
        by classifiers"""

        path_info = os.path.join(self.path_parent, 'info_train_test.p')

        dict_info = {'random_seed': self.random_seed,
                     'category': self.category,
                     'test_size': self.test_size,
                     'min_images': self.min_images,
                     'train': [], 'test': []}

        for images in self.dict_train.values():
            dict_info['train'] += images

        for images in self.dict_test.values():
            dict_info['test'] += images

        classes = sorted(self.dict_filtered.keys())
        dict_info['classes'] = classes

        pickle.dump(dict_info, open(path_info, "wb"))
        print(f'\nTrain/test information saved to {path_info}')
