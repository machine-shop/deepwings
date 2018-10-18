#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:25:18 2018

@author: theo
"""

import numpy as np
import os
import random
from shutil import copy2, rmtree
import pandas as pd


class Sorter:
    
    
    def __init__(self, path, test_size, min_images, category):
        if path[-1] != '/':
            path+='/'
        self.path = path
        self.min_images = min_images
        if category== 'species' or category == 'genus':
            self.path_sorted= 'training/' + 'sorted_' + category + '/'
            self.category = category
        else:
            print('ERROR: category undefined')
            
        self.test_size = test_size
        self.train_path = self.path_sorted +'train/'
        self.test_path = self.path_sorted + 'test/'
        
        
    def list_category(self):
        
        files = os.listdir(self.path)
        dic = {}
        for f in files:
            genus = f.split()[1].lower()
            if self.category=='genus':
                category = genus
            else : 
                category = genus + '_' + f.split()[2].lower()
            if not category in dic:
                dic[category]=1
            else:
                dic[category]+=1
        ls = list(dic.keys())
        for category in ls:
            if dic[category]<self.min_images:
                del dic[category]
        print('\nAt least %i images per %s' %(self.min_images, self.category))
        if self.category=='genus':
            print('Processed genera (%i) :\n' %len(dic))
        else: 
            print('Processed species (%i) :\n' %len(dic))
        series = pd.Series(dic)
        
        print(series.sort_values())
        print('')
        print('TOTAL :',series.sum() )
        """
        keys = list(dic.keys())
        values = list(dic.values())
        indexes = np.argsort(values)
        for i in indexes:
            print('-%s (%i)' %(keys[i], values[i]))
            """
            
        return list(dic.keys())
    
    def generate_image_paths(self):
        """takes the location of a folder on a computer and returns a list of image
        paths respecting min images of the chosen category"""
        img_list = []

        valid_categories = self.list_category()
       
                
        files = os.listdir(self.path)
        for f in files:
            genus = f.split(' ')[1].lower()
            if self.category=='genus':
                _category = genus
            else : 
                _category = genus + '_' + f.split()[2].lower()
            if _category in valid_categories:
                img_list += [self.path + f]
        
        return img_list
    
    def create_subfolders(self, subfolder, files):
        
        list_elements=[]
        
        for f in files:
            genus = f.split()[1].lower()
            if self.category=='genus':
                name = genus
            else : 
                name = genus + '_' + f.split()[2].lower()
            folder = self.path_sorted + subfolder+ '/' + name
            if not os.path.exists(folder):
                os.makedirs(folder)
                list_elements.append(name)
            copy2(self.path +f, folder)
        n = len(list_elements)
        if subfolder=='train':
            self.ncategories_train = n
            
        elif subfolder=='test':
            self.ncategories_test = n
            
            
            
        
        
        
    def sort_images(self):
        
        
        try:
            if os.path.exists(self.path_sorted):
                rmtree(self.path_sorted)
                print(self.path_sorted +' already exists : deleted')
            os.makedirs(self.path_sorted)
            print(self.path_sorted + ' created')
        except OSError:
            print('Error: Creating directory '+ self.path_sorted)
            
        files = os.listdir(self.path)
        
        categories_processed = self.list_category()
             
        
        files_processed=[]
        for f in files:
            genus = f.split()[1].lower()
            if self.category=='genus':
                name = genus
            else : 
                name = genus + '_' + f.split()[2].lower()
            if name in categories_processed:
                files_processed.append(f)
                
        self.n_raw = len(files)
        self.n_processed = len(files_processed)
        random.seed((5))
        random.shuffle(files_processed)
        n_train = int((1-self.test_size)*self.n_processed)
        train = files_processed[:n_train]
        test = files_processed[n_train:]
        self.n_train = n_train
        self.n_test = self.n_processed - n_train
        self.create_subfolders('train', train)
        self.create_subfolders('test', test)
        
        
        
        
        
        
        
        
        
    
    