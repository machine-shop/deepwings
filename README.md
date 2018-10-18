# Deepwings 

This project follows the beewing project : https://github.com/machine-shop/beewing, which is itself inspired by the idBee project from the University of Wisconsin-Madison : http://idbee.ece.wisc.edu/


## Project motivation

As stated in the idBee project :

"About 35% of the world's crops and 80% of the world’s flowers are pollinated by bees! Most bees that are farmed for pollination are European Honeybees, whose physique and behavior makes them very efficient. But native bees can share in the pollination effort, and some may be more efficient.

Recently, honeybees have been dying off at an extraordinarily high rate, and no one is quite sure why. Researchers call it Colony Collapse Disorder, and the problem is serious because of our dependence on honeybees to pollinate food crops. While many are researching the cause of Colony Collapse Disorder, research into native bees may uncover more productive alternatives to the European honeybee. Finding the cause and potential native alternatives involve tracking wild bee populations and habits.

There are many species of bees, more than 500 in Wisconsin alone, but it's not easy to tell which species an individual belongs to.

While bee species identification is essential to research, identifying the species of a bee can be expensive and time-consuming. Since there are few experts who can reliably distinguish many different species, bees must be captured and sent to an expert, which may take several months. Bee research can be excruciatingly slow.

Rather than capturing, killing, and sending the bees off to be catalogued, imagine an iPhone app that lets graduate students and researchers identify bees in the field. One could simply take a photo of a bee and instantly record its species and location. Research could be conducted much faster, and identified bees could be released back to nature."

## Improvements

This time, we propose two methods : 
* A Convolutional Neural Network approach implementing the VGG16 pretrained network
* A classical pipeline with features extraction followed by an artificial Artificial Neural Network classifier

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Dependencies
* Keras
* TensorFlow <= 1.10.0
* Numpy
* Pandas
* Python 3.5
* Scikit-image
* Scikit-learn
* Scipy

### Installing

To download the project :
```
$ git clone https://github.com/machine-shop/deepwings
$ cd deepwings
$ python install.py
```

## Usage
### Images specifications
The typical size of our images is 2039x1536 pixels. The images should feature only the wing of the bee (body on the left side of the image) as follows :

![image](./examples/ex1.jpg)
![image](./examples/ex2.jpg)

On the other side, those types of images should be avoided :

![image](./examples/ex3.jpg)
![image](./examples/ex4.jpg)
![image](./examples/ex5.jpg)




### Predicting
This project comes with pretrained models for both the CNN and ANN methods. Move the pictures you want to predict to */deepwings/prediction/raw_images/test/*. Then run:

#### CNN
This method is preferred as it is faster and slightly more accurate. To predict species of your images, run :
```
$ python pipeline.py -pred cnn 
```
#### Feature extraction + ANN
First you need to extract the features from your pictures, then run the ANN classifier :
```
$ python pipeline.py -e pred
$ python pipeline.py -pred ann
```
or :
``` 
$ python pipeline.py -e pred -pred ann
```
Optional arguments :
```
$ python pipeline.py -e pred -pred ann --plot --category genus
```
* *--plot, -p* : if specified, explanatory figures will be plotted in */deepwings/prediction/explanatory_figures/* 
* *--category, -c* : 'genus' or 'species'(default). Specifies if the model must be a genus or species classifier.

The prediction results should appear in a csv file in */deepwings/prediction/*.


### Training
Training is only available with the features extraction model.
But you can tune the CNN model as you wish : */deepwings/method_cnn/models/VGG16_2nd_method_dataug_110epft20ep.h5* 

The dataset used in this project is available [here](https://www.dropbox.com/sh/r04kyryo6ljs6x0/AAAhAU4XKVJzuRyrroYLVdnua?dl=0)
#### 1. Filename convention
The name of each image is composed of an unique identification number (per bee), the genus of the bee, the species of the bee.
In addition, you may add the following information (optional) : the subspecies of the bee, whether the right or left wing was used, the gender of the bee, and finally the magnification of the stereoscope with each value separated by a space.     

Example : *30 Lasioglossum rohweri f left 4x bx.jpg* or *1239 Osmia lignaria propinqua f right 4x.jpg*
#### 2. Features extraction
Move all your raw images to */deepwings/training/raw_images/*. Then run:
```
$ python pipeline.py -e train -restart
```
This process can last a few hours depending on the size of your dataset.
This will create several spreadsheets :
* *data_6cells.csv* : Features of images with no 3rd submarginal cell detected
* *data_7cells.csv* : Features of images with 3rd submarginal cell detected
* *invalid.csv* : List of images whose features could not be properly extracted

If this process gets interrupted at some point, you can pick up extraction where you left off with:
```
$ python pipeline.py -e train
```
Optional arguments :
```
$ python pipeline.py -e train --plot --n_fourier_descriptors 25
```
* *--n_fourier_descriptors, -fd* : number of fourier descriptors extracted for each cell detected (15 by default) 
* *--plot, -p* : if specified, explanatory figures will be plotted in */deepwings/training/explanatory_figures/* 

#### 3. Artificial Neural Network
This is a very basic ANN, any other classifier could be used for this task. Once the features are extracted, you can run:
``` 
$ python pipeline.py -t ann 
```
This will update the models in */deepwings/method_features_extraction/classifiers/models_ann/*.

You can add a few optional arguments:
```
$ python pipeline.py -t ann --category genus --min_images 5 --test-size 0.3
```
* *--category, -c* : 'genus' or 'species'(default). Specifies if the model must be a genus or species classifier.
* *--min_images, -m* : minimum number of images needed for a category (genus/species) to be taken into account
* *--test-size, -ts* : number between 0 and 1, specifying the ratio *#test/#total* (0.2 by default)


