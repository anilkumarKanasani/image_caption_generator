# Pre requisites

#### The Dataset of Python based Project

For the image caption generator, we will be using the Flickr\_8K dataset. There are also other big datasets like Flickr\_30K and MSCOCO dataset but it can take weeks just to train the network so we will be using a small Flickr8k dataset. The advantage of a huge dataset is that we can build better models.

* [Flicker8k\_Dataset ](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k\_Dataset.zip)
* [Flickr\_8k\_text ](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k\_text.zip)

#### Software requirements

The Flickr\_8k\_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”).

This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language processing.

Make sure you have installed all the following necessary libraries:

* pip install tensorflow
* keras
* pillow
* numpy
* tqdm
* jupyterlab

#### Source code file structure

* **Models –** It will contain our trained models.
* **Descriptions.txt –** This text file contains all image names and their captions after preprocessing.
* **Features.p –** Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
* **Tokenizer.p –** Contains tokens mapped with an index value.
* **Model.png –** Visual representation of dimensions of our project.
* **Testing\_caption\_generator.py –** Python file for generating a caption of any image.
* **Training\_caption\_generator.ipynb –** Jupyter notebook in which we train and build our image caption generator.

\
\
