# Loading Dataset

**Loading dataset for Training the model**

In our **Flickr\_8k\_test** folder, we have **Flickr\_8k.trainImages.txt** file that contains a list of 6000 image names that we will use for training.

For loading the training dataset, we need more functions:

* **load\_photos( filename ) –** This will load the text file in a string and will return the list of image names.
* **load\_clean\_descriptions( filename, photos ) –** This function will create a dictionary that contains captions for each photo from the list of photos. We also append the \<start> and \<end> identifier for each caption. We need this so that our LSTM model can identify the starting and ending of the caption.
* **load\_features(photos) –** This function will give us the dictionary for image names and their feature vector which we have previously extracted from the Xception model.

```python
# Python Code

#load the data 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features


filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)
```
