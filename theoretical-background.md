# Theoretical background

#### What is Convolutional neural network ( CNN )?

Convolutional Neural networks are specializes deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.

CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.



<figure><img src=".gitbook/assets/image (6).png" alt=""><figcaption><p><a href="https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529">https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529</a></p></figcaption></figure>

It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.

#### What is LSTM?

LSTM stands for **Long short term memory**, they are a type of RNN (**recurrent neural network**) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

This is what an LSTM cell looks like –

<figure><img src=".gitbook/assets/image (11).png" alt=""><figcaption><p><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">https://colah.github.io/posts/2015-08-Understanding-LSTMs/</a></p></figcaption></figure>

So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

* CNN is used for extracting features from the image. We will use the pre-trained model Xception.
* LSTM will use the information from CNN to help generate a description of the image.

#### Image caption generation algorithm

<figure><img src=".gitbook/assets/image (12).png" alt=""><figcaption><p><a href="https://github.com/ABX9801/Image-Caption-Generator">https://github.com/ABX9801/Image-Caption-Generator</a></p></figcaption></figure>
