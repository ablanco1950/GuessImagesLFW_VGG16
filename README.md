# GuessImagesLFW_VGG16
Simple application of VGG16 for the recognition of images, obtained from LFW, of a limited number of famous(15) with good performance (greater than 80%)

It has been obtained from the model shown in the query
  https://stackoverflow.com/questions/52575271/keras-vgg16-same-model-different-approach-gave-different-result

and

  https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

Requirements:

keras and tensorflow must be installed with the necessary modules to execute

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Flatten, Dropout

from keras import optimizers

from tensorflow.keras.models import load_model

Operation:

Downloaded the folder to disk and unzipped the lfw5 folder that has the images for train and test:

Train the model:

TrainGuessImagesLFW_VGG16Model.py

which produces the model: ModelGuessImages_LFW_VGG16.h5

Execute:

GuessImagesLFW_VGG16.py

that produces a console list with the correctly recognized images and the erroneously recognized ones.

References:

Images downloaded from LFW http://vis-ww.cs.umass.edu/lfw/#download

https://stackoverflow.com/questions/52575271/keras-vgg16-same-model-different-approach-gave-different-result

https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

https://github.com/ablanco1950/LFW_SVM_facecascade
