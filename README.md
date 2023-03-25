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

TrainModelGuessImagesLFW_VGG16Model.py

which produces the model: ModelGuessImages_LFW_VGG16.h5

Execute:

GuessImagesLFW_VGG16.py

that produces a console list with the correctly recognized images and the erroneously recognized ones:

Abel_Pacheco is class 0
Alejandro_Toledo is class 1
Angelina_Jolie is class 2
Ariel_Sharon is class 3
Arnold_Schwarzenegger is class 4
Barbra_Streisand is class 5
Bill_Clinton is class 6
Bill_Gates is class 7
Britney_Spears is class 8
Carlos_Menem is class 9
Mahmoud_Abbas is class 10
Meryl_Streep is class 11
Michael_Jackson is class 12
Michael_Schumacher is class 13
Venus_Williams is class 14

Reading imagenes from  lfw5\lfw5test\

directories test read:  1
Total sum of images to test  40

List of successes/errors:
ERROR Abel_Pacheco_0001.jpg is assigned class 9 Carlos_Menem
Alejandro_Toledo_0001.jpg is assigned class 1 Alejandro_Toledo
Alejandro_Toledo_0002.jpg is assigned class 1 Alejandro_Toledo
Angelina_Jolie_0002.jpg is assigned class 2 Angelina_Jolie
Angelina_Jolie_0003.jpg is assigned class 2 Angelina_Jolie
Angelina_Jolie_0010.jpg is assigned class 2 Angelina_Jolie
Angelina_Jolie_0011.jpg is assigned class 2 Angelina_Jolie
ERROR Angelina_Jolie_0019.jpg is assigned class 13 Michael_Schumacher
Ariel_Sharon_0002.jpg is assigned class 3 Ariel_Sharon
Ariel_Sharon_0003.jpg is assigned class 3 Ariel_Sharon
Ariel_Sharon_0005.jpg is assigned class 3 Ariel_Sharon
Ariel_Sharon_0007.jpg is assigned class 3 Ariel_Sharon
Ariel_Sharon_0008.jpg is assigned class 3 Ariel_Sharon
Arnold_Schwarzenegger_0001.jpg is assigned class 4 Arnold_Schwarzenegger
Arnold_Schwarzenegger_0002.jpg is assigned class 4 Arnold_Schwarzenegger
Arnold_Schwarzenegger_0005.jpg is assigned class 4 Arnold_Schwarzenegger
Arnold_Schwarzenegger_0007.jpg is assigned class 4 Arnold_Schwarzenegger
ERROR Arnold_Schwarzenegger_0008.jpg is assigned class 13 Michael_Schumacher
Arnold_Schwarzenegger_0009.jpg is assigned class 4 Arnold_Schwarzenegger
ERROR Barbra_Streisand_0001.jpg is assigned class 8 Britney_Spears
Bill_Clinton_0001.jpg is assigned class 6 Bill_Clinton
Bill_Clinton_0005.jpg is assigned class 6 Bill_Clinton
Bill_Clinton_0007.jpg is assigned class 6 Bill_Clinton
Bill_Clinton_0008.jpg is assigned class 6 Bill_Clinton
Bill_Gates_0001.jpg is assigned class 7 Bill_Gates
Bill_Gates_0002.jpg is assigned class 7 Bill_Gates
Bill_Gates_0003.jpg is assigned class 7 Bill_Gates
Britney_Spears_0001.jpg is assigned class 8 Britney_Spears
Carlos_Menem_0002.jpg is assigned class 9 Carlos_Menem
Carlos_Menem_0003.jpg is assigned class 9 Carlos_Menem
Mahmoud_Abbas_0001.jpg is assigned class 10 Mahmoud_Abbas
Mahmoud_Abbas_0002.jpg is assigned class 10 Mahmoud_Abbas
Meryl_Streep_0002.jpg is assigned class 11 Meryl_Streep
Meryl_Streep_0008.jpg is assigned class 11 Meryl_Streep
Michael_Jackson_0001.jpg is assigned class 12 Michael_Jackson
Michael_Jackson_0002.jpg is assigned class 12 Michael_Jackson
ERROR Michael_Schumacher_0009.jpg is assigned class 4 Arnold_Schwarzenegger
ERROR Michael_Schumacher_0013.jpg is assigned class 4 Arnold_Schwarzenegger
Venus_Williams_0005.jpg is assigned class 14 Venus_Williams
Venus_Williams_0006.jpg is assigned class 14 Venus_Williams

Total hits = 34
Total failures = 6
Accuracy = 85.0%

References:

Images downloaded from LFW http://vis-ww.cs.umass.edu/lfw/#download

https://stackoverflow.com/questions/52575271/keras-vgg16-same-model-different-approach-gave-different-result

https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

https://github.com/ablanco1950/LFW_SVM_facecascade
