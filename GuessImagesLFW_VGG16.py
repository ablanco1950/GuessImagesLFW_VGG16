# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , March 2023
"""

######################################################################
# PARAMETERS
######################################################################
dirname = "lfw5\\lfw5train"
dirname_test = "lfw5\\lfw5test"

######################################################################

import os
import re

import cv2

import numpy as np


#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
    directories = []
   
    prevRoot=''
    cant=0
    
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    Y=[]
    TabNumImage=[]
    TabDenoClass=[]
    TotImages=0
    for root, dirnames, filenames in os.walk(imgpath):
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
                
                image = cv2.imread(filepath)
                                          
                images.append(image)
                if NumImage < 0:
                    NumImage=0
                Y.append(NumImage)
               
               
                TabNumImage.append(filename)
                TotImages+=1
                if prevRoot !=root:
                  
                    prevRoot=root
                    directories.append(root)
                    
                    DenoClass=filenames[0]
                    DenoClass=DenoClass[0:len(DenoClass)-9]
                    
                    
                    TabDenoClass.append(DenoClass)
    print("")
    print('directories test read: ',len(directories))
    
    print('Total sum of images to test ',str(TotImages))
    
    return images, Y, TabNumImage, TabDenoClass
 
###########################################################
# MAIN
##########################################################

from tensorflow.keras.models import load_model

model = load_model('ModelGuessImages_LFW_VGG16.h5')

X_train, Y_train, TabNumImage, TabDenoClass = loadimages (dirname)

print( "")

for i in range(len(TabDenoClass)):
    print(TabDenoClass[i]+ " is class " + str(i))
print("")

X_test, Y_test, TabNumImage_test, TabDenoClass_test = loadimages(dirname_test)

x_test=np.array(X_test)

# Scale images to the [0, 1] range
x_test = x_test.astype("float32") / 255
#print("Number de Clases = " + str(num_classes))

TotalHits=0
TotalFailures=0
predictions1=model.predict(x_test)
predictions=np.argmax(predictions1, axis=1)
print(predictions1[0])

print("")
#print(TabDenoClass_test)
print("List of successes/errors:")       
for i in range(len(x_test)):
    DenoClass=TabNumImage_test[i]
    DenoClass=DenoClass[0:len(DenoClass)-9]
    if DenoClass!=TabDenoClass[(predictions[i])]:
        TotalFailures=TotalFailures + 1
        print("ERROR " + TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
              + " " + TabDenoClass[(predictions[i])] )
              
    else:
      print(TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
          + " " + TabDenoClass[(predictions[i])])
      #print(TabNumImage_test[i]+ " is assigned class " + str(predictions[i]))
    
      TotalHits=TotalHits+1
          
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%") 

