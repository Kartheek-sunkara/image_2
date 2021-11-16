#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator


# In[5]:


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[6]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[7]:


classifier.add(Flatten())


# In[13]:


classifier.add(Dense(128, activation = 'relu'))


# In[16]:


classifier.add(Dense(1, activation = 'sigmoid'))


# In[17]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics ='accuracy')


# In[53]:


train_datagen = ImageDataGenerator(rescale=1./255, 
     shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\New folder (2)\training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\New folder (2)\training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
classifier.fit_generator(train_set, steps_per_epoch=8000/32, epochs=25, validation_data=test_set, validation_steps=2000/32)


# In[54]:


classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[88]:


import numpy as np
from keras.preprocessing import image
#test_image = image.load_img(r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\New folder (2)\training_set\poles\Utility_pole_in_Michigan-scaled.jpg', target_size = (64, 64))
#test_image = image.load_img(r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\New folder (2)\training_set\animal\cat.104.jpg', target_size = (64, 64))
test_image = image.load_img(r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\test_samples\13.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
output = result[0][0]
if output == 1.0:
    print("Image contains pole")
else:
    print("No pole in image")


# In[95]:


for i in range(13):
    number = i+1
    print(number)
    #path = r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\test_samples\poles\{}.jpg'.format(number)
    path = r'C:\Users\ssc\Desktop\Sharath\image_classification_poc\test_samples\no_pole\{}.jpg'.format(number)
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    output = result[0][0]
    if output == 1.0:
        print("Image contains pole")
    else:
        print("No pole in image")


# In[92]:


from keras.models import model_from_json


# In[94]:


# serialize model to JSON
model_json = classifier.to_json()
with open("model_v1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model_v1.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_v1.h5")
print("Loaded model from disk")


# In[ ]:





# In[42]:


test_image = image.img_to_array(test_image)


# In[43]:


test_image = np.expand_dims(test_image, axis = 0)


# In[44]:


result = classifier.predict(test_image)


# In[33]:


train_set.class_indices


# In[45]:


result


# In[103]:


import glob
import os.path

folder_path = r'C:\\Users\\ssc\\Desktop\\Sharath\\image_classification_poc\\model_v1\\static\\uploads\\'
file_type = '.jpg'
files = glob.glob(folder_path + file_type)
files=["C:\\Users\\ssc\\Desktop\\Sharath\\image_classification_poc\\model_v1\\static\\uploads\\1.jpg"]
print(files)
max_file = max(files, key=os.path.getctime)

print (max_file)


# In[105]:


import os
 
dir = 'C:\\Users\\ssc\\Desktop\\Sharath\\image_classification_poc\\model_v1\\static\\uploads\\'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))


# In[ ]:




