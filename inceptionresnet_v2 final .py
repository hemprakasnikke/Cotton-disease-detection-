#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.resnet import ResNet152
#from tensorflow.keras.applications.resnet import preprocess_input
#from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[4]:


IMAGE_SIZE = [224, 224]


# In[5]:


train_path = 'cotton dataset\train'
valid_path = 'cotton dataset\val'
test_path = 'cotton dataset\test'


# In[8]:


#mport tensorflow
inception = InceptionResNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#resnet152V2 =tensorflow.keras.applications.ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#resnet = ResNet152(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[9]:


for layer in inception.layers:

    layer.trainable = False


# In[10]:


folders = glob(r'cotton dataset/train/*')


# In[11]:


print(len(folders))


# In[12]:


x = Flatten()(inception.output)


# In[13]:


prediction = Dense(len(folders), activation='softmax')(x)
print(prediction)
model = Model(inputs=inception.input, outputs=prediction)


# In[14]:


model.summary()


# In[15]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[16]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[17]:


training_set = train_datagen.flow_from_directory(r'cotton dataset\train',
                                            target_size=(224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[18]:


val_set = train_datagen.flow_from_directory(r'cotton dataset\val',
                                            target_size=(224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[19]:


test_set = test_datagen.flow_from_directory(r'cotton dataset\test',
                                            target_size=(224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[20]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',
        mode='min', verbose=1, patience=5)


# In[21]:


r = model.fit_generator(training_set,
 validation_data= val_set,
 epochs=50,
 steps_per_epoch= len(training_set),
 validation_steps = len(val_set),
 callbacks=[early_stop],)


# In[25]:


scores = model.evaluate(training_set)


# In[22]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss_InceptionResnetV2')
# plot the accuracy for 30 epochs
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc_InceptionResnetV2')


# In[26]:


from tensorflow.keras.models import load_model
model.save('model_InceprionResnetV2.h5')


# In[27]:


y_pred = model.predict(test_set)


# In[28]:


print(y_pred)


# In[29]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[30]:


y_pred


# In[31]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[32]:


model=load_model('model_InceprionResnetV2.h5')


# In[33]:


from sklearn.metrics import classification_report
y_pred = model.predict(test_set)
# y_pred = model.predict(training_set)
y_pred_bool = np.argmax(y_pred, axis=1)


# In[34]:


y_pred_bool


# In[35]:


train_path = 'cotton dataset\train'
valid_path = 'cotton dataset\val'
test_path = 'cotton dataset\test'


# In[36]:


import os
import cv2

train_path = r'cotton dataset\train'  # Adjust the path syntax

x_train = []

for folder in os.listdir(train_path):
    sub_path = os.path.join(train_path, folder)
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)


# In[37]:


import os
import cv2

test_path = r'cotton dataset\test'  # Adjust the path syntax

x_test = []

for folder in os.listdir(test_path):
    sub_path = os.path.join(test_path, folder)
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_test.append(img_arr)


# In[38]:


import os
import cv2

val_path = r'cotton dataset\val'  # Adjust the path syntax

x_val = []

for folder in os.listdir(val_path):
    sub_path = os.path.join(val_path, folder)
    for img in os.listdir(sub_path):
        image_path = os.path.join(sub_path, img)
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_val.append(img_arr)


# In[39]:


train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)
train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0


# In[40]:


train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes


# In[43]:


from sklearn.metrics import accuracy_score,classification_report
import numpy as np
#predict
#y_pred=model.predict(test_x)
#y_pred=np.argmax(y_pred,axis=1)
#get classification report
#print(classification_report(test_y, y_pred))

from sklearn.metrics import classification_report

# Assuming `test_y` and `y_pred` are the true labels and predicted labels respectively

# Remove class 4 from `test_y` and `y_pred`
filtered_test_y = np.delete(test_y, np.where(test_y == 4))
filtered_y_pred = np.delete(y_pred, np.where(test_y == 4))

# Generate classification report without class 4
report = classification_report(filtered_test_y, filtered_y_pred)

print(report)


# In[45]:


from sklearn.metrics import precision_score
precision_score(test_y, y_pred, average='weighted', labels=np.unique(y_pred))


# In[46]:


from sklearn.metrics import recall_score
recall_score(test_y, y_pred, average='weighted', labels=np.unique(y_pred))


# In[52]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, y_pred)


# In[53]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, y_pred)

# Creating a DataFrame for an array-formatted Confusion matrix
cm_df = pd.DataFrame(cm,
                     index=['Disease Leaf', 'Disease Plant', 'Fresh Leaf', 'Fresh Plant', ' '],
                     columns=['Disease Leaf', 'Disease Plant', 'Fresh Leaf', 'Fresh Plant', ' '])


plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[54]:


y_pred=model.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)
y_pred


# In[55]:


from sklearn.metrics import roc_curve,roc_auc_score
fpr , tpr , thresholds = roc_curve(test_y, y_pred, pos_label=_)


# In[56]:


fpr


# In[57]:


tpr


# In[58]:


thresholds


# In[59]:


def plot_roc_curve(fpr,tpr):
 plt.plot(fpr,tpr)
 plt.axis([0,1,0,1])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.show() 
 
plot_roc_curve (fpr,tpr)


# In[ ]:




