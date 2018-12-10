
# coding: utf-8

# In[1]:


NB_EPOCH  = 1
VALID_SPLIT       = 0.05
BATCH_SIZE        = 100
image_size = 56


# In[2]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
import os  
seq2point_weight_bin = 'alex-net_v27'
bin_checkpointer = ModelCheckpoint(
    filepath=os.path.join('.', '%s-weights.hdf5'%seq2point_weight_bin), 
    verbose=1, 
    save_best_only=True)

bin_earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)


# In[3]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet_model(n_classes=2364, l2_reg=0.,
    weights=None):

    # Initialize model
    input_shape = (image_size, image_size, 1)
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=input_shape,
        padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(384, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(384, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet


# In[4]:


alex_model = alexnet_model()
alex_model.summary()


# In[5]:


from keras import losses
from keras.optimizers import rmsprop
alex_model.compile(loss='categorical_crossentropy',
    optimizer=rmsprop(lr=0.0001, decay=1e-6),
    metrics=['accuracy'])


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

from keras.models import load_model
model = load_model('alex-net_v26-weights.hdf5')
model.summary()


# In[7]:


for j in range(0,5):
    for i in range(0,1) :
        data = pd.read_csv('/mnt/b/train/train_data'+str(i)+'.csv')
        labels = pd.read_csv("./foo.csv")
#         data = pd.read_csv('./here.csv')
#         labels = pd.read_csv("./label.csv")
        images = data.iloc[:, :].values
        images = images.astype(np.float)
        images = np.multiply(images, 1.0 / 255.0)
        image_size = images.shape[1]
        image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
        images = images.reshape(-1, image_width, image_height, 1)
      #  print(images.shape)
        
        labels_flat = labels.iloc[:,0].values.ravel()
     #   print(labels_flat)
  #      print(labels_flat.shape)
        labels = np_utils.to_categorical(labels_flat, 2364)
        #labels = dense_to_one_hot(labels_flat, 1120)
        labels = labels.astype(np.uint8)
        print(images.shape)
        print(labels.shape)
        model.fit(images, labels, 
                    callbacks=[bin_checkpointer, bin_earlystopper],
                    epochs=NB_EPOCH,
                    validation_split=VALID_SPLIT,
                    batch_size=BATCH_SIZE)



