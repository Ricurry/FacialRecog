import tensorflow as tf
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
# Import load models
from keras.models import load_model
#GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling2D

# YOU WILL DESERVE LOVE ONLY THROUGH YOUR WORK. NOTHING ELSE.
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# REMEMBER ZIA
# TRUST YOURSELF
# TRUST NOBODY
# WORK HARD
# WORK SMART
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Use NVIDIA CUDA GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Verify GPU is being used
# Lets begin!
data_path = 'face_dataset'

# Memory growth allows the GPU to use more memory than it normally would
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


# # Of images.
num_images = len(os.listdir(data_path))
print("SYS_APOLLO ; Dataset Images : {an}".format(an = num_images))


# Define data as data folder in tf set label mode to categorical, set class names to human and not human
data = tf.keras.preprocessing.image_dataset_from_directory(data_path, label_mode='binary', class_names=['human', 'not_human'])
# Partition the train, value and test data. 70% train, 20% validation, 10% test
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
# Define the model
model = Sequential() # a sequential model is like a linear equation. so a constant rate of change. :3
# Make output shape 255x255x3
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)))

#convolution is like a treasure chest. It highlights the recurring patterns in a image :3.
# and im going to touch you too, cuz you my treasure >:)

# Add the first max pooling layer with a pool size of 2x2
model.add(MaxPooling2D())
# Add the second convolutional layer with 32 filters, 3x3 kernel size, relu activation
model.add(Conv2D(32, (3,3), activation='relu')) # x in this case for relu is the input of the image, and so if it has a positive value then it will pass on hence (x + |x|)/2 ,  max (0,x) to make sure it is above 0. 
# Add the second max pooling layer with a pool size of 2x2
model.add(MaxPooling2D())
# Add the third convolutional layer with 64 filters, 3x3 kernel size, relu activation
model.add(Conv2D(64, (3,3), activation='relu')) # 32 + 64 = 96
# Add the third max pooling layer with a pool size of 2x2
model.add(MaxPooling2D())

model.add(Conv2D(1, (1,1), activation='sigmoid')) # 96 + 1 = 97, sigmoid divides the value of significance so it is between 0 and 1.
# Logits and labels must have the same shape so we do this to make it work and retain the shape
model.add(GlobalAveragePooling2D())
# But the above makes it 2d so we need to make it 4d again

model.add(tf.keras.layers.Reshape((1,1,1))) # Reshaping into 3d instead cuz im gangster >:) this wont work for opencv but i have a idea >:))
# The above line averages out the data into a sign
# Make output shape 253x253x3
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

logdir='logs'

# Define the tensorboard callback
tensorboard_callback = TensorBoard(log_dir=logdir)

# Define the number of epochs

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback]) # epochs help make it more accurate basically the more epochs the more accurate it is. :3

# Use tf.function to get best fit for X or model.
full_model = tf.function(lambda x: model(x))
# What is a concrete function? Basically the output of full_model. 
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)) # Dtype = float32
# What is a frozen model? Frozen model is a model that optimizes the model for usage
# In most areas where some layers cant be interpreted by the model.
#Such as openCV.
# Include the function to convert variables to constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# To do these we convert all the variables into constants,
# Const variables are easier to process than normal dynamic variables because they cant be changed as easily.

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
# The line above simply sets the graph to the frozen model.

# Now that we used the line above, we can use it to specify how to write the file as a pb,
# So opencv can use it.

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="frozen_graph2.pb",
                    as_text=False)

print(model.output_shape)
# And we are done! Now we can use the model in opencv.