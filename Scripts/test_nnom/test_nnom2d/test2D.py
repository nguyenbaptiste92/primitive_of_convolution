import sys
import os
#sys.path.append(os.path.abspath("../lib/nnom"))
sys.path.append(os.path.abspath("../../"))
print(sys.path)

import tensorflow as tf
import os

import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import *
#from loaddigits import * # A modifier
from lib.nnom.nnom_utils import * # A modifier
from lib.layers.activeshiftconv import ActiveShiftConv2D

print(tf.config.list_physical_devices('GPU'))

def generate_test_h(x,number_of_sample=1,name='test_data.h'):
    '''
    this method generate the
    :param x:  input x data size
    :return:
    '''
    # quantize input x
    min_value = np.min(x)
    max_value = np.max(x)
    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits = 7 - int_bits
    x = np.round(x*2**dec_bits).astype(np.int8)
    x = x[0:number_of_sample]

    # get data
    print(x.shape)
    x = np.reshape(x,(x.shape[0],-1))
    print(x.shape)
    with open(name, "wb+") as f:
        f.write(b"char liste_image[")
        f.write(str.encode(str(x.shape[0])))
        f.write(b"][")
        f.write(str.encode(str(x.shape[1])))
        f.write(b"]={{")
        np.savetxt(f, x, delimiter=',', newline='},\n{',fmt='%d')
        f.seek(0,2)                 # end of file
        size=f.tell()               # the size...
        f.truncate(size-3)
        f.write(b"};")



def Conv2DModel(inputs_shape=[32,32,3],kernel_size=3,filters=4,padding="same"):

    inputs = tf.keras.Input(shape=inputs_shape)
    conv1 = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,activation='linear',padding=padding,use_bias=True,bias_initializer='glorot_uniform')
    
    x = conv1(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
def ShiftConv2DModel(inputs_shape=[32,32,3],filters=4):
    
    inputs= tf.keras.Input(shape=inputs_shape)
    shift1 = ActiveShiftConv2D(filters=filters,shift_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),use_bias=True,bias_initializer='glorot_uniform')
   
    
    x = shift1(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
    
##############################################################################################################
#Load dataset
##############################################################################################################   
max_samples = 1000
number_samples = 512
input_size = [32,32,3]
groups = 2
kernel_size = 3
filters = 16
padding = "valid"

liste_image = tf.random.uniform(shape=[max_samples, input_size[0], input_size[1], input_size[2]], minval=0., maxval=1.)

##############################################################################################################
#Train model
#############################################################################################################   
model = ShiftConv2DModel(inputs_shape=input_size,filters=filters)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model(liste_image[0:512],training=True)

##############################################################################################################
#Generate neural network .h files
############################################################################################################## 

generate_model(model, liste_image.numpy(), name='weights.h', format='hwc', quantize_method='max_min',batch_norm=True)

##############################################################################################################
#Generate images file (.h)
##############################################################################################################   

"""generate_test_h(np.copy(liste_image),number_samples, name='test_data.h')

##############################################################################################################
#Compile .c file and do the inference on the .h file image with .h file model
##############################################################################################################  
# build NNoM
os.system("scons")

# do inference using NNoM
cmd = ".\test_nnom.exe" if 'win' in sys.platform else "./test_nnom"
os.system(cmd)

##############################################################################################################
#Comparaison between tensorflow and c
##############################################################################################################  
#Compare c with tensorflow output
tensorflow_y = model(liste_image[0:number_samples],training=False)
print(tensorflow_y.shape)

output_list = []
with open('test_result.txt') as f:
    line = f.readline()
    while line:
        line = line.split("\n")[0]
        output = line.split(" ")[:-1]
        output_list.append(output)
        line = f.readline()

       
output_list = [[int(elem) for elem in output] for output in output_list]
output_array = np.reshape(np.array(output_list),tensorflow_y.shape)
print(output_array.shape)

with open('test_shift.txt') as f:
    line = f.readline()
    line = line.split("\n")[0]
    shift = int(line.split(" ")[0])

output_array = output_array/(2**shift)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
#Compare output for single layer
a = np.transpose(np.reshape(tensorflow_y.numpy(),[number_samples,-1]))
b = np.transpose(np.reshape(np.array(output_array),[number_samples,-1]))
print("L1-NORM")
print(np.sum(np.abs(a-b)))"""
