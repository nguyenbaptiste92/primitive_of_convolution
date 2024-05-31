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
from lib.layers.group_conv import GroupConv1D
from lib.layers.activeshiftconv import ActiveShiftConv1D

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



def Conv1DModel(inputs_shape=[150,6],kernel_size=3,filters=4,padding="same"):

    inputs = tf.keras.Input(shape=inputs_shape)
    conv1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,activation='linear',padding=padding,use_bias=True,bias_initializer='glorot_uniform')
    
    x = conv1(inputs)
    
    return tf.keras.Model(inputs={"input":inputs}, outputs={"label":x})
    
def GroupConv1DModel(inputs_shape=[150,6],groups=2,kernel_size=3,filters=4,padding="same"):
    
    inputs= tf.keras.Input(shape=inputs_shape)
    conv1 = GroupConv1D(filters=filters,kernel_size=kernel_size,activation='linear',padding=padding,use_bias=True,bias_initializer='glorot_uniform',groups=groups)
    
    x = conv1(inputs)
    
    return tf.keras.Model(inputs={"input":inputs}, outputs={"label":x})
    
def DepthConv1DModel(inputs_shape=[150,6],kernel_size=3,padding="same"):

    inputs = tf.keras.Input(shape=inputs_shape)
    depth1 =tf.keras.layers.DepthwiseConv1D(kernel_size = kernel_size,activation='linear',padding=padding,use_bias=True,bias_initializer='glorot_uniform')
    
    x = depth1(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
def DepthSeparableConv1DModel(inputs_shape=[150,6],kernel_size=3,filters=4,padding="same"):

    inputs = tf.keras.Input(shape=inputs_shape)
    depth1 =tf.keras.layers.DepthwiseConv1D(kernel_size = kernel_size,activation='linear',padding=padding,use_bias=True,bias_initializer='glorot_uniform')
    conv1 = tf.keras.layers.Conv1D(filters=filters,kernel_size=1,activation='linear',padding='same',use_bias=True,bias_initializer='glorot_uniform')
    
    x = depth1(inputs)
    x = conv1(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
def ShiftConv1DModel(inputs_shape=[150,10],filters=4,strides=1):
    
    inputs= tf.keras.Input(shape=inputs_shape)
    shift1 = ActiveShiftConv1D(filters=filters, strides=strides, shift_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), use_bias=True, bias_initializer='glorot_uniform')
   
    
    x = shift1(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
    
##############################################################################################################
#Load dataset
##############################################################################################################   
max_samples = 1000
number_samples = 1
input_size = [50,16]
groups = 2
kernel_size = 3
filters = 16
padding = "valid"
strides = 1

liste_image = tf.random.uniform(shape=[max_samples, input_size[0], input_size[1]], minval=0., maxval=1.)

##############################################################################################################
#Train model
#############################################################################################################   
model = GroupConv1DModel(inputs_shape=input_size,groups=groups,kernel_size=kernel_size,filters=filters,padding=padding)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model(liste_image[0:1],training=True)

print(model.layers[1].kernel.shape)

##############################################################################################################
#Generate neural network .h files
############################################################################################################## 

"""generate_model(model, liste_image.numpy(), name='weights.h', format='hwc', quantize_method='max_min',batch_norm=True)


##############################################################################################################
#Generate images file (.h)
##############################################################################################################   

generate_test_h(np.copy(liste_image),number_samples, name='test_data.h')

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
tensorflow_y = model(liste_image[0:number_samples],training=False)['label']

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

with open('test_shift.txt') as f:
    line = f.readline()
    line = line.split("\n")[0]
    shift = int(line.split(" ")[0])
    
print("kernel:",model.layers[1].kernel)
print("bias:",model.layers[1].bias)
print("input:",liste_image[0:number_samples])


output_array = output_array/(2**shift)
print(tensorflow_y)
print(output_array)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
#Compare output for single layer
a = np.transpose(np.reshape(tensorflow_y.numpy(),[number_samples,-1]))
b = np.transpose(np.reshape(np.array(output_array),[number_samples,-1]))
print("L1-NORM")
print(np.sum(np.abs(a-b)))"""
"""print(a.shape,b.shape)
mse = mean_squared_error(a, b,multioutput="raw_values")
print("MSE")
print(mse.shape)
print(np.mean(mse),np.std(mse))
rmse = np.sqrt(mse)
print("RMSE")
print(rmse.shape)
print(np.mean(rmse),np.std(rmse))
"""


#Accuracy for complete model
#label = np.argmax(y_test[0:number_samples], axis=1)
#tensorflow_result = np.argmax(tensorflow_y, axis=1)
#output_result = np.argmax(output_array, axis=1)
#tensorflow_acc = np.sum(tensorflow_result == label).astype('float32')/int(tensorflow_result.shape[0])
#output_acc = np.sum(output_result == label).astype('float32')/int(tensorflow_result.shape[0])
#print("tensorflow accuracy : ",tensorflow_acc)
#print("output accuracy : ",output_acc)




