## Folder organization

* ARM_CMSIS : ARM_CMSIS middleware (5.8.0) with CMSIS-NN (3.0.0) need for using SIMD instructions (DSP)

* test_nnom : contain the python script (test1D.py,test2D.py) to test the NNoM layers. They generate a .h file containing the model/layer informations (weights.h) and a .h file containing a randomized input (test_data.h) then the inference on MCU is simulated (with SConstruct and test_nnom.c) on computer (only without using SIMD instructions).

* lib : python files with the convolution primitives' implementation on tensorflow and modified NNoM scripts (nnom_utils.py) to generate a .h file containing the model/layer informations.

* nnom_micro : modified NNoM middleware (Embedded C) with the additional grouped convolution (1D,2D), add convolution (2D), shift convolution (1D,2D) and batch normalization layers. 

## Other files

Third_party folder, build.sh and shrc enable us to use cuda 11.0. They are not necessary for other people.
