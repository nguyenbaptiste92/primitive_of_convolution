# Evaluation of Convolution Primitives for Embedded Neural Networks on 32-bits Microcontroller

Project to evaluate the latency and consumption of different convolution primitives on ARM Cortex-M microcontrollers. Our implementations of the different primitives are integrated into the [NNoM library](https://github.com/majianjia/nnom) (in the Scripts/nnom_micro folder). To use SIMD instructions, the [middleware CMSIS-NN](https://arm-software.github.io/CMSIS_5/NN/html/index.html) is required.

[Paper accepted at the 22nd International Conference on Intelligent Systems Design and Applications](https://link.springer.com/chapter/10.1007/978-3-031-27440-4_41).

Copy of the [official git repository](https://gitlab.emse.fr/b.nguyen/primitive_of_convolution).

## Environment
See `environment.txt`.

## Convolution primitives

* Addconvolution from [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_AdderNet_Do_We_Really_Need_Multiplications_in_Deep_Learning_CVPR_2020_paper.pdf) : the code from the [official repository](https://github.com/huawei-noah/AdderNet) is adpated in tensorflow.
* Shiftconvolution from [Constructing Fast Network through Deconstruction of Convolution](https://proceedings.neurips.cc/paper/2018/file/9719a00ed0c5709d80dfef33795dcef3-Paper.pdf) : implementation taken from the [official repository](https://github.com/jyh2986/Active-Shift-TF).
* Grouped convolution from [Deep Roots:
Improving CNN Efficiency with Hierarchical Filter Groups](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ioannou_Deep_Roots_Improving_CVPR_2017_paper.pdf) : implementation taken from https://medium.com/@krzechowski/custom-group-convolution-for-tensorflow-2-fc74a83189ce .

## Folder organisation

Our project includes several folders:

* Scripts: folder with the python files (tensorflow layer and NNoM scripts) and c libraries (modified NNoM, ARM CMSIS-NN).
* Project_on_NUCLEO-F401RE: project's folder on NUCLEO-F401RE on which the consumption measurement are done.
* Experiments: folder with the experiments data on csv files (Due to the size of the data, we ignored them in this git, to get access, you can send me an email at nguyen.baptiste92@yahoo.fr or go on the [official git repository](https://gitlab.emse.fr/b.nguyen/primitive_of_convolution)).
* Results: folder with python scripts which process the data from the Experiments folder to create the various figures and tables of the article.

## Update

* Implementation of the 1D version of the Convolution, the Depthwise convolution, the Grouped convolution and the Shift convolution (algorithme naive + algorithme Im2Col with SIMD instructions) used in the chapter 4 of my [thesis manuscript](https://theses.fr/s263650).

## TO DO

* Implementation of left-over case (no impact on the study to the choice of layer's dimensions) : 
	- nnom_local.c : Convolution (ligne 697) et Depthwise-Convolution (ligne 1155)
    - nnom_local_shiftconvolution.c : ligne 224
	- nnom_local_groupconvolution.c : ligne 407, 486, 753
	- nnom_local_addconvolution.c : ligne 819


