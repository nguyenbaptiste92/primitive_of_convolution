################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c 

OBJS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.o 

C_DEPS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/%.o Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/%.su: ../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/%.c Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -DARM_MATH_CM4 -D_DSP_PRESENT=1 -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core_A/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/PrivateInclude/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include -I../Nnom/inc/layers -I../Nnom/inc -I../Nnom/port -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ConvolutionFunctions

clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ConvolutionFunctions:
	-$(RM) ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.su
	-$(RM) ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.su

.PHONY: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ConvolutionFunctions

