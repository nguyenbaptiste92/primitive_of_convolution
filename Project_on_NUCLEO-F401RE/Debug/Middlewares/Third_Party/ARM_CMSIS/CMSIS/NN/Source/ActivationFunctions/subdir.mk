################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c 

OBJS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.o 

C_DEPS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/%.o Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/%.su: ../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/%.c Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -DARM_MATH_CM4 -D_DSP_PRESENT=1 -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core_A/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/PrivateInclude/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include -I../Nnom/inc/layers -I../Nnom/inc -I../Nnom/port -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ActivationFunctions

clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ActivationFunctions:
	-$(RM) ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.su

.PHONY: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-NN-2f-Source-2f-ActivationFunctions

