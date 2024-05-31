################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Nnom/src/core/nnom.c \
../Nnom/src/core/nnom_layers.c \
../Nnom/src/core/nnom_tensor.c \
../Nnom/src/core/nnom_utils.c 

O_SRCS += \
../Nnom/src/core/nnom.o \
../Nnom/src/core/nnom_layers.o \
../Nnom/src/core/nnom_tensor.o \
../Nnom/src/core/nnom_utils.o 

OBJS += \
./Nnom/src/core/nnom.o \
./Nnom/src/core/nnom_layers.o \
./Nnom/src/core/nnom_tensor.o \
./Nnom/src/core/nnom_utils.o 

C_DEPS += \
./Nnom/src/core/nnom.d \
./Nnom/src/core/nnom_layers.d \
./Nnom/src/core/nnom_tensor.d \
./Nnom/src/core/nnom_utils.d 


# Each subdirectory must supply rules for building sources it contributes
Nnom/src/core/%.o Nnom/src/core/%.su: ../Nnom/src/core/%.c Nnom/src/core/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -DARM_MATH_CM4 -D_DSP_PRESENT=1 -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core_A/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/PrivateInclude/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include -I../Nnom/inc/layers -I../Nnom/inc -I../Nnom/port -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Nnom-2f-src-2f-core

clean-Nnom-2f-src-2f-core:
	-$(RM) ./Nnom/src/core/nnom.d ./Nnom/src/core/nnom.o ./Nnom/src/core/nnom.su ./Nnom/src/core/nnom_layers.d ./Nnom/src/core/nnom_layers.o ./Nnom/src/core/nnom_layers.su ./Nnom/src/core/nnom_tensor.d ./Nnom/src/core/nnom_tensor.o ./Nnom/src/core/nnom_tensor.su ./Nnom/src/core/nnom_utils.d ./Nnom/src/core/nnom_utils.o ./Nnom/src/core/nnom_utils.su

.PHONY: clean-Nnom-2f-src-2f-core

