################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Nnom/src/backends/nnom_local.c \
../Nnom/src/backends/nnom_local_addconv2d.c \
../Nnom/src/backends/nnom_local_batchnormalization.c \
../Nnom/src/backends/nnom_local_groupconv2d.c \
../Nnom/src/backends/nnom_local_q15.c \
../Nnom/src/backends/nnom_local_shiftconv2d.c 

O_SRCS += \
../Nnom/src/backends/nnom_local.o \
../Nnom/src/backends/nnom_local_addconv2d.o \
../Nnom/src/backends/nnom_local_batchnormalization.o \
../Nnom/src/backends/nnom_local_groupconv2d.o \
../Nnom/src/backends/nnom_local_q15.o \
../Nnom/src/backends/nnom_local_shiftconv2d.o 

OBJS += \
./Nnom/src/backends/nnom_local.o \
./Nnom/src/backends/nnom_local_addconv2d.o \
./Nnom/src/backends/nnom_local_batchnormalization.o \
./Nnom/src/backends/nnom_local_groupconv2d.o \
./Nnom/src/backends/nnom_local_q15.o \
./Nnom/src/backends/nnom_local_shiftconv2d.o 

C_DEPS += \
./Nnom/src/backends/nnom_local.d \
./Nnom/src/backends/nnom_local_addconv2d.d \
./Nnom/src/backends/nnom_local_batchnormalization.d \
./Nnom/src/backends/nnom_local_groupconv2d.d \
./Nnom/src/backends/nnom_local_q15.d \
./Nnom/src/backends/nnom_local_shiftconv2d.d 


# Each subdirectory must supply rules for building sources it contributes
Nnom/src/backends/%.o Nnom/src/backends/%.su: ../Nnom/src/backends/%.c Nnom/src/backends/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -DARM_MATH_CM4 -D_DSP_PRESENT=1 -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core_A/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/PrivateInclude/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include -I../Nnom/inc/layers -I../Nnom/inc -I../Nnom/port -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Nnom-2f-src-2f-backends

clean-Nnom-2f-src-2f-backends:
	-$(RM) ./Nnom/src/backends/nnom_local.d ./Nnom/src/backends/nnom_local.o ./Nnom/src/backends/nnom_local.su ./Nnom/src/backends/nnom_local_addconv2d.d ./Nnom/src/backends/nnom_local_addconv2d.o ./Nnom/src/backends/nnom_local_addconv2d.su ./Nnom/src/backends/nnom_local_batchnormalization.d ./Nnom/src/backends/nnom_local_batchnormalization.o ./Nnom/src/backends/nnom_local_batchnormalization.su ./Nnom/src/backends/nnom_local_groupconv2d.d ./Nnom/src/backends/nnom_local_groupconv2d.o ./Nnom/src/backends/nnom_local_groupconv2d.su ./Nnom/src/backends/nnom_local_q15.d ./Nnom/src/backends/nnom_local_q15.o ./Nnom/src/backends/nnom_local_q15.su ./Nnom/src/backends/nnom_local_shiftconv2d.d ./Nnom/src/backends/nnom_local_shiftconv2d.o ./Nnom/src/backends/nnom_local_shiftconv2d.su

.PHONY: clean-Nnom-2f-src-2f-backends

