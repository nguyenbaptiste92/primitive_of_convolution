################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.c \
../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.c 

OBJS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.o \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.o 

C_DEPS += \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.d \
./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/%.o Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/%.su: ../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/%.c Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -DARM_MATH_CM4 -D_DSP_PRESENT=1 -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Include -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/Core_A/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/PrivateInclude/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/ -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include -I../Nnom/inc/layers -I../Nnom/inc -I../Nnom/port -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-DSP-2f-Source-2f-InterpolationFunctions

clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-DSP-2f-Source-2f-InterpolationFunctions:
	-$(RM) ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctions.su ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.d ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.o ./Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.su

.PHONY: clean-Middlewares-2f-Third_Party-2f-ARM_CMSIS-2f-CMSIS-2f-DSP-2f-Source-2f-InterpolationFunctions

