################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32f401retx.s 

OBJS += \
./Core/Startup/startup_stm32f401retx.o 

S_DEPS += \
./Core/Startup/startup_stm32f401retx.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/%.o: ../Core/Startup/%.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -DDEBUG -D_DSP_PRESENT=1 -c -I../Nnom/inc/layers -I../Nnom/inc -I"C:/Users/BN262210/STM32CubeIDE/workspace_1.10.0/Test_DSP/Nnom/port" -I../Middlewares/Third_Party/ARM_CMSIS/CMSIS/DSP/Include/dsp -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Core-2f-Startup

clean-Core-2f-Startup:
	-$(RM) ./Core/Startup/startup_stm32f401retx.d ./Core/Startup/startup_stm32f401retx.o

.PHONY: clean-Core-2f-Startup

