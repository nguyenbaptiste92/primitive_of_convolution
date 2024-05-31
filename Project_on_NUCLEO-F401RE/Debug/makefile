################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Nnom/src/layers/subdir.mk
-include Nnom/src/core/subdir.mk
-include Nnom/src/backends/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/SoftmaxFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/SVDFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ReshapeFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/PoolingFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/NNSupportFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/FullyConnectedFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConvolutionFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ConcatenationFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/BasicMathFunctions/subdir.mk
-include Middlewares/Third_Party/ARM_CMSIS/CMSIS/NN/Source/ActivationFunctions/subdir.mk
-include Drivers/STM32F4xx_HAL_Driver/Src/subdir.mk
-include Core/Startup/subdir.mk
-include Core/Src/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := Test_DSP
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
Test_DSP.elf \

MAP_FILES += \
Test_DSP.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
Test_DSP.list \


# All Target
all: main-build

# Main-build Target
main-build: Test_DSP.elf secondary-outputs

# Tool invocations
Test_DSP.elf Test_DSP.map: $(OBJS) $(USER_OBJS) C:\Users\BN262210\STM32CubeIDE\workspace_1.10.0\Test_DSP\STM32F401RETX_FLASH.ld makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-gcc -o "Test_DSP.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m4 -T"C:\Users\BN262210\STM32CubeIDE\workspace_1.10.0\Test_DSP\STM32F401RETX_FLASH.ld" --specs=nosys.specs -Wl,-Map="Test_DSP.map" -Wl,--gc-sections -static --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

Test_DSP.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "Test_DSP.list"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) Test_DSP.elf Test_DSP.list Test_DSP.map default.size.stdout
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified

-include ../makefile.targets
