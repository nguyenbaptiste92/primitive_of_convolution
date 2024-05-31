#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {4, 27, -9, -32, 65, -56, -67, 60, -15, -5, 44, -19, -3, 18, 34, -13, -34, 37, -69, 33, 39, -5, -22, 48, -40, -15, 41, -41, 13, -27, 43, -32, 47, -67, -8, 36, -17, -8, -18, 14, -13, 33, 39, -51, -27, -29, -7, -35}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (7)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-109, 2, -55, 16, -48, -64, 5, 94, -19, -39, -67, 76, 7, -47, 63, 101}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, 0, -1, 0, 0, 0}


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ACTIVE_SHIFT_CONV2D_OUTPUT_SHIFT 7

/* bias shift and output shift for each layer */
#define ACTIVE_SHIFT_CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT-ACTIVE_SHIFT_CONV2D_OUTPUT_SHIFT)
#define ACTIVE_SHIFT_CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT-ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT)
#if ACTIVE_SHIFT_CONV2D_OUTPUT_RSHIFT < 0
#error ACTIVE_SHIFT_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if ACTIVE_SHIFT_CONV2D_BIAS_LSHIFT < 0
#error ACTIVE_SHIFT_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t active_shift_conv2d_weights[] = ACTIVE_SHIFT_CONV2D_KERNEL_0;
static const nnom_weight_t active_shift_conv2d_w = { (const void*)active_shift_conv2d_weights, ACTIVE_SHIFT_CONV2D_OUTPUT_RSHIFT};
static const int8_t active_shift_conv2d_bias[] = ACTIVE_SHIFT_CONV2D_BIAS_0;
static const nnom_bias_t active_shift_conv2d_b = { (const void*)active_shift_conv2d_bias, ACTIVE_SHIFT_CONV2D_BIAS_LSHIFT};
static const int8_t active_shift_conv2d_shift[] = ACTIVE_SHIFT_CONV2D_SHIFT_0;
static const nnom_shift_t active_shift_conv2d_s = { (const void*)active_shift_conv2d_shift};

/* nnom model */
static int8_t nnom_input_data[3072];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 3), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
