#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {-27, 29, 8, 66, -8, -66, -46, -4, -29, 51, -44, 11, 53, 63, 67, -24, 62, -24, 58, -45, -28, -17, 53, 26, 21, 0, -41, -45, 26, 49, 69, -53, -53, -1, -54, -49, 48, -9, -40, 40, -32, -3, 56, -35, 42, 49, 56, -18, -12, -34, -48, 15, 7, -46, 31, 31, -12, 6, 28, -50, -49, -68, -40, 2}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (7)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-57, -90, -15, 70}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (7)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, -1, 0, -1, 0, -1, 0, -1, 1, 0, 0}


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ACTIVE_SHIFT_CONV2D_OUTPUT_SHIFT 6

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
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(4, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 4), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
