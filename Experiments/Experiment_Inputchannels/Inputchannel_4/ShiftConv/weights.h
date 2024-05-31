#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {-20, 52, 17, -66, -12, -50, 30, -23, -68, -19, -5, 60, 17, -59, 42, 53, 16, -42, 7, 38, 60, 20, 29, -61, -44, 50, -57, -48, -54, 36, -12, -48, 57, -59, 58, -48, -62, -38, -25, -55, -50, 22, 55, 46, -4, 26, -69, 66, 0, 22, -52, 0, 43, -6, -17, -18, 21, 3, 49, 34, -25, -19, -34, -59}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (7)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {105, -51, 103, 67, -87, 38, -76, -45, 16, -81, 98, -71, -85, 31, 109, -12}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, 1, 1, -1, 0, -1, 1, 1}


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
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 4), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
