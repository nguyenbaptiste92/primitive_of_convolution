#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {-8, 26, 13, -70, -16, 23, -23, 108, -104, -7, -7, 84, 111, 112, -109, 63, -104, -53, 127, -53, -67, -16, -45, -7, 91, -37, -109, 51, 95, -89, 44, -104, -52, 110, -121, -69, -49, -65, -81, 33, 13, 32, -107, 108, -5, -42, -16, 93, -44, -40, 116, 78, -65, -70, -78, -56, -103, -95, -94, 97, -26, 87, -103, 67, -73, 32, -120, -51, 41, 25, -28, -126, -42, 126, 112, 122, -47, 74, 108, -73, 102, -3, 98, -47, 83, 1, 100, -74, 89, -13, 47, -25, 54, -16, -76, 98, -73, -117, -24, 68, 120, -47, -48, 116, -35, -126, 98, 76, -16, -126, -51, 31, 115, 107, -20, 11, 1, 29, -57, 84, -77, -31, -70, -56, -49, -109, -39, 49}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-33, 83, -71, 57, -22, -7, -59, -81, 33, -46, 20, 48, -86, 46, -88, 45}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, -1}


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
static int8_t nnom_input_data[8192];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 8), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
