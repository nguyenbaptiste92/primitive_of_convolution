#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {12, -44, -23, 27, 6, -29, 70, -13, 100, -109, 19, -15, -26, -47, -83, 46, -48, -83, 42, -74, 96, -30, -53, -35, 94, 82, 56, -47, 39, 34, -42, -99, 95, 31, 43, -101, 71, -48, -81, -101, -89, -73, 61, 85, 33, -5, 103, -92, 85, 4, -84, -23, -21, -3, 105, 16, 92, -69, -103, 90, 57, 106, -40, 22, -49, -21, 77, -99, 49, -39, -13, -72, 100, -60, -15, 61, -79, -41, 82, 27, -48, -80, -48, 63, 99, -16, 30, 33, 36, -108, -68, 30, 97, 11, -44, -105, -110, 100, 34, 29, 29, -96, -23, -50, 64, -8, -93, 14, -88, -94, 108, -76, 90, -84, 74, 32, 11, 74, -88, 66, 59, 10, 90, -108, 59, 50, -31, 23, -27, 52, -56, -106, 13, 1, -102, -60, -71, -56, 43, -24, 73, 92, -54, 4, 40, -87, 54, -34, 103, 88, -76, -21, -27, -27, -3, 5, -93, -7, 64, 110, 76, 44, -92, 1, -7, 57, 74, 101, -16, -41, 4, 75, -97, 18, -64, -1, -77, 25, 55, 48, 45, -84, 83, -46, 13, 25, 31, -88, 21, 109, -49, 81, -34, -18, -56, -28, 34, -9, -47, -32, 85, -33, 64, -8, -83, -74, 78, -88, 38, 14, -104, 105, 59, 85, 46, 19, -33, -33, -92, 90, -9, 23, 94, -23, 108, 35, 25, 9, 21, -71, -65, 25, -88, -103, -29, 17, 35, -83, -16, 55, 49, -22, 39, 57, -104, 95, -94, 19, -96, 43, -37, 11, 87, -25, 80, -47}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {103, 51, 88, -78, -96, -86, -93, -20, 58, -42, 15, 41, -62, -32, 58, 13}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {0, 0, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0}


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
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
