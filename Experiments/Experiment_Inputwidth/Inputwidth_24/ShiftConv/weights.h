#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {42, -10, -107, -107, -73, 9, 3, 95, -36, -56, -57, 103, 5, -37, 15, 72, -107, 12, 44, -53, 103, 29, 4, 28, -72, 76, -49, 40, 106, -97, -93, -35, 105, 14, -89, -101, 94, -15, -33, 38, 85, 82, -62, 17, -35, -19, -14, -67, 97, 35, 84, 17, 20, -44, -51, -110, -28, -89, 111, 13, 97, 14, -39, -71, 47, 87, 4, 57, -73, 20, -6, -87, 14, 63, -14, 106, 37, -97, -69, -73, 60, -109, -32, -36, 89, -96, 73, 50, 26, 20, 85, -97, 13, 40, 42, 6, 26, 86, -10, 60, 95, 100, 49, 54, 30, -9, 49, -69, 63, 104, 75, -62, 75, -74, 18, 66, -109, 73, -18, -84, 42, -83, -66, 40, 99, 3, -57, -51, 90, 56, -60, -37, -14, -85, -92, -9, -64, -15, 94, 58, 97, 46, -97, -110, 78, -74, -99, -11, -56, 82, -11, 26, 92, -67, 75, 54, 92, 8, 111, 51, -73, -85, -63, 55, 103, 10, 5, 53, 60, -91, 2, -38, 61, 110, -69, 3, -68, 104, 5, 46, 32, 14, 48, -24, -75, -47, 52, 3, 57, 93, 6, -37, 50, -21, 96, 94, -1, 11, -11, 88, -54, -3, 79, 2, 17, 89, -5, -71, 18, 54, -94, 37, -28, -99, 72, -11, -41, 82, -28, -15, 53, 87, -105, -94, 35, 13, -6, -95, -98, 109, -65, 98, -83, -108, -84, 83, 0, 27, -43, -85, -78, -62, 44, 97, -13, 110, 97, -86, 79, -41, 47, -31, -78, 63, 53, 31}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {89, -22, -43, 67, -99, 30, 14, 93, -75, 104, 110, -60, 5, 1, -16, -111}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {0, -1, 1, -1, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, -1, -1, 1, 0, 0, 1, -1, 1, 0, 0, 0, 1, -1, 0}


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ACTIVE_SHIFT_CONV2D_OUTPUT_SHIFT 5

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
static int8_t nnom_input_data[9216];
static int8_t nnom_output_data[9216];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(24, 24, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(24, 24, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
