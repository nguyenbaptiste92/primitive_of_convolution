#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {105, 47, 52, -98, -51, 54, -38, 83, -38, -20, 25, -18, -62, 50, 106, -79, -67, -100, 70, -83, 83, -78, -88, 94, -62, -38, 56, 23, -47, -75, -77, 44, -6, -30, -78, 59, -85, -76, -90, -111, 16, 89, 59, -95, 9, 58, -85, 13, 0, 44, 71, -27, 64, 3, 26, 101, -80, 73, 108, 42, 25, 20, -99, -38, -89, 104, 73, -69, -50, 70, -111, 56, -14, -40, 91, 22, 94, 1, -72, -108, -57, -23, -66, 85, 20, -44, -8, 95, -19, 64, -47, 15, 32, 93, -37, 88, -61, 42, -85, 100, -44, 51, -65, -88, -44, -89, -17, 52, -98, -75, 97, -99, 64, 6, -10, 24, -93, 15, -72, 78, 16, 31, -66, 74, 108, -95, -33, -6, -76, -16, -100, 104, -74, -51, -52, 73, 101, 109, -55, -3, -73, -48, 86, -106, -76, 65, -7, 70, 36, -109, -51, 69, 15, -49, -35, -92, -69, 67, -110, -97, -4, 47, 39, -36, -40, -44, -41, 41, -57, -97, -28, -47, 86, 61, -38, -79, 58, -49, 106, 51, 15, -92, 93, 29, -79, -7, 104, -11, 60, -23, 51, 86, -8, 34, 109, -29, -26, 19, 0, 88, 104, -93, 65, 85, -35, 26, 21, 15, 105, 111, 74, 27, 59, 93, 47, 18, -87, 44, -110, 98, 104, 28, 55, 11, 92, -105, -13, 21, 102, 20, 71, 83, -64, -6, -17, 70, -56, -92, 1, 8, -32, 101, -10, 20, 109, 73, -107, -52, -94, 108, 50, -56, 18, 92, -1, -33}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-9, 12, -89, 28, 77, 39, 56, 88, 47, 92, -81, 47, -60, 35, 107, 52}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, 0, 1, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0, 1, -1, 1, 0, 1, -1, -1, 0, 0, 0, 1, 1}


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
