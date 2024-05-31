#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {-103, -74, 3, 101, 78, -90, -89, -59, 80, 50, -5, -14, -67, 78, 33, -15, -6, 41, -80, -91, 83, -57, -26, -37, -8, -93, 19, -15, -51, 66, 82, 68, -18, -50, -6, 33, 31, -109, -80, -65, 97, -65, 52, -87, 4, 8, 68, -24, -54, 9, 52, -62, -105, -20, -75, -51, -36, -101, -35, 38, -59, -105, 10, 5, 57, -61, -55, 7, 84, -55, 35, 57, 94, -79, 48, 48, 23, -76, -45, -40, 30, -4, -44, -87, -74, 21, 84, 74, 71, 95, 2, 62, 69, 107, -76, 5, -59, -68, -65, -31, 57, -16, 90, 92, -64, -43, -93, 80, 41, -78, 74, 107, 100, -55, 16, -52, -53, 29, -59, -102, 54, -20, 10, 14, -35, 107, -87, -88, -60, 29, 29, -37, 20, 75, -101, -39, 12, -52, -56, 9, 21, -108, 95, 78, -1, -14, 58, 37, 97, -9, -76, 93, -14, 53, 85, -27, 32, -65, 58, 37, -18, 97, 9, 47, 19, -64, -12, -45, 40, -104, 8, -77, 9, 66, -71, -17, 90, -9, -90, -3, -26, -7, 69, 36, -47, 12, 22, 7, 26, -32, -31, 85, -17, 78, -66, -107, -107, 46, 62, 9, -13, -65, 53, 51, 75, 9, 104, -3, 8, 22, -75, 16, -63, -25, 86, -40, -47, 47, -79, 94, -65, 15, 81, 69, 67, -11, -6, -107, -79, -102, 42, -74, -57, 28, 49, 2, 3, -100, 20, 44, 40, -21, -88, 88, -96, 91, 18, 93, -4, -90, 87, 11, -105, 96, 74, 24}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {78, 45, 33, -43, -81, 13, 52, 42, -108, -105, 65, -46, -54, -57, -61, -109}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 1, -1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1}


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
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(16, 16, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(16, 16, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
