#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {56, -81, -32, -31, -2, -16, -4, -40, -43, 20, -60, -37, 83, -63, 28, 50, 13, -64, -24, -37, 36, -82, 32, -84, 33, 44, -12, -7, -61, -76, 4, 4, -46, 55, -2, -10, 4, 25, 13, 60, -4, -5, 48, -14, -27, -26, -86, -50, 46, -69, 40, -48, -2, 22, -11, -82, -42, -39, -49, -77, 31, 64, -74, 62, -67, 72, -60, -55, -66, -9, 63, -70, 19, 5, -19, 89, 18, 81, -42, -82, 10, 55, -36, 3, -40, -5, -33, 70, 19, -65, -21, -4, 35, 80, -19, -53, 68, 80, 50, -22, -74, -28, 20, 25, -85, -24, 47, -80, 15, 33, 57, -20, -35, 82, 76, -35, -59, -3, 4, -8, 40, -50, -51, -18, 65, -76, -18, 20, 56, 1, 22, 89, 58, 80, 50, 8, -85, -44, 56, -41, -20, -25, -91, -32, -38, -48, 86, -9, -67, 51, 40, -53, 49, 68, -28, 31, -8, 64, -65, -82, 70, -27, 88, 34, 51, -7, -21, -14, -12, -62, 20, 50, 89, 27, -83, 64, 80, 18, -46, 76, -71, -39, -57, 15, 15, -12, 70, -27, 54, 3, 32, -17, 84, -37, -12, -26, 62, -58, 79, -11, -51, -16, 63, 14, 38, -34, 24, 82, 50, -21, -31, 28, 17, 7, -49, 67, -29, 26, -23, -61, 5, 21, 78, 75, -42, 77, 38, 66, -8, 39, 23, -67, -59, 8, -61, -9, 14, -43, -58, -83, -47, 82, 0, 66, -21, 47, 1, -63, -90, -29, 38, 74, -15, 10, -85, 28, 54, -73, 79, -89, -17, 17, 13, 41, 9, -43, -75, -10, 40, 88, -52, 55, 54, -29, 34, 79, 3, -59, 66, -87, -35, 36, 32, 20, -4, -53, 2, 28, 65, -67, -76, 78, 13, 76, -42, 81, -47, -74, 65, 31, 72, 50, 51, 47, 3, 90, 58, -14, 5, -52, -63, 35, -19, -3, -89, 22, -18, 51, -66, -28, 76, 10, -14, 65, 34, -24, 9, -71, 77, 9, 1, -35, 6, -29, -39, 56, 58, 4, 11, 53, -37, -24, -17, 85, 87, -11, -88, -32, -75, -68, 34, 26, 90, -39, 82, -33, 9, 31, -22, 77, -83, -83, 50, 63, -87, 59, 10, 51, 19, -16, 50, 6, -28, -47, 19, 4, 9, 55, 59, 88, 46, 58, 59, 63, 29, -40, 22, 0, -62, 84, 14, -16, 54, 65, 58, 17, 7, 55, -33, -33, 57, 58, -20, 31, 41, -63, -71, -43, -20, 39, 3, 38, 14, 20, -36, 58, 78, 68, 88, 22, -1, -89, 41, -64, 31, -43, -43, -14, -65, 41, -29, 50, 44, -24, -33, 59, 11, 38, 36, 87, -4, -88, 21, 14, -39, -69, 25, -74, 63, 90, -18, -58, -62, -28, -27, -72, -19, -14, 87, 28, -18, 1, 12, 54, -40, 78, 59, -42, 42, -62, -82, 43, -66, 79, 24, 63, -72, -13, -84, 64, -17, -34, 2, 2, 78, -61, 1, -53, 55, 2, 72, 76, 64, -5, 31, -80, 80, 46, 39, 82, -55, -27, 76, 59, 36, -43, -63, -29, -14, -54, -90, -60}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-53, -67, 12, -10, -62, -76, 14, 39, -42, 14, -45, 32, 11, 12, 25, 43, 63, 56, -35, 73, -18, -72, 47, -61, -35, 5, -74, -69, 59, 27, 15, 25}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, -1, 0, -1, 0, 1, 0, 1, -1, 0, 1, -1, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, 1, -1, 1}


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
static int8_t nnom_output_data[32768];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(32, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 32), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
