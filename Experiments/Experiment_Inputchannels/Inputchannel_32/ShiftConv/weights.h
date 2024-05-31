#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {10, -3, -59, -9, 68, 13, 87, -60, -48, 34, 82, 55, 32, -84, -65, -46, 43, 90, -20, 83, -67, -63, 73, -10, -27, 22, -85, 56, 82, 44, -13, -41, -88, 58, -83, -33, -39, 35, 26, -21, 68, 65, 87, 81, -73, -70, -37, 52, 70, 58, 90, 36, 6, 7, 23, -27, 85, -88, 32, 6, -62, 44, 12, -7, 82, 37, -24, -69, 41, 64, 15, -43, -21, 74, -24, -87, -11, -1, 24, 82, -81, 1, 68, -74, 54, -32, 74, -13, 32, 34, -44, 12, -64, -69, 63, 55, 63, 13, -48, 31, 75, -15, 70, -75, -83, 80, 81, -2, -25, 40, 43, -55, 63, 30, -38, -59, 32, 84, -86, -58, 6, -14, -20, -37, -65, -49, -82, 62, -50, -84, -69, 88, -22, -28, 22, 44, -43, -12, 50, 60, 70, -24, -47, 66, 7, 73, 14, 63, 11, -76, 60, -82, 50, -82, -71, -70, -41, -19, 2, 85, -74, 31, -64, -63, 30, -7, 6, -71, -66, -69, 20, 41, -46, 49, -53, -38, 28, -40, -79, 55, -50, 70, 82, 84, 57, 51, -69, 28, -46, -15, 5, -31, 47, 25, 81, -39, -66, -62, 1, -54, -7, 65, -40, -82, 82, -6, 30, 47, -11, 1, -3, -43, -88, -27, 72, -87, -74, 78, -62, -62, 15, 43, 31, -48, 67, -20, 48, -42, 26, 78, 55, -35, 61, 52, -73, 58, 45, -43, -16, -49, 58, -88, -64, 76, 32, -40, -34, -27, -87, -8, 30, -10, -90, 37, -87, 78, 32, 38, 21, 42, -34, 6, 23, 45, -6, 16, -46, 44, 45, 73, 13, 19, 75, -69, -54, 25, 40, -52, 64, -85, 16, -21, 74, 61, 50, 5, 12, -67, -1, -5, -72, -13, -88, -15, -80, -90, 29, -25, -59, -59, 68, -34, -31, 90, 26, 36, 16, 78, 69, 28, -8, -66, -63, 71, 68, -73, -71, -3, 80, -4, -42, -4, 61, 20, -71, 55, 37, -61, 20, 49, -50, 72, 86, 51, -73, -3, -33, -54, -80, -1, 83, -4, 24, -81, -42, 0, 47, 39, 77, -16, 63, 71, -50, 60, -14, -20, 60, 78, -73, -54, -83, 7, 71, 24, -61, -7, -15, 0, -4, 78, 80, 0, 47, 85, 75, -77, -28, 37, -58, -69, -1, 30, 32, -67, -80, 80, 80, -42, 10, -13, 58, -4, -80, -25, 89, -54, -24, 37, -66, 49, -46, -55, -25, -80, 48, -78, 50, 64, 17, -21, -54, 35, 86, -15, 69, -32, -36, 85, 47, -32, -1, 88, -22, 49, 49, -41, -78, 71, 65, -51, 74, 41, -51, 15, 75, -71, -36, 39, 25, -74, 73, -76, -19, -24, -40, -39, -38, -17, 48, -45, 50, 37, -69, 51, -1, 71, 77, -32, -44, -61, -13, 40, 6, 16, -53, 28, -55, -5, 72, -88, -40, -72, 47, 6, -53, -75, 4, -62, -8, 53, -22, 74, 12, -61, 58, 50, -49, -14, -6, -83, 66, -11, -34, -67, -78, 23, 41, -69, -15, -64, 30, -74, -18, -76, 76, -8, 58, -45, 15, 23, -83, 84}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-86, -51, 54, 13, -106, 44, 96, -105, -23, -13, -63, 101, -73, 55, -110, 41}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {1, -1, 0, 1, 0, 1, -1, 1, -1, 0, -1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0, 1, -1}


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
static int8_t nnom_input_data[32768];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 32), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
