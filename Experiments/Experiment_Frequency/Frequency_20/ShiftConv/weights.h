#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {58, -47, 0, 14, 83, -79, -30, -14, 101, -5, 15, -30, 89, 97, 63, 50, 32, -16, 56, 81, 99, 75, -78, -76, -46, -67, 45, -53, -48, 19, 78, -110, -63, -102, 86, 27, 97, -26, -59, 42, -17, -95, -69, 77, 22, 61, -30, -76, 5, 85, -62, 84, 79, -13, 32, -51, 106, -82, 77, -62, -99, 90, -1, -2, -13, 104, 77, -28, 1, -20, -36, -33, -71, -9, -1, 78, 63, -38, 30, -90, -110, -108, 25, -23, 44, 46, -80, -85, -93, -101, -1, -66, -100, -54, -27, 8, -4, 48, 47, -31, -37, 24, -23, 37, -70, 17, 96, -11, 92, -15, 58, 11, -31, -9, -62, 61, -53, -45, 63, 97, -32, 93, -12, 18, 94, -35, 53, 51, -99, 99, 10, -46, 78, -31, -43, -69, -2, 40, 87, -25, 45, 58, -44, 6, -9, -107, 98, -97, -1, -77, 77, 32, 102, 56, 57, 29, 50, 69, -107, 56, 25, 24, 69, -17, -4, 87, -18, 0, -103, 35, 65, -14, -43, -69, -78, -95, -61, 87, 8, 11, -77, -80, -1, 60, -11, -86, 74, -53, 35, 72, -101, -35, -79, 38, -78, -9, 61, 34, -78, 98, -31, 57, 81, -79, -74, 3, 48, 7, -47, 107, -15, -60, -27, -91, 88, -111, 6, -64, 50, 16, -5, 34, -47, 68, -61, -43, 32, -101, -91, 39, 86, 76, 8, -39, 45, -22, 55, -88, -72, 39, -48, -96, -37, -13, 28, -66, 9, -82, 105, 48, -87, 49, 77, 53, 57, -23}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {71, -61, -106, 8, -37, -43, 76, 43, -51, -80, 57, -29, 20, 59, 29, 25}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {0, 0, -1, -1, 0, 0, 0, 1, 1, -1, 0, -1, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 1, 0, 0, 0}


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
