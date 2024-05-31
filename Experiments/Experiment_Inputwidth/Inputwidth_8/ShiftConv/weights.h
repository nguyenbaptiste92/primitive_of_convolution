#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {-103, 34, 68, 7, -68, -3, -89, 28, -100, -72, 40, 96, -69, -87, -15, -17, -60, -87, 62, -59, -15, -68, -96, 43, 76, -86, -54, 21, -33, 56, 53, 102, 75, -85, 24, -91, -92, -29, 2, 17, -18, -82, 98, 37, 86, 43, -80, -70, 15, 97, 63, 58, 37, 87, -24, 61, 35, 32, 104, 12, -16, 78, -47, -48, 47, -8, -70, -13, 31, 87, -21, -50, -71, -89, 23, -80, 78, 95, -16, -9, -20, -47, 89, 104, 66, -102, 17, 80, -22, 21, -100, 19, -49, -10, -104, 62, 0, 73, -34, -44, 76, -23, -16, -17, 110, 96, 71, 67, -6, 48, 21, -27, -16, -80, 109, 0, -60, 44, -90, -58, 25, -46, 34, 105, 54, 102, 3, 97, -75, 37, 106, 23, -90, 98, -79, 33, 49, 63, 104, 34, 98, 30, 69, 81, -27, -57, 63, 21, 59, 65, 19, 65, -41, -40, 76, -45, 108, 73, -101, 81, 63, -85, -76, 6, 20, -34, 38, 38, 3, 71, -104, -80, 0, -8, -97, -103, -17, -38, 18, -16, 42, 48, -88, -44, -41, -70, 51, 37, -25, 1, -60, 9, -63, 3, 15, 7, 29, -38, 47, 29, -51, -44, 64, 103, -69, -44, 104, 76, 82, -60, 40, -75, 71, -38, -22, 71, -62, -31, -15, 29, -109, 37, 55, -67, 97, -11, 61, 80, 68, -101, -34, 28, 65, -9, 44, -43, -41, -66, -73, -96, -100, -61, 71, 107, -18, -52, -87, 30, -56, 34, -20, -44, 44, -5, -3, -21}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-110, 9, -98, 13, -9, -47, -97, -3, -19, -109, -98, -7, -100, 51, -49, -33}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {-1, 0, 0, 0, 0, -1, 0, 1, 1, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 0, -1, -1, -1, 0}


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
static int8_t nnom_input_data[1024];
static int8_t nnom_output_data[1024];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(8, 8, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(16, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(8, 8, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
