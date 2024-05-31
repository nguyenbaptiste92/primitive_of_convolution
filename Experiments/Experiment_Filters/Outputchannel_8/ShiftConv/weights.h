#include "nnom.h"

#define ACTIVE_SHIFT_CONV2D_KERNEL_0 {115, -116, 15, -78, 60, 60, 93, -10, 7, -117, -77, -20, 56, 70, -55, 90, 111, 93, -84, -87, 99, 14, 87, -109, 88, -123, -31, -43, 35, -12, 54, 13, 46, -126, 44, 120, -63, -126, -40, 119, -79, -86, 103, -18, 2, 91, -127, 37, 78, 51, -88, -5, -74, -27, -74, 38, 120, -82, 95, 79, -11, -28, 99, 112, 113, 56, -25, 105, 39, -18, -76, -121, -30, -45, -32, 122, 104, -43, 8, -86, -49, 57, 30, 55, -60, -92, -9, -99, -22, 126, 109, -48, 13, 19, -66, -75, 50, -8, -73, 76, -34, -15, -21, 108, -32, -120, 72, 98, -70, 29, -78, -48, -108, 108, 13, 118, 33, 124, -48, -95, 89, -93, 77, 18, 110, -88, 4, 66}

#define ACTIVE_SHIFT_CONV2D_KERNEL_0_SHIFT (8)

#define ACTIVE_SHIFT_CONV2D_BIAS_0 {-38, 8, 22, 72, -18, -56, 17, -40}

#define ACTIVE_SHIFT_CONV2D_BIAS_0_SHIFT (7)

#define ACTIVE_SHIFT_CONV2D_SHIFT_0 {-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 1, -1, -1, 0, 1, 0, 0, -1, 0}


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
static int8_t nnom_output_data[8192];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(ShiftConv2D(8, &active_shift_conv2d_w, &active_shift_conv2d_b, &active_shift_conv2d_s), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 8), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
