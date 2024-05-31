#include "nnom.h"

#define CONV2D_KERNEL_0 {21, -39, 52, -33, 59, 39, -37, 104, -70, 100, -88, -30, -48, 92, 58, 84, 83, 85, -104, 80, 2, -102, -82, -63, -68, 83, -93, 31, -21, -75, 92, -12, -111, 50, 105, 79, 5, -74, -48, 92, 100, 83, -101, 105, -96, 76, 54, -2, -38, 26, 78, -48, -39, 29, 52, -4, -84, 58, -103, -67, -92, -15, 61, 49, 86, -86, 75, -4, 7, -57, -110, -22, -79, 70, 17, -50, -79, -11, 87, -35, -69, 46, 100, -10, -65, 38, -94, 41, 66, -7, -39, -42, -18, -104, -77, -13, -12, 93, 1, -85, -27, 0, -89, 15, 67, 15, -41, -16, -64, 21, 75, -109, -54, -47, -15, 56, -1, -36, -88, 90, 9, -93, -49, -42, 89, -82, -43, -81, -26, 15, -97, 76, -27, -39, -27, 47, -110, 38, 37, -59, 95, -106, -92, -30, -68, 84, -37, -55, -92, 87, -94, 86, -18, -109, -21, -39, 34, -40, 69, 70, 1, 2, 77, -104, 27, 94, -68, -92, -32, 47, 44, 69, -108, -39, 46, 93, -26, 10, 44, -27, -28, 56, -64, -106, 7, -106, 92, -21, -61, 57, -30, 102, 46, 72, -45, 0, -81, -75, -41, 57, 3, 19, -5, -87, -1, 92, 21, -9, -109, 70, 33, 16, -89, -47, -24, -37, 66, -3, -49, 29, 48, 33, -64, -108, 12, 74, 103, 52, 35, 55, 21, 97, -30, -61, -106, 91, -4, -58, -98, -71, 32, 65, -64, 23, -71, -50, -102, 90, 39, 13, 67, 24, 1, -39, 25, -74}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-30, -89, 2, 84, -71, -108, -24, 24, 4, -74, -90, -52, -3, -17, -23, 92}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 5

/* bias shift and output shift for each layer */
#define CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_OUTPUT_SHIFT)
#define CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_BIAS_0_SHIFT)
#if CONV2D_OUTPUT_RSHIFT < 0
#error CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if CONV2D_BIAS_LSHIFT < 0
#error CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t conv2d_weights[] = CONV2D_KERNEL_0;
static const nnom_weight_t conv2d_w = { (const void*)conv2d_weights, CONV2D_OUTPUT_RSHIFT};
static const int8_t conv2d_bias[] = CONV2D_BIAS_0;
static const nnom_bias_t conv2d_b = { (const void*)conv2d_bias, CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
