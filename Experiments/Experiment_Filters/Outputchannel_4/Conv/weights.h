#include "nnom.h"

#define CONV2D_KERNEL_0 {80, -65, -93, -91, -25, -50, 80, -55, 63, 90, 53, 52, 57, -38, 66, -46, 39, 72, 0, -20, -4, -9, -10, -20, -74, 53, 29, -93, 39, 53, 32, 46, -53, -83, 48, 82, -12, -78, 46, 55, 46, -80, 6, -55, -13, -67, 72, 78, 39, -89, 69, 17, 88, 11, 61, 91, -20, 20, 8, 24, 25, 74, 1, -3, -73, 29, -81, -45, 13, 69, 61, 16, 1, 63, -1, -58, 39, 61, 49, -27, -51, 58, -78, -64, -40, 91, 89, 65, -8, -12, 64, 3, -9, -59, 15, 52, 0, -72, -18, 66, 59, 31, 27, -39, 55, -2, 45, -48, -90, 39, -63, 57, 38, 22, 43, -10, 4, 42, -74, -33, 57, -85, 8, -3, -50, 14, 55, -40, 1, -54, 15, -38, 22, -7, 8, -22, 4, 77, 29, -10, 41, 73, -70, -44, 49, 19, 45, -41, -48, 40, -68, -30, -9, 29, -57, 65, -88, -79, -2, 18, 79, 85, -72, 45, 88, -92, 76, -83, 48, -81, -88, -8, 49, -92, 73, -85, -18, 23, 12, -71, 43, 86, 9, -70, -67, 45, -86, -76, 83, 3, -36, 42, -72, 33, -53, 67, -81, -18, -91, 34, -37, 48, 64, -27, -38, 62, -30, -89, 58, 72, 81, -93, -78, 92, 50, -64, 27, 60, -36, -62, -20, -55, -85, -11, 64, 4, 3, -68, -22, 75, 90, -72, 75, 0, -57, -68, 13, 38, -65, 12, -64, 34, -64, -60, -75, -38, -38, 43, -55, 65, -1, 11, 10, -25, -83, 82, 38, -36, -11, 13, 1, -59, -35, 72, -74, -63, 33, -82, 78, -4, 49, -44, -29, -65, -13, 88, -81, -68, 70, 82, 69, -89, 15, -3, 91, -8, 89, -70, 35, -47, -49, -51, -52, -51, 10, 62, -21, -38, 60, -35, -78, 14, 19, 79, -60, 14, 9, 59, -53, 15, 89, 23, 82, -75, 59, -42, -13, -42, 61, 60, 55, 92, -42, -61, 16, 76, -36, 33, -46, 21, 39, 47, -86, 7, -30, -89, -1, -39, 46, -9, 35, -14, 84, -34, -69, 69, -55, 32, -86, 34, 29, 74, 72, -49, -27, -34, 17, -4, 13, 56, -65, -43, -56, 27, 15, -28, 14, -21, -3, 12, -19, 63, 57, 41, -44, 56, -77, 64, 49, 3, -63, -44, -24, -12, 74, 51, 75, 18, -10, -34, 43, 78, -83, 7, -39, 14, 67, -16, 31, -4, 86, -78, 39, -17, -59, 69, 60, 34, 64, 82, 57, 91, -4, -11, 83, -63, -53, -78, -57, 50, -27, 62, 3, 31, 21, 26, -29, 15, 35, 6, -45, 20, 43, 0, -41, -26, -52, -47, -48, 80, 40, 61, 11, 37, 90, 39, 72, 32, -75, -82, -50, -15, -73, -1, -22, -58, 73, 18, -41, -92, -21, 60, -31, -69, -11, -42, -42, 51, 73, 50, 66, 71, -9, 22, -53, -53, -34, -25, 60, -11, 84, 80, 54, -49, -24, 16, -34, 38, 11, -62, 37, -18, 82, 50, 60, 59, -55, 74, -11, -53, -46, -67, -75, 9, 93, 9, -78, 33, 62, -38, 30, 29, 7, 56, 45, -78, -88, 16, 25, -6, 67, -75, -31, -28, -84, -6, 68, 69, 55, 10, -69, -54, -49, -47, 76, 44, 11, 64, 36, 6, 74, -26, -21, 21, -13, -35, 47, -68, -42, 84, 21, 80, -14, -12, 66, 1, -49, -31, 16, 19, -76, -40, 5, 90, -71, 79, 58, -26, 30, -45, -85, -65, -52, 75, 31, -72}

#define CONV2D_KERNEL_0_SHIFT (9)

#define CONV2D_BIAS_0 {51, 93, -71, -35}

#define CONV2D_BIAS_0_SHIFT (7)


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
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(Conv2D(4, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 4), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
