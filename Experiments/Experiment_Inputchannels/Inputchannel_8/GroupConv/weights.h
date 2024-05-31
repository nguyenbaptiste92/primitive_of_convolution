#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {88, 80, -9, -87, 18, -6, -83, 0, -7, -60, -18, 54, -51, 58, -81, 22, 33, -60, 50, -50, 2, 10, 12, 15, -39, 55, 69, 43, -22, -47, -24, -55, 76, -70, -4, 34, 19, 83, -1, -93, 87, 0, -75, -25, -5, 28, 47, 73, 36, -7, 44, -37, -70, -15, 65, -2, -60, 0, 61, -70, -74, 11, 41, 43, -82, 31, 74, 36, -88, -61, -37, -7, -87, -27, -88, 75, -52, 8, 93, -21, -8, -93, 57, 72, -53, 49, 58, -52, -9, -38, 12, 57, 9, -1, -8, -59, 48, 71, -38, -90, -58, -12, -59, 76, 7, 41, 73, -83, 41, 10, -62, 40, 75, 6, 21, -66, -38, 59, -48, -78, -57, 13, -49, 32, 44, 24, -43, -48, -68, -12, 15, 67, 24, 81, 41, 39, 78, 13, 11, 85, -89, 62, 58, -69, -25, 38, 53, 18, 83, -19, -60, -4, -10, 12, 50, -15, 10, -70, -6, 30, 88, -2, -27, 16, 6, -85, 39, 21, 9, 10, 74, 86, -85, -9, 31, -22, 61, 18, 9, -77, -70, 85, 42, 68, -89, -15, 35, -67, 0, -70, 87, 71, -31, 77, 51, -19, 9, -10, 59, -73, 64, -12, -43, 27, -38, 83, 46, -81, -22, -86, -57, 68, 81, 34, -5, -59, -80, 76, 78, -26, 38, 18, 65, 11, -88, 87, -20, -32, 90, -54, -47, 11, 18, -71, -77, -86, -57, 44, -14, -82, 35, -65, 19, -68, -2, -26, 10, -48, -87, 75, -91, -39, 12, -69, -28, -59, 43, -7, -77, -83, 33, 22, -70, 0, -57, -10, 32, 63, 17, -76, 55, 45, 55, -53, -38, 4, 76, -40, -54, 44, -14, 76, 38, -88, -84, 10, -78, 7, 46, -12, -80, 38, 76, 1, 23, 35, -4, -65, 31, -48, -88, -26, 18, -31, -83, 90, 39, 9, -75, -80, 34, 44, 23, 92, -43, -26, -32, 84, -53, -73, 87, 71, -81, -53, 14, 83, -28, 75, 37, 52, 40, 23, 13, -31, 63, 53, -12, -12, -33, 56, -88, 93, 42, 56, 29, 17, -74, 19, 63, 87, 28, 62, -17, -40, 20, 4, 8, -80, 58, -63, 24, -50, 28, -63, -69, -89, 77, 43, 67, -53, -59, -10, 18, 20, -31, -30, 29, 84, 58, 33, -45, -51, 78, -67, 5, -28, -39, 82, 63, -85, 66, -22, -75, -91, 19, 22, -10, -37, 38, -55, 36, 71, 87, 18, -13, -55, 65, -83, 88, 92, -85, 58, 9, 81, 46, 40, -62, -7, -4, 86, 28, 73, -53, 67, -43, -81, -73, 61, 24, -15, 62, 47, -49, 38, 90, -23, -81, -1, 9, -48, -29, -79, -48, 17, 30, 46, -88, 5, 75, -42, -56, -50, -43, -42, -41, 73, -25, -93, 36, -64, 68, -80, 84, -91, 2, 69, -54, -57, -20, -31, -41, -4, 2, -9, -25, -72, -26, -82, -74, -37, -93, -75, -2, -15, -37, -84, 1, -29, -50, 73, 85, 90, -51, 20, 81, 73, -65, -37, -37, -8, 48, -72, 71, 28, 47, -23, 55, -36, -87, 34, -41, -75, 29, 76, 40, -19, -50, -83, -69, -75, 81, 5, -47, 59, -4, 77, 7, -78, -93, -23, -64, 4, 28, -46, 21, -39, 71, 47, 9, 92, -91, -54, -6, 38, -37, 39, -7, -36, 78, -14, -47, -79, -78, 71, 93, -50, 36, 3, -84, -23, -20, -22, 16, 13, -91, 15, 1, 40, -73, -22, 22, 8, -86, 43, -79, 89}

#define GROUP_CONV2D_KERNEL_0_SHIFT (9)

#define GROUP_CONV2D_BIAS_0 {-78, -107, 21, 12, 7, 85, -52, 94, -30, -97, -87, 99, -46, -45, -37, -47}

#define GROUP_CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define GROUP_CONV2D_OUTPUT_SHIFT 6

/* bias shift and output shift for each layer */
#define GROUP_CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+GROUP_CONV2D_KERNEL_0_SHIFT-GROUP_CONV2D_OUTPUT_SHIFT)
#define GROUP_CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+GROUP_CONV2D_KERNEL_0_SHIFT-GROUP_CONV2D_BIAS_0_SHIFT)
#if GROUP_CONV2D_OUTPUT_RSHIFT < 0
#error GROUP_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if GROUP_CONV2D_BIAS_LSHIFT < 0
#error GROUP_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t group_conv2d_weights[] = GROUP_CONV2D_KERNEL_0;
static const nnom_weight_t group_conv2d_w = { (const void*)group_conv2d_weights, GROUP_CONV2D_OUTPUT_RSHIFT};
static const int8_t group_conv2d_bias[] = GROUP_CONV2D_BIAS_0;
static const nnom_bias_t group_conv2d_b = { (const void*)group_conv2d_bias, GROUP_CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[8192];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 8), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(16, 2, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
