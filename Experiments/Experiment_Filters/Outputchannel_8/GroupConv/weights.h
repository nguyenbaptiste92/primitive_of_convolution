#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {67, 4, -50, -38, -101, -64, 83, 56, 54, -90, 65, 19, 51, -65, -71, 37, -50, 31, 74, 28, 54, -56, 12, 69, 40, 33, -96, 94, -21, 64, -84, -77, -80, 41, -30, 27, 55, 39, -49, -67, 68, -34, 23, -2, 64, 19, -3, -95, -97, 94, -90, 27, 4, -81, 71, -98, 83, -95, 3, 57, 23, 7, 19, -30, -20, -3, 42, -70, -100, 78, -23, 38, 27, -12, 23, -59, -39, 78, -98, -71, -30, -7, -33, -19, 53, -48, -84, -59, 85, 23, 47, -58, 2, 93, 59, 95, -11, 14, 22, 86, -54, -58, 70, 20, 56, -51, 38, 104, -50, -85, 51, -84, 54, -102, 56, -33, -67, 80, 99, -39, -89, -85, 60, -3, -36, 36, 66, 67, -94, -22, -78, 98, 42, 99, -52, 85, -35, 101, -77, 33, 36, -42, -98, 18, 87, 41, -12, -17, -22, 75, -49, -85, -65, 38, 67, -35, -26, -13, -42, 15, 99, 42, 55, 98, -76, -45, -104, 67, 0, -20, 53, -96, -3, -90, 81, 35, -44, -1, -39, -84, 24, 91, -44, -21, 42, 29, -35, -8, -44, 67, -43, -12, -58, -4, 89, 97, -80, -54, 95, 102, 21, -54, -101, -40, 95, -25, -37, 79, -58, -20, -32, 75, 64, 57, -42, 41, 72, 31, -89, -29, 93, 67, 0, 32, 14, -16, -58, -40, 0, -86, -31, 102, 87, 98, -15, 13, -35, 92, 87, -19, -34, -67, -104, -57, 27, -76, 94, -34, -26, -74, -32, 32, -6, -39, -69, 28, 59, -7, -58, 36, 25, 3, 102, 79, -83, 11, 7, 8, 34, -86, -75, -39, 4, 101, 16, 80, -19, -74, -43, 31, 63, 18, 37, -20, 86, -9, -100, -48, 78, -10, 7, 92, 59, 10, 40, 53, 29, -64, -24, 12, -9, 72, -69, -51, -72, 10, 45, -67, 64, 26, 5, 49, -64, -55, -47, -55, -57, 58, 58, -104, -102, -18, -65, 37, -25, 79, -60, -79, 31, 39, 38, 50, 82, -25, -33, 10, 60, -69, -18, 36, 57, 85, 16, 48, 43, -63, 76, -101, -76, 45, 5, 11, -82, 72, -103, -51, 89, -42, 100, 104, 6, 30, 29, 96, 42, 31, 16, -22, 24, -103, 25, -50, 77, 67, -78, -73, -89, 48, 97, 38, 94, -100, -48, 6, -30, -64, -92, 22, -28, 96, -25, -41, -88, -12, 5, 85, 103, 62, 12, 96, 104, 42, 49, -12, 79, 67, -80, -84, 70, -95, 75, 16, 18, 96, 45, 91, -9, -32, -67, 79, -22, -100, -7, 14, -23, -54, 3, -46, -8, 51, 81, -63, 60, 47, 50, 52, 82, -22, -40, 28, -49, 32, -99, -67, -24, 47, -11, 81, 48, 12, -19, -26, 48, 73, 103, -20, 85, 4, 32, -86, -45, 3, 81, 64, -47, 80, -87, -64, -36, 36, 14, -40, 44, 100, 98, 43, -48, -73, -25, 83, -54, -32, 0, -9, -52, -22, -101, -19, 91, -16, -60, -79, 94, 64, 57, -92, 27, -45, -8, -1, -36, -87, -39, -28, 62, 81, -54, 9, 58, 56, -37, 41, 69, -57, 63, -98, 63, -34, 28, -92, 92, -72, 44, 3, -80, 58, 1, -14, 34, -90, 69, 62, -85, 23, -8, -74, 31, -104, 80, -62, -21, -73, -70, 21, 23, 22, 5, 47, 45, 22, -98, -95, -41, 98, 14, -31, -39, 57, 103, 32, 69, 38, 75, -18, -91, -83, 68, -48, -37, -58, -17, -39, 70, 14, 76, 17}

#define GROUP_CONV2D_KERNEL_0_SHIFT (9)

#define GROUP_CONV2D_BIAS_0 {-49, -15, -44, -53, 77, 64, -66, -4}

#define GROUP_CONV2D_BIAS_0_SHIFT (7)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define GROUP_CONV2D_OUTPUT_SHIFT 5

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
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[8192];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(8, 2, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 8), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
