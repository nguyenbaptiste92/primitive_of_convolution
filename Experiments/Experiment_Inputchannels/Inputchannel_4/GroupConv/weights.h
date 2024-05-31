#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {-77, -55, 91, 13, -71, 45, 20, 6, 36, -47, -62, -80, -6, 6, 93, 6, -38, 38, 66, 66, -81, 63, 87, -46, -35, -40, 14, -21, -52, -52, -97, 48, 24, -14, -4, 60, -90, -24, -53, 5, -6, -30, 34, 30, -75, 74, 56, -40, 70, -15, 95, 93, -19, -46, -10, 52, -77, -21, 4, -88, -95, 71, 20, -79, 27, -66, -34, 29, 6, -13, -11, 80, 76, 90, -89, -45, 0, 64, -36, -43, -19, -78, 87, -13, 5, -89, 39, 14, 55, 57, 67, 74, 25, -23, -28, -65, 12, -19, 86, 85, -58, 77, 15, 23, 97, -22, -1, -43, 28, 25, -10, -3, -67, 82, 71, -73, -35, 73, 40, 86, 83, -27, 32, 13, 11, 33, -96, 74, 21, -85, 6, -58, -25, 2, 27, -35, 92, 32, 14, -22, 83, -21, -2, 68, 9, -24, -30, 6, 51, 12, 82, 14, -5, 13, 33, -26, 73, 32, 55, -30, -57, 58, 37, -24, -68, 79, 54, -36, 46, -72, -8, 54, 69, 47, 58, -59, 4, -33, -32, -73, -78, -49, 18, 94, -72, -67, 5, -62, -37, 78, -98, -28, -56, 30, 74, -87, -44, 29, 50, -31, 92, -67, 27, 46, 98, -12, -94, -47, 69, 89, -63, 13, 6, -85, -13, 47, 1, 35, 2, 55, 66, -43, -3, -5, -95, 61, -60, -71, 13, -9, 16, -65, 6, -53, 74, -36, 43, 3, -25, 55, 53, -6, -71, 21, -41, -96, -11, -13, -39, -55, 0, 21, -9, 61, 51, 6, -81, -52, -39, -21, 37, -33, -10, 70, 25, 69, 21, 9, -43, -90, 72, 94, -5, -63, 41, -86, 12, -52, -87, -77, 36, 26, 29, 75, 17, -35, 77, 88}

#define GROUP_CONV2D_KERNEL_0_SHIFT (9)

#define GROUP_CONV2D_BIAS_0 {-108, -78, 61, 69, 77, 108, -32, 77, -38, -102, 100, -63, 17, -35, 22, 5}

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
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 4), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(16, 2, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
