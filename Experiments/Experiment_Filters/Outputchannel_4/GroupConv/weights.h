#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {43, 96, 98, -91, 118, 2, -32, 62, -32, -91, 73, 3, -81, 11, -99, -80, -8, 54, -7, -27, -48, -100, 114, 36, -63, 8, 22, -32, 39, -33, -105, -48, 98, 112, 107, -6, 87, 55, 18, 104, -6, 108, 86, 18, 103, 84, -14, -90, 9, -75, -110, -80, -38, 120, 15, -29, 5, -63, -99, 40, -59, -74, 93, 35, 18, 89, 14, -71, -69, 120, -30, -42, 5, 41, 117, -76, -23, 38, -67, 82, 20, 92, 76, -80, 34, -53, 10, 98, -114, 119, -34, 22, -106, -52, -87, 64, -70, -18, -41, 7, -12, 14, -105, 4, -58, 52, -30, 26, -104, -14, 92, 108, 66, 67, 114, 40, -28, 35, 88, -111, -116, 41, -70, 43, -74, 47, 104, 92, 106, -7, 59, -78, -89, 111, -26, -67, -77, 35, 96, 19, 1, 40, 23, 84, -58, -53, 16, -41, 68, 36, -7, 21, -90, 29, 75, 119, 88, 73, -74, -77, 71, -82, -109, 99, -103, -66, -105, -1, 64, 115, 81, -2, 89, 110, 120, 118, 109, 81, -11, 14, 109, 81, -83, 16, -97, -66, -2, 25, 47, 102, 112, -89, 15, -62, 2, -85, 24, -8, 61, -2, 92, 119, 0, 10, -40, 35, 38, 107, 114, 103, -70, -104, 93, -56, -93, 29, 114, 115, 59, 41, 46, 17, 108, -88, -12, -120, -8, -35, 115, -19, -66, 104, 91, 93, -13, -44, -24, 55, -15, 25, 54, -104, -120, 37, -119, 75, -87, -74, 4, -15, -12, 4, 46, 58, -40, 12, -99, 70, 92, -115, 30, 55, 28, -18, 26, -71, 99, -113, 19, -27, 73, 115, 0, 117, 104, 67, -71, -28, -51, 9, 90, -99, -120, -90, -101, -107, -30, -101}

#define GROUP_CONV2D_KERNEL_0_SHIFT (9)

#define GROUP_CONV2D_BIAS_0 {-109, 4, 98, 17}

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
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(4, 2, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 4), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
