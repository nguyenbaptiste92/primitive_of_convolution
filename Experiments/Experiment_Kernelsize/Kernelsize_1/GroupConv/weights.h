#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {-77, -63, -72, 10, 87, 59, 37, -66, 116, 69, -31, -18, 68, 83, 30, -88, 42, -82, -26, 127, -101, 27, 72, -8, 95, -29, 108, -33, -122, -24, -114, 104, -105, -55, 27, -86, -100, -14, 55, -45, 33, 15, -31, 73, -90, -32, 19, -89, -31, 30, 77, -96, 57, -122, -80, 35, 80, 88, 33, 112, 77, 30, -125, -82, -97, -75, 59, 118, -120, -94, 0, 126, 107, 43, 126, 112, 118, -121, 105, -18, -4, -78, 40, 67, -13, 26, -122, -81, -9, 17, -85, 66, -61, -14, -39, 113, 70, -6, 66, 71, 51, 3, 64, 74, 83, -22, 111, 106, -66, -25, -60, -106, -23, 113, -90, 60, -27, 119, 37, -48, -6, 62, 29, 23, -17, -6, 67, -97}

#define GROUP_CONV2D_KERNEL_0_SHIFT (8)

#define GROUP_CONV2D_BIAS_0 {92, 108, -96, -98, 70, -33, -109, 89, 15, 94, -55, -77, -69, -87, 106, 72}

#define GROUP_CONV2D_BIAS_0_SHIFT (8)


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
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(16, 2, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
