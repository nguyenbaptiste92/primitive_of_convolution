#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {33, 4, 24, 80, 49, 8, 62, 18, 62, -60, 62, -56, -62, 72, 29, -82, -19, 67, 4, -51, 69, 73, -7, -8, -81, 83, 69, 58, 31, -7, 25, -36, -38, -82, -27, 79, -18, -6, 70, -44, 37, 21, 79, 32, -73, -52, -85, 40, 61, -64, -83, -54, 29, 17, 40, 57, -52, 18, -20, -37, 27, 13, 78, -25, -29, -34, -25, 38, 67, 54, 35, 60, -15, 26, 83, -13, 39, -79, -9, 83, -31, -42, 32, -12, 55, 28, -72, 38, -43, -11, -24, 50, 34, -37, -37, 32, 28, 26, 53, -83, 4, -80, -25, 46, -72, 81, -48, -22, 53, -63, 45, 79, 69, -56, 15, 67, 2, -9, -34, -27, -39, -24, -41, 31, 3, 63, 79, 16, -24, -61, -66, -2, 72, -27, -67, 22, 56, 73, -50, -33, -27, 70, -11, 6, -5, -60, 21, -21, -25, 44, -19, -58, 47, -74, -54, -52, 60, -5, -74, 62, 10, -60, -68, 60, -68, 75, -72, -65, 52, 29, 69, -13, -63, 79, 67, -35, -72, -25, -40, 11, -34, 51, 26, 13, -21, 77, 35, -3, -7, -20, 78, 77, 59, -20, 83, -43, -68, 84, -30, -81, -36, -82, 47, 66, -2, 40, -78, -6, -18, -53, 85, -25, -42, -73, -64, -38, -30, 13, 24, 59, 81, -14, 84, 60, -33, 78, -42, 75, 43, -43, -67, 42, 41, 33, 15, -51, -42, 58, -66, -61, -29, 60, 48, 1, -7, 27, -13, -83, -60, 43, 66, -19, 77, -30, 23, 62, 58, 30, -78, 58, -14, -48, -20, 59, -63, -54, 78, 84, -18, 69, -32, 73, 4, -29, -42, 69, -16, -16, 34, -19, -79, 37, -23, 38, 71, 20, 6, 11, 8, -45, 41, 24, 4, 31, 32, 60, -11, 47, 75, -5, 57, -66, 14, -6, 71, 25, -28, -31, -53, 74, 42, 60, 71, -37, -65, 5, -17, 5, -8, -61, -34, -80, -37, -25, -53, -56, -19, 18, 74, -44, 65, 51, -1, 61, 64, -39, 66, 12, -57, -75, 56, -84, -26, -21, 79, -70, 65, 49, 27, 21, 82, 82, -66, 19, 14, -19, 40, -48, -10, 38, 53, 27, 53, 36, 31, -12, 37, 14, -77, -47, 68, 38, -25, 77, 47, 83, 71, -40, 40, 8, -42, -6, -54, -72, 76, 5, -47, 31, -2, 61, -68, -50, 12, -51, -10, -51, 11, -38, 32, 67, 8, -66, 20, -35, 40, 78, 4, 48, 22, 27, 78, -17, 50, 44, -83, -69, 32, -31, -46, 21, 12, 41, 61, 15, -18, -45, -3, 11, 13, 61, 38, 32, -21, 71, 70, -11, 62, 8, 54, -9, -67, 20, -17, 39, 66, -18, 64, 60, -49, 84, 34, -72, 29, -83, 47, -46, 57, -13, 1, 70, 51, -8, -78, -7, -2, 10, 27, 59, -7, -81, 5, 65, -59, -32, 46, -63, 38, -53, 60, 33, -65, -50, -63, -73, 8, 27, 21, -31, -31, 19, -23, 7, -61, 6, -57, 9, 4, 8, 84, -63, 64, 15, -3, 42, -8, 83, -77, 38, -53, 61, 83, -40, 83, -69, 2, -40, 34, 8, -13, -54, 0, -65, 54, -47, -56, -33, -5, 22, 74, 65, -74, -27, 74, 9, -38, 67, -52, 40, 10, -54, 10, -32, 72, 78, -42, 42, 9, 72, 74, 13, -53, -74, -41, 60, -74, 70, 83, -47, -4, 77, -79, -48, -68, -31, 33, 80, -62, 31, 5, 24, 11, -20, -34, 36, 36, 23, 52, -13, -42, -14, 46, 42, -54, 66, 21, 85, 76, -22, 72, 83, 43, 71, -53, -84, 38, -4, -62, -21, -50, 41, 81, 60, -64, 4, -62, 70, -8, 38, 78, 75, 60, -54, 75, 65, -79, 12, 44, 85, -8, -9, 57, -29, -11, 54, -17, 44, 47, 40, 3, 61, 14, -62, -17, -5, -26, 39, -66, -60, -40, 78, -37, 14, -1, -16, 41, 20, -10, 71, 22, -77, 61, 13, 84, -68, 53, -78, 78, 46, 46, -29, 25, -34, 55, 66, 5, -68, 19, 19, 31, -67, -68, -35, -17, 9, 30, -5, 27, -54, -22, 31, -33, 58, -52, -28, -9, -82, -30, -57, 74, 65, -34, 16, 11, 52, -77, -59, 33, 37, 24, -76, 82, 18, 27, 57, 50, -50, -39, -16, 1, -42, 63, 31, -28, -84, -67, -72, -82, 69, 78, -1, -14, 19, 41, -74, -70, 57, -14, -81, -41, 43, -12, 67, 76, 52, -70, 71, -21, 85, -22, 50, 3, 80, -54, 25, -64, 1, -44, 23, -63, 42, 35, -22, -74, 70, -26, 79, -36, 84, 57, -4, -20, 75, 28, -58, -77, -5, -62, -59, -36, 47, -29, -70, -59, -58, -38, 26, -77, 25, 2, -45, 57, -10, 85, -44, 21, -63, -82, -36, 39, -53, 23, -43, 73, 79, 17, 20, 84, 76, -45, -9, 78, -51, -8, -24, -5, 41, -23, -31, -39, 67, -40, 81, -37, 64, -25, 6, -45, 61, -57, -84, 83, 56, -61, 36, 49, 63, 78, -52, 11, 3, 18, 22, 78, 77, -61, -75, 67, -48, -42, 85, 30, 49, 7, -53, 21, 26, 61, -63, -75, -70, 80, 43, 5, 75, 10, -37, -13, -72, 85, 30, -51, 44, 51, 17, 38, 20, 45, -81, 1, 68, 72, -44, -83, -34, -77, -73, -5, 48, 13, -68, 18, -73, 42, 58, -46, 52, -18, 39, 73, -63, 80, 63, 36, -74, -77, -9, -42, 29, -19, -73, 55, 26, -56, 6, -27, -35, 52, -57, 7, 20, -56, 79, -68, 39, -44, -58, 35, -43, 65, 83, -29, 76, -43, 27, -9, -56, -39, 13, -25, -18, 1, -19, 73, -38, 2, -74, -28, 55, 74, 60, 66, 62, 78, -85, -65, -41, 58, 56, -58, -51, 74, -11, -24, 8, 79, -62, 65, -78, 85, -57, -47, 53, 36, -64, 27, -17, -32, -83, 3, 69, 80, 14, -2, 53, -8, 22, -75, -14, 67, -31, -50, -41, 8, 10, 28, 21, 76, 35, 74, 36, -49, 79, -79, -46, 62, -77, 70, -13, 1, 43, 71, 67, 75, -28, -46, -47, -6, 27, -81, 13, 80, 40, -45, -59, -75, 17, 84, -65, 41, 65, 12, 15, 22, 16, 37, -5, -20, -69, 34, 73, 41, -4, -62, 23, 39, -59, -60, -68, 49, 12, -38, -72, 57, 83, 42, 71, -45, -40, 52, 78, 78, 28, -9, 52, 13, 52, 36, -48, 60, 11, -19, 20, 42, 22, 54, -32, -51, 41, 15, 22, 41, -79, 79, 12, -23, 75, -50, 69, 49, 76, 21, -43, 25, -27, 60, 18, -44, -2, -50, -39, -54, -80, 50, 69, -24, 28, 72, 27, 11, -80, -78, -21, 71, 76, 48, -39, -33, -35, 20, 4, 37, -42, 46, -75, 30, -12, -9, 37, 52, -47, -22, 73, 27, 45, 6, 3, 55, -83, -3, 24, -41, -32, -75, -45, -4, 61, 44, 42, -35, -43, -56, -14, -66, -2, -74, 52, 31, 41, -83, 70, -61, -59}

#define GROUP_CONV2D_KERNEL_0_SHIFT (9)

#define GROUP_CONV2D_BIAS_0 {-69, -110, 80, -83, 8, -11, 105, 109, 87, -81, 66, 32, 59, 58, -100, -39}

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
	layer[1] = model.hook(GroupConv2D(16, 2, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
