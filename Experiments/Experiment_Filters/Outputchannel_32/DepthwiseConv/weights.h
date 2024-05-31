#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {16, 29, 56, 70, -51, 91, -91, -47, 77, 87, -71, -98, 25, -31, -59, -9, 48, -71, 7, -26, -75, 15, 1, 70, 40, -5, 60, -72, 94, 79, 1, 83, 53, 19, 60, 89, -68, -67, -39, -66, 45, 33, -69, -50, 63, -91, 82, 84, 81, -78, -75, -31, -84, 63, 73, -67, -19, -67, 98, 87, -64, -66, 58, -89, -3, -4, -87, -96, 86, -101, 90, -99, -35, 37, 25, 5, 41, 70, 47, -25, 32, -79, -38, -88, -47, -96, 80, -44, -83, 35, -47, 93, 19, 81, 34, 32, -98, -71, 16, -63, -74, 76, 35, 21, 27, -95, -100, 72, -96, -12, 95, 53, 96, -96, 60, 35, -13, 79, -26, -31, 40, 96, 93, -12, -32, 91, 26, 32, 78, -88, -44, 1, 55, 84, 53, 29, 13, 67, -91, -10, 23, 62, 90, 81}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {72, 31, -25, -36, 26, 28, 74, -67, -82, -98, -60, -25, 4, 12, -17, -4}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-53, -84, -53, 89, 67, -36, 63, 58, 71, 61, -59, -71, 66, 13, 7, 79, -31, -51, 33, 27, -88, -29, 89, 46, 47, -87, 30, 82, -73, -38, -65, 14, 23, -75, 67, -44, 45, -39, -79, -81, -19, 55, -43, -14, 21, 21, 68, 39, 44, -58, -36, -81, -81, 46, -66, 34, 9, -41, -37, 64, 41, -1, -24, 90, -73, 88, 15, -26, 77, 22, 48, 24, 22, 46, -24, 33, 33, -62, 9, 85, -35, 8, 52, -22, -67, 32, 77, -33, 88, 51, -7, -22, -55, -28, -57, 83, -9, -33, 8, 27, -39, 38, 86, 13, -66, -4, -31, 83, 76, 76, 38, 71, 19, 81, -70, -48, 16, -59, -79, 39, -27, 63, 17, 10, -86, -43, 27, -73, 39, -89, 79, 53, 63, -55, -30, -12, 69, -69, 12, -74, 46, -8, -26, 34, 69, -29, -21, -9, -74, -20, -28, 7, -89, -4, -18, -20, -45, -29, 29, 53, 89, -19, 74, 48, 88, -14, 84, -65, -29, 49, 2, -32, 60, 9, 76, 19, -51, 75, -86, -57, 78, 8, -41, 33, 78, -48, 17, -27, -39, 0, 73, -85, 48, -26, -35, 32, 61, -32, -78, 81, -35, -18, -61, 50, -76, -36, 50, 61, -1, 55, -28, -75, 55, -29, -71, -58, -42, 17, -24, -58, 90, -67, -13, -77, -45, 71, 36, 49, 38, -60, 78, -64, 0, 88, -7, -62, -23, 89, -63, -62, 7, 79, -64, 10, -83, 30, -38, 14, 85, 17, -29, 51, -5, 3, 31, -57, -76, -8, -2, -20, 32, -28, -5, 59, -81, 39, 43, 38, 62, -19, -86, 76, 86, 72, 57, 12, -25, -89, 30, 44, 61, 35, -77, 77, 49, 53, -40, 16, -68, -78, 36, 36, 73, 77, -75, 49, 59, -47, 71, 60, -6, 61, 7, -74, -60, 74, 53, 1, -46, -53, -88, -26, -37, -79, 11, 62, -85, -22, 86, -34, 62, -32, -77, -73, 45, -26, -28, 13, 48, 17, -86, 3, -8, -23, -74, -39, 68, -87, -76, 37, 82, -51, -35, 34, 11, -25, -80, 3, 16, -7, -22, -69, 89, 22, -83, -42, -39, 13, 22, -78, -32, -73, -70, 22, -49, 34, 1, 52, 57, 68, 79, -76, -65, 87, -40, -68, 41, 24, -54, -55, -5, -66, 22, -57, -29, -7, 17, 0, 38, 30, 60, 29, -5, 45, -44, 34, 37, -61, -16, -81, 62, -60, 16, 52, 58, -78, 12, -52, 63, -82, -78, 82, 0, -77, -77, 12, -54, -16, 83, 60, -10, 32, 44, -66, -9, -64, -64, -13, -33, 65, -23, -26, 88, 60, 54, -68, -66, 12, -45, 56, 49, 90, -66, -58, 31, 39, 41, 32, -78, -37, -20, 71, 38, -19, 64, 80, 33, -31, 71, -66, 11, 38, 15, 42, -61, -72, 65, 22, 62, -86, -77, -18, -53, -68, 36, 86, -90, -6, 7, -84, -64, 12, -46, 72, -11, -17, -39, -3, -40, 81, -4, 31, -21, -67, 24, 3, 7, 16, 68, -37, -22, 18, -59, 16, 1, 69, -35, 3, -10, 15, -90, 47}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-70, -33, -43, -72, 18, 53, -35, -48, -23, -56, -71, -7, -69, 56, -46, 12, 22, -18, 71, -55, 30, 19, -21, -71, -9, -16, 59, 24, -3, 26, -26, 45}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 6

/* bias shift and output shift for each layer */
#define DEPTHWISE_CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT-DEPTHWISE_CONV2D_OUTPUT_SHIFT)
#define DEPTHWISE_CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT-DEPTHWISE_CONV2D_BIAS_0_SHIFT)
#if DEPTHWISE_CONV2D_OUTPUT_RSHIFT < 0
#error DEPTHWISE_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DEPTHWISE_CONV2D_BIAS_LSHIFT < 0
#error DEPTHWISE_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif
#define CONV2D_OUTPUT_RSHIFT (DEPTHWISE_CONV2D_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_OUTPUT_SHIFT)
#define CONV2D_BIAS_LSHIFT   (DEPTHWISE_CONV2D_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_BIAS_0_SHIFT)
#if CONV2D_OUTPUT_RSHIFT < 0
#error CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if CONV2D_BIAS_LSHIFT < 0
#error CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t depthwise_conv2d_weights[] = DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0;
static const nnom_weight_t depthwise_conv2d_w = { (const void*)depthwise_conv2d_weights, DEPTHWISE_CONV2D_OUTPUT_RSHIFT};
static const int8_t depthwise_conv2d_bias[] = DEPTHWISE_CONV2D_BIAS_0;
static const nnom_bias_t depthwise_conv2d_b = { (const void*)depthwise_conv2d_bias, DEPTHWISE_CONV2D_BIAS_LSHIFT};
static const int8_t conv2d_weights[] = CONV2D_KERNEL_0;
static const nnom_weight_t conv2d_w = { (const void*)conv2d_weights, CONV2D_OUTPUT_RSHIFT};
static const int8_t conv2d_bias[] = CONV2D_BIAS_0;
static const nnom_bias_t conv2d_b = { (const void*)conv2d_bias, CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[32768];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(32, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 32), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
