#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {-5, -21, 42, 91, -42, -80, 81, 33, 72, 1, 13, -41, -49, 56, 4, -63, -49, 74, 64, 0, -62, -25, 100, -59, 76, -99, -16, -19, 39, 68, -18, 76, 48, 95, 41, 17, -69, 75, 82, -92, 5, 14, 22, -61, 58, 17, -11, 67, -43, 55, 21, -95, -29, -50, 34, -50, 51, -96, 21, -96, 72, 85, 65, 43, -73, -97, 60, -20, -25, 3, -19, 46, 10, 32, 7, -88, 52, -3, -43, 61, -90, 100, 34, 67, 40, 100, -82, -43, -63, -6, 42, -73, 5, -39, 30, -25, 45, 32, -48, -21, 100, 33, 35, 21, -14, -52, -77, 38, -96, -91, 89, 98, 86, 19, 94, -93, -99, -68, -79, 45, 31, 56, -53, -13, 9, -84, -42, 62, 70, -4, -60, -5, 71, 56, -60, -87, 68, -90, 73, 26, 34, 16, -88, 15}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {72, -99, 105, 110, 12, 80, 14, 7, -56, -35, 104, -29, -45, 99, -9, 45}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {94, -17, 24, 84, 54, 74, -83, -42, -27, 29, 23, 47, -86, -95, -20, -75, 48, 18, -30, -22, 84, -71, 24, -96, 74, -50, -23, -94, 0, 6, -26, 72, -94, -76, 41, -106, -18, 101, 29, 82, 32, -66, 98, -3, -104, -14, 51, 64, -13, -79, -93, -15, -100, -111, 41, -4, 63, 76, -75, -38, 8, 90, -16, 104, -43, -56, 107, 10, 59, 91, -76, 1, -58, -39, 26, 29, -108, 96, 85, 40, 51, -104, 14, 55, -17, -60, -91, -29, -107, -99, -97, 39, 85, -92, -57, 6, -55, 56, -90, 94, -20, -62, -107, 36, 37, -71, 5, 77, 9, -68, -108, -5, 13, 85, 4, -51, -67, 13, -33, 105, -18, 15, -38, -57, 1, 10, 89, 104, 108, -18, 4, 88, 92, -10, -51, 3, 69, -1, 64, -19, -65, 17, -5, 39, 22, -17, 56, -65, -37, -52, 42, -99, 33, 89, 39, -96, -83, 10, 36, -82, 101, 104, 97, 89, 62, -8, -79, -7, 31, -1, -23, -11, 52, -42, 7, 30, 7, -78, -80, 64, -84, -15, -82, -97, -99, 81, -38, -10, 108, 39, 57, 77, -50, 10, -24, 47, -23, -99, 78, 7, 29, 94, -30, 40, 32, -21, -46, -66, 76, 26, 38, 97, 92, -9, -86, -5, -104, -16, -34, -56, -101, 51, -79, -85, -59, 24, 72, -19, 99, 79, 48, 43, 90, 38, -2, 108, 52, 6, 16, 28, 30, -71, 15, -31, 95, 86, -55, 13, 24, 2, 64, -82, -6, -89, -34, 78}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {99, 82, -29, -72, -51, -92, -30, 50, -47, -8, 24, -5, -8, 33, -101, -3}

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
static int8_t nnom_input_data[1024];
static int8_t nnom_output_data[1024];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(8, 8, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(8, 8, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
