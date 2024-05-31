#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {34, -1, -74, -54, -85, -9, 77, -16, 100, -5, 6, -99, -79, -99, 84, 87, 79, 62, -19, 18, 83, -28, -23, -49, 92, 90, 66, -19, -5, -14, 37, 66, 13, 0, 7, 81, -96, 47, 26, -51, 84, -62, 57, -79, -50, -15, 62, 55, -14, 22, 82, -69, -93, -64, 64, -92, -98, -14, 58, 14, 43, -43, 72, -38, 15, 45, -70, 77, 81, 5, 14, -65, -92, 6, 35, 34, 24, 35, -22, -58, -77, 92, 92, 47, -39, -60, 17, -45, -57, -97, -42, 90, -74, 46, -70, -66, 26, -38, 65, 32, -81, -63, -79, 84, -84, -1, -71, -27, 64, 17, 55, -13, 97, -39, 43, -2, -40, 15, 49, -65, 61, -95, -92, -29, -19, -90, -51, -11, -54, -18, 79, -97, 93, 86, -48, -19, -61, 11, 28, -56, 14, 82, -58, 69}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {25, 74, 0, -69, 25, -21, 10, -75, -34, 74, -107, -8, 17, -33, -31, 5}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-18, 97, -22, 98, -103, 21, -34, 46, 25, -48, 8, -21, -55, 65, -83, -79, -44, 18, 4, -23, -52, -13, 60, -19, 42, 14, -54, 76, -55, -12, -25, 21, 33, -46, -5, -14, 44, -60, 38, -34, -7, 25, -30, -52, -69, -47, 21, 16, 110, -10, -6, 49, -38, 8, 19, 28, 89, 74, -53, -107, 32, 34, -51, -35, 51, 35, 47, -50, 47, 58, -24, -2, 42, 39, -37, -76, 97, -84, -94, 26, 45, 67, -66, 43, 48, 22, -3, 44, -24, -8, -82, 19, 48, -67, -37, -21, 62, 8, 37, -6, 26, -46, -21, -88, 61, -9, -48, 83, 106, -25, -107, 24, 24, 57, 96, 97, 76, 3, 99, -102, -6, 7, 25, -54, -6, -104, -93, 63, 40, -105, 18, 29, 55, 78, -97, 32, 8, -96, -103, 42, 25, -41, 49, -80, 51, 38, -76, -77, -93, -94, 92, -52, 30, 96, -22, -16, -60, -18, 12, -55, 108, 40, -87, -64, 74, -62, -79, 89, 89, 72, -79, 17, 70, -66, 62, 39, 28, -71, -12, 16, -48, -22, 40, -78, -60, 48, 49, 87, 46, -82, -13, 60, 25, 76, -6, -56, -2, 75, -26, 46, -37, 64, -16, 15, 94, 20, 26, 26, -58, -51, -85, -104, -25, -37, -105, -105, 48, 100, -3, 30, 92, -11, 35, 60, -47, 104, 35, 83, -95, 3, -95, 27, -65, 56, -44, 8, 93, 62, -61, 27, -41, 32, -27, 88, 95, -3, -39, 9, -26, -2, -101, -52, -17, -101, -105, 89}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-96, 62, -60, 15, -39, -9, -87, 8, 64, 104, 2, -33, -74, 57, -41, -50}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 6
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
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
