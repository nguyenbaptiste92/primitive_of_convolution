#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {9, -43, -28, 41, 12, -21, -5, 21, 28, -2, 14, -11, -40, -40, 36, 11, 21, 41, 65, -44, 28, 24, -2, -6, -48, -67, -26, 15, 17, 38, 13, 4, -48, -66, -10, -50, 22, 40, -29, -34, -66, 17, -5, 1, 54, 68, 27, 11, 46, 1, 57, -15, -9, 34, 69, 61, 19, -18, -55, 68, -8, -33, 40, -24, 21, -62, -64, -5, -54, 0, -17, 13}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (8)

#define DEPTHWISE_CONV2D_BIAS_0 {-69, 8, 27, -58, -3, -67, -43, 56}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (7)

#define CONV2D_KERNEL_0 {-73, 100, 12, 67, -41, 58, 78, 20, 2, -93, 91, 11, -104, -15, 70, 105, -14, 43, 70, -2, 22, 21, -7, -16, -20, -92, -75, -11, 99, 106, 59, 56, 31, 98, 110, 67, -47, 83, 50, -34, 52, -65, 16, -123, -2, -38, 67, -80, -84, 57, 53, -54, 9, 110, -57, 68, -24, 113, -76, 121, -82, 108, 108, -22, 60, -29, 102, -27, -127, 50, 13, -123, -87, -111, 97, 49, 78, -128, -43, -124, 34, -38, -20, -62, 19, -52, 66, -83, -120, 7, 44, 115, 89, 85, 76, -101, -77, 64, 63, 91, 87, 8, 52, -114, -6, -62, -100, -16, 8, 17, -97, 105, -118, 92, 79, 108, 89, -71, 110, 53, 48, -26, -127, 122, 21, 40, 125, 22}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-23, -36, 67, -93, -96, -74, -88, 92, -110, -6, 106, -20, 102, 36, -9, -22}

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
static int8_t nnom_input_data[8192];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 8), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
