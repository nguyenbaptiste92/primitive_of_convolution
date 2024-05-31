#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {72, -87, 69, 25, -52, -8, 30, 90, -58, 73, 85, 22, 24, 50, 74, 15, 74, 9, 90, 72, 97, -77, 82, -76, -37, 4, 85, -84, -38, -6, -47, -89, -36, -53, 69, 92, -61, 8, -81, 63, 3, -74, -48, -23, 91, 7, -11, 1, -4, -80, 54, 80, -85, -46, 10, 92, 90, -25, -38, 36, -34, -82, -51, -6, 38, -62, -37, 29, 97, -43, -86, -25, 19, 25, 34, 99, -47, -32, 5, -18, 56, -21, 42, -43, -33, -76, -2, -56, -78, 22, -55, 20, -8, -13, -13, 42, -42, -34, 11, -13, 43, 83, -69, 96, -29, -39, -91, 96, -63, 34, 60, -77, 25, -75, 43, -2, 2, 3, -92, -37, 45, -59, 66, -79, -81, 25, -35, -100, -90, 69, -19, -5, 24, 27, 85, 97, 68, 70, 25, 0, -1, -40, -82, -25}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {-3, 107, 109, -26, 61, -62, -67, -92, 13, -5, -99, 0, 39, 84, -73, -31}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-127, 17, 1, 47, 66, 107, -79, -36, -66, 17, 70, -61, 91, 122, 66, 3, -93, -78, -78, -31, -102, -48, 10, -110, 81, -112, -34, -104, -8, 37, -5, 63, 57, 72, 110, -91, -120, 91, -51, 99, 12, 113, 112, 21, 15, -117, -50, -123, 86, 122, -71, 65, 96, 29, 123, -110, -64, 125, -65, -98, 107, -44, -38, -18, 9, 22, 22, 6, -29, 128, -97, 9, -100, -34, 14, 76, -127, -119, 3, 92, -119, 46, 111, -62, 80, -21, -98, 110, -49, -10, -4, -33, -13, -105, 111, 57, -55, -9, 118, 25, 0, 2, -96, -19, 27, -13, -126, 69, -29, 40, -7, -14, 124, 12, -116, -15, 87, 43, -84, 45, -33, -87, 76, -1, 72, -107, 123, -4}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-30, -72, -63, 44, -73, 27, 61, 55}

#define CONV2D_BIAS_0_SHIFT (7)


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
static int8_t nnom_output_data[8192];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(8, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 8), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
