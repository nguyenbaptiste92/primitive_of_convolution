#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {-92, -79, -97, 79, -17, 21, 45, 99, 81, -99, 20, -93, 51, -109, -110, 57, 70, -103, -8, -96, -57, 90, -118, -26, -40, -97, -79, -105, 21, 16, -4, 116, -9, -40, -67, 104, -55, 74, 88, -59, -86, -99, -55, 45, 89, -22, 93, 8, 60, 52, 45, -36, 64, 68, 75, -91, -45, -15, -4, -13, 37, -7, 111, 4, 80, 55, -112, -3, 83, 72, 91, 11, -115, 63, -23, 38, -121, 21, 27, 24, -15, -81, -91, 23, -1, 118, -75, 91, 7, -63, -49, -106, -65, 98, -84, -34, 77, -101, 75, 26, -12, 45, -56, 14, -102, -85, -39, -80, 65, -17, 82, -120, -121, 46, 104, 63, 53, 14, 116, 52, -121, 37, -49, -4, -14, 34, -15, -28, -69, 26, 17, -115, -57, 77, -93, -93, 36, 28, 55, 64, -41, 22, -107, 9, -27, 46, 17, 40, 44, -72, -55, 15, -26, 121, 78, 88, 78, 19, -16, -10, 33, 7, 48, 95, -63, -14, 4, 26, 46, -95, 105, -63, 84, -1, 119, 73, 33, -23, 114, -4, 84, -90, 112, 42, -1, 21, 39, 18, 37, -24, -63, 84, -85, 17, 19, 60, -12, -70, 50, -28, 113, 70, -52, 99, 104, -69, 66, 58, 110, 75, 53, 94, 83, -63, 55, -96, -44, 22, -32, 86, -38, 4, -77, -23, -49, 85, -70, -56, 27, -108, -69, -14, 44, 90, 57, -20, 24, 94, 68, 63, -83, 85, -1, 86, -103, 69, -87, 82, -115, -6, -107, -88, -34, 115, -72, -16, 40, 75, 113, -66, -111, 79, 120, 115, -8, -94, 96, 29, 89, 0, 112, 36, 11, 11, -15, -92, -38, 101, 35, -37, 96, 109, 100, -114, 3, 103, -110, 22, 41, -98, -16, 60, 90, -43, -56, 112, 78, 26, 15, -9, 34, 87, 71, 93, -56, -105, -53, 64, -43, 109, -80, -22, 113, 116, 104, -5, -36, 56, -58, -77, -43, 6, -92, 121, -87, -120, -14, -94, -28, -109, -92, -118, 34, 41, -5, -9, 13, -51, -94, -30, -57, 75, -75, -116, -41, 111, 69, -92, -115, -8, -53, 65, -30, -70, -117, -95, 15, 101, 25, 12, -100, 30, 25, -71, -8, 56, 60, 62, -113, -76, -10, -32, 81, 18, 102, -48, -30, 5, 110, -23, 37, -42, 57, -84, 41, -64, 45, -115, 36, -73, -36, -3, 37, -82, -89, -109, -100, 21, -64, 7}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (10)

#define DEPTHWISE_CONV2D_BIAS_0 {7, 68, 90, 105, -83, -67, -12, 66, 38, -56, 12, 81, 26, -72, 43, 70}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {5, 14, -42, 110, 62, 3, -34, 28, -73, 4, -100, -12, 38, -63, 61, -92, 97, 55, 32, -60, 101, -6, 106, 24, -57, 66, -84, 86, 88, -63, -38, -35, 100, 79, -83, -79, -19, -58, -33, 18, 56, 2, 23, -30, 12, -45, -20, 91, 33, 85, -4, 16, 74, -70, -109, -56, 104, 37, 29, -75, -87, -88, 16, 25, 70, 102, -34, -70, -17, 72, -84, -47, 1, 97, 58, 93, -75, 106, -56, 36, -90, -109, -28, -12, 77, -52, 10, -97, 75, -19, 0, -54, 16, -97, 46, -46, -53, 106, -16, -58, -65, -69, -96, -87, -52, -25, 70, 110, -18, 110, -98, -78, -43, -74, -33, 65, 85, 91, 56, -36, -75, -104, -2, -109, 5, -75, 72, 83, -9, -84, 82, -20, -31, 89, -106, 28, 78, 93, -32, -79, 83, 22, 10, -86, -71, -17, -73, -76, 102, 53, 2, -76, -4, 104, -2, 74, 21, 71, 110, -80, 87, 49, 13, 37, 87, -101, -70, -40, 19, 73, 92, 80, -95, -74, 61, -34, -5, -50, 2, 100, -20, 19, 52, 10, 70, -78, 82, -50, -105, -9, 0, 84, -73, -52, -69, -5, 16, -42, 15, -58, -66, 38, -63, 12, 46, 107, -19, 86, 81, -75, -82, -48, 52, -104, -17, 39, -79, -2, -40, 91, -87, -47, -27, -68, 26, 45, 78, 37, -103, -40, 15, -69, -13, 102, 36, 78, -61, 7, -51, -61, -44, -36, 61, -110, -87, -101, -96, 100, -87, 60, 81, -86, -75, 107, 80, 84}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-38, 88, 46, -106, 80, 38, 78, -79, -71, -31, -100, -71, 65, -85, 30, -30}

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
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(5, 5), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
