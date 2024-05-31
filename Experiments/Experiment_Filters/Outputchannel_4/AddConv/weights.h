#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {-15, 34, -2, 58, -76, 38, 76, -78, 72, 8, 8, -63, -88, 33, -21, 75, 16, 1, -2, -35, 65, 44, 49, -92, -64, 37, -74, -23, 18, -71, 64, 8, 59, 5, -43, -27, -73, -83, 76, -14, 63, 9, 92, 75, 1, -31, 62, 41, 28, 68, -2, -48, -92, 48, -78, 66, -11, 23, 19, 82, -18, -77, -63, 41, -2, -1, 17, 81, -17, -6, 36, 3, -68, -65, -32, -53, -17, 58, 22, 86, -47, -48, 79, -84, -19, 17, 14, 15, -29, 88, 47, 91, -29, 64, -74, -90, -91, -57, -47, 56, -49, -69, -39, -76, -31, 47, -81, -84, -2, -81, 62, -67, 18, -50, 33, -11, 43, -92, -11, -2, -51, 69, -30, -39, -63, 66, -83, -84, 70, -75, 10, 2, -29, -62, -3, 87, 78, 20, 21, 63, 14, -88, 55, -18, -12, 1, -15, 64, -64, 46, 84, -40, -54, 13, -32, 55, -69, -37, 54, -6, 79, -29, -79, -79, 91, 82, 39, -16, -47, -30, 43, -64, 32, -65, -63, -86, 75, 31, -48, -35, 66, -30, -42, 1, -56, 69, 49, 4, 66, 57, -52, 37, -70, 65, 34, 10, -65, -68, 64, -77, 4, 66, 29, 87, 71, -48, -15, 5, -90, -32, -24, 76, -71, -91, -91, 1, -12, -43, -5, -24, 3, 75, 59, 70, 29, 86, -82, 80, 34, 64, -33, -88, -62, 31, -13, -17, 62, 63, 5, -18, -17, 30, 41, 80, -93, 18, 61, -31, -55, 59, 26, -24, -15, -48, -34, -84, 43, 64, -40, 33, 66, -31, -20, -25, -43, -29, -82, 57, 71, 36, 76, -59, 28, -76, 29, -36, 32, -6, 34, -14, -64, -9, -80, 29, 39, -89, 81, -34, 52, 19, 85, -28, 34, -2, 80, 65, -63, 85, -37, 54, -56, 75, 73, -40, -39, 43, -6, 34, -36, -4, -18, 31, -86, 90, -2, 27, -2, 18, -13, -55, 28, -16, -37, -10, -38, -67, -27, -50, 4, -36, 23, 86, -91, -16, 42, 59, -70, -84, -16, 5, -13, 44, -16, -56, 40, 23, -88, 12, 19, 75, 22, 15, -46, -42, 68, -84, -77, 61, 2, -74, -1, 12, 92, -15, -22, -55, -80, -84, 1, -89, -55, 38, -77, -57, -47, -11, -61, -24, 77, -73, 50, -47, 7, 14, -39, -40, -1, -66, 16, -15, -12, -21, 42, 13, 47, -6, 8, -25, -9, -54, 49, -52, 31, 23, 93, -84, 20, 72, 20, -21, -76, -26, -79, -87, 84, 70, 71, 42, -52, 18, -52, -40, 55, -61, 86, -61, -45, -38, 25, -41, 83, -40, 60, -13, 9, -29, 76, 8, 67, -54, 19, -78, -9, 63, 51, -78, 28, -23, 45, -61, 24, -27, -34, -40, -63, -9, -58, 41, -52, 87, 84, -46, 0, -73, 90, -9, -32, 35, 61, -64, 47, 61, -90, 14, -87, -9, -17, 69, 59, 0, -60, -46, -92, -14, -72, 4, 28, 51, -82, 86, -15, 44, -23, -26, 39, 14, -42, -12, -66, -71, 44, -2, 45, 42, 17, -62, -53, -67, -50, 33, -91, 37, 10, -40, 72, -13, 11, 6, -66, -12, 73, -10, 70, 45, 35, 48, 63, -1, 29, 81, -39, 75, 83, 20, 52, -43, -63, 89, 69, 75, -4, -13, -81, -26, 24, -68, 73, 78, 93, 86, -41, -80, -32, -31, 35, 30, -33, 47, -93, 25, -11, 40, -60, 43, 49, 63, 13, 14, 22, -82, 50, -49, -90, 85, -24, -24}

#define ADD_CONV2D_KERNEL_0_SHIFT (9)

#define ADD_CONV2D_BIAS_0 {33, -119, 4, 30}

#define ADD_CONV2D_BIAS_0_SHIFT (8)

#define BATCH_NORMALIZATION_KERNEL_0 {103, 103, 102, 104}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {86, 86, 86, 85}

#define BATCH_NORMALIZATION_BIAS_0_SHIFT (7)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ADD_CONV2D_OUTPUT_SHIFT 0
#define BATCH_NORMALIZATION_OUTPUT_SHIFT 0

/* bias shift and output shift for each layer */
#define ADD_CONV2D_INPUT_LSHIFT (ADD_CONV2D_KERNEL_0_SHIFT-INPUT_1_OUTPUT_SHIFT)
#if ADD_CONV2D_INPUT_LSHIFT > 0
#define ADD_CONV2D_MODE 1
#define ADD_CONV2D_INTER_SHIFT ADD_CONV2D_KERNEL_0_SHIFT
#define ADD_CONV2D_INTER_LSHIFT (ADD_CONV2D_KERNEL_0_SHIFT-INPUT_1_OUTPUT_SHIFT)
#elif ADD_CONV2D_INPUT_LSHIFT < 0
#define ADD_CONV2D_MODE 2
#define ADD_CONV2D_INTER_SHIFT INPUT_1_OUTPUT_SHIFT
#define ADD_CONV2D_INTER_LSHIFT (INPUT_1_OUTPUT_SHIFT-ADD_CONV2D_KERNEL_0_SHIFT)
#else
#define ADD_CONV2D_MODE 0
#define ADD_CONV2D_INTER_SHIFT ADD_CONV2D_KERNEL_0_SHIFT
#define ADD_CONV2D_INTER_LSHIFT 0
#endif
#define ADD_CONV2D_OUTPUT_RSHIFT   (ADD_CONV2D_INTER_SHIFT-ADD_CONV2D_OUTPUT_SHIFT)
#define ADD_CONV2D_BIAS_LSHIFT   (ADD_CONV2D_INTER_SHIFT-ADD_CONV2D_BIAS_0_SHIFT)
#if ADD_CONV2D_OUTPUT_RSHIFT < 0
#error ADD_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if ADD_CONV2D_BIAS_LSHIFT < 0
#error ADD_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif
#define BATCH_NORMALIZATION_OUTPUT_RSHIFT (ADD_CONV2D_OUTPUT_SHIFT+BATCH_NORMALIZATION_KERNEL_0_SHIFT-BATCH_NORMALIZATION_OUTPUT_SHIFT)
#define BATCH_NORMALIZATION_BIAS_LSHIFT   (ADD_CONV2D_OUTPUT_SHIFT+BATCH_NORMALIZATION_KERNEL_0_SHIFT-BATCH_NORMALIZATION_BIAS_0_SHIFT)
#if BATCH_NORMALIZATION_OUTPUT_RSHIFT < 0
#error BATCH_NORMALIZATION_OUTPUT_RSHIFT must be bigger than 0
#endif
#if BATCH_NORMALIZATION_BIAS_LSHIFT < 0
#error BATCH_NORMALIZATION_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t add_conv2d_weights[] = ADD_CONV2D_KERNEL_0;
static const nnom_weight_t add_conv2d_w = { (const void*)add_conv2d_weights, ADD_CONV2D_OUTPUT_RSHIFT};
static const int8_t add_conv2d_bias[] = ADD_CONV2D_BIAS_0;
static const nnom_bias_t add_conv2d_b = { (const void*)add_conv2d_bias, ADD_CONV2D_BIAS_LSHIFT};
static const nnom_addconv_parameter_t add_conv2d_parameter = { (const void*)NULL, ADD_CONV2D_MODE, ADD_CONV2D_INTER_LSHIFT};
static const int8_t batch_normalization_weights[] = BATCH_NORMALIZATION_KERNEL_0;
static const nnom_weight_t batch_normalization_w = { (const void*)batch_normalization_weights, BATCH_NORMALIZATION_OUTPUT_RSHIFT};
static const int8_t batch_normalization_bias[] = BATCH_NORMALIZATION_BIAS_0;
static const nnom_bias_t batch_normalization_b = { (const void*)batch_normalization_bias, BATCH_NORMALIZATION_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(AddConv2D(4, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(4, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 4), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
