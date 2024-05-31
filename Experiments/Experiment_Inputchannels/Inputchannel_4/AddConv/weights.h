#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {77, 27, 43, 77, -56, 33, -34, 89, 66, 40, 24, -78, -45, 40, 1, 71, -35, 22, -26, 85, -68, 14, -2, 15, 77, 71, 73, -46, -12, -2, 80, -88, -66, 30, -67, 18, -64, 68, 93, -61, -67, 4, 16, -31, 18, -2, -45, 45, 25, -24, 42, -55, 6, 45, 11, -23, 26, 66, 1, -52, -51, -90, 34, 55, -70, -53, 89, -58, -22, 42, 56, 47, -92, -79, -73, 50, 20, -85, -72, -31, -31, 16, 66, 34, 5, 65, 83, 82, -6, 64, -83, -12, -24, 47, 53, 52, -72, 85, 41, 49, -47, 61, -51, -73, -16, -50, -9, -82, 22, -26, -22, -25, 19, -80, 73, 90, -82, -53, -84, 84, 73, -3, 80, 79, 42, 10, 52, 19, -59, -89, 83, 83, 0, 24, 22, -59, 19, -5, 89, -64, 24, -83, 5, -40, -69, -46, -6, -41, -60, 28, 56, -30, 46, -90, 91, 11, -59, -72, -92, 21, -48, 43, -89, 35, -63, 49, -36, -77, 10, -87, 20, -26, 71, 25, -89, -31, 54, 67, -38, -7, 50, 30, -36, -16, -59, 63, -79, 0, 44, 28, 3, -2, 38, 8, 26, 48, 18, -52, 42, -59, 15, 63, 53, 5, -81, -70, -45, -22, 21, -44, 48, -66, 35, -21, 10, 3, 12, 3, 81, 4, 71, 66, -18, 77, 29, -8, -21, 88, 7, 11, 79, -36, -9, -2, 5, 55, 53, 40, -25, 81, 50, -5, -74, -30, 6, 74, 70, -41, -26, 33, -84, -64, -54, 51, 79, 4, 29, 47, -61, 71, 35, -51, 81, -10, 80, -16, 65, 68, 11, 39, 78, -30, -1, -19, 31, -7, -11, -64, -52, 54, 25, 2, -42, 25, 87, -29, -76, 2, -39, -75, 12, 38, -13, 85, 81, -83, -29, -61, 56, 69, 86, -28, 68, -31, 65, 43, -19, -33, -68, -10, 64, 30, -30, -8, 8, -90, 26, 59, 30, 23, 48, -78, 39, -23, 43, -35, -76, 39, -52, -13, 80, 63, -81, -59, 33, 45, -75, -11, -21, -78, 15, -18, -40, -84, -9, -34, 57, 10, 22, 93, 19, -40, -65, 17, -32, 90, -8, 4, 20, 8, -22, 41, -30, -64, -18, -5, -15, 60, -39, 48, -18, -18, -59, -33, 72, -12, 33, -40, 93, -90, -41, 92, -12, -34, 73, -10, 0, 53, 35, -35, -30, -58, -3, 1, 93, -74, 29, -66, -57, -34, 80, -72, 42, 21, -72, -2, 61, 60, 85, -93, 18, 10, -87, -82, 72, 3, -90, 87, -29, 59, 2, 91, 3, -49, 87, 65, 14, 70, -90, 47, -12, -45, -36, -64, 4, 63, 88, 35, 92, -20, 30, 61, -4, -45, -71, -17, -26, 70, 75, -3, -56, 77, 46, 35, -75, -39, -87, 8, 18, -33, 61, 44, -84, -18, -59, -68, -29, -90, -54, -5, -68, 2, 18, 25, 48, -67, -57, -83, -58, 65, -55, -65, 87, 7, 31, -30, -36, 25, -4, 23, 3, 31, 61, -59, 13, 28, 55, 35, 14, -14, -15, -73, 54, -35, -71, 39, 48, -91, 18, -26, 57, 13, -4, 42, 22, -71, -82, 4, -51, -33, -62, 56, -68, 52, -67, -37, -5, 17, -59, -11, -36, 50, 5, 27, -76, 62, 75, 20, -39, 19, -26, -60, -66, 37, -26, 71, 27, 12, 30, -57, -49, -48, -56, -39, -15, 49, -67, -5, -12, -92, -31, 34, -65, 80, 73, 76, 27, 5, 8, -22, 31, 56, 66, -70, -91, 11}

#define ADD_CONV2D_KERNEL_0_SHIFT (9)

#define ADD_CONV2D_BIAS_0 {93, -51, 87, 20, -76, -96, 8, 37, -84, 106, 22, 49, -30, -81, 26, 73}

#define ADD_CONV2D_BIAS_0_SHIFT (8)

#define BATCH_NORMALIZATION_KERNEL_0 {125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {67, 70, 69, 68, 73, 70, 67, 67, 69, 69, 70, 69, 70, 71, 70, 70}

#define BATCH_NORMALIZATION_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ADD_CONV2D_OUTPUT_SHIFT 2
#define BATCH_NORMALIZATION_OUTPUT_SHIFT 2

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
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 4), nnom_input_data);
	layer[1] = model.hook(AddConv2D(16, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(16, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
