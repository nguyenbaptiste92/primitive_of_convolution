#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {6, 12, -34, -52, 48, 51, 14, -18, 36, -32, -47, -9, -27, -58, -53, -65, 70, 34, 56, -22, -3, -38, 21, 22, -78, -39, 13, -56, 55, -71, -17, 3, -56, -84, 74, 46, 5, -69, 38, 1, 34, -73, 8, -73, 63, -55, -41, 40, 4, -50, 39, 71, 6, -67, 20, -40, 18, -53, -78, -50, 31, -8, -34, -1, -81, 54, 2, 76, -28, 81, -73, 23, 23, 21, 56, -21, 73, -64, 68, 52, 10, -21, 46, 76, -80, 12, -81, -21, 46, -45, -63, 29, -27, -27, 85, 16, 38, 71, -3, 82, 78, -50, -80, -85, 76, 10, 14, 16, -63, 61, -47, 63, -31, 3, 85, 23, 81, 11, -69, -79, -33, 82, -3, 23, 69, 75, -62, -75, 58, 44, -33, 73, -21, -50, 46, -58, -26, -72, 37, -55, 15, 11, -36, 40, -79, -16, -79, 51, 73, -67, 71, 58, -78, -62, -46, -69, -25, -1, 43, -52, 66, -55, 58, -83, -50, 72, -50, 69, -72, 13, -59, 31, -21, 30, 31, 63, 1, -6, -61, 35, -82, 63, 5, 15, 65, 65, 83, -42, 43, -64, 31, -81, -60, 38, -44, 44, 79, -4, 17, 2, -52, 57, 5, 28, 73, -21, -35, 77, -59, 55, -25, -26, 10, -36, -5, 2, -28, 42, 79, 61, -34, -22, -78, -53, 40, 46, -28, 56, -32, -54, 22, 5, 61, 10, -35, 26, 37, 36, -85, 9, -64, 48, -74, 9, -8, -53, 17, -59, -80, 21, 53, 38, -72, -53, 79, 64, -53, -31, -9, 27, -73, 17, 52, 80, 72, 64, 41, -81, -57, 64, -77, -16, -28, 66, 40, -80, -53, 67, -59, -78, 39, 30, -59, -70, 7, -25, -29, -37, -85, 35, 77, 33, -44, -36, -60, 64, -6, 39, 12, 50, -74, -73, -49, 68, 15, 37, -11, 37, -79, -13, -78, 22, 75, -49, -35, 68, 9, 46, 35, -15, -77, 45, -62, -9, -61, 55, 71, -4, -23, -4, 18, 19, 24, 47, -46, -33, 68, -45, -17, 5, 56, 14, -59, 43, 22, 50, -82, -72, -42, 40, 29, -14, -65, -23, 83, 23, 82, -49, 32, 62, 52, 60, -7, 74, 34, -82, 81, 2, -67, 68, 40, -85, 82, 17, 66, -54, 24, -30, -70, -43, -22, -38, 62, -6, -32, 20, -33, -81, 3, -6, 66, -4, 38, 20, -15, -47, 13, 21, 53, -1, 82, 58, 67, -14, 58, -43, 37, 30, -55, 40, 27, 29, 0, 7, 71, 76, -64, -75, 11, -69, -38, -85, -68, -55, 69, -56, -57, -5, -63, -59, -83, 82, -67, -62, -74, 42, -33, -5, 3, 3, -14, 3, 77, -3, 0, 48, -13, -53, 0, 47, 23, -69, -55, 67, -10, -61, 1, 54, 54, 46, -1, 71, 5, 13, -28, 81, 70, -68, -4, 19, -6, -64, -3, -48, 69, -47, -32, 67, 45, 30, -55, 9, -8, 48, 18, -26, -14, 72, -16, 29, -48, -37, -85, 69, -15, 8, 40, 16, 5, -11, -45, 39, -50, -61, 6, -76, 42, -51, 61, -67, -13, 59, 42, -74, 8, -24, 44, 79, -82, 12, 24, -64, 56, -54, -32, -20, 62, -71, 59, 64, -77, -61, -77, 50, 3, -81, 6, 19, 79, 62, -66, 14, -43, 75, -56, 9, -43, 46, -60, -33, -11, -26, -64, -31, -66, -45, 30, -20, 55, -13, 57, -46, -31, 75, -4, -26, -30, -24, 33, -8, -82, 83, 14, 12, 20, 84, -68, -50, -10, 2, -38, 29, -79, -30, -60, -17, -66, 51, 69, 15, 29, 65, 6, -12, -13, 68, -78, -53, -43, 7, -59, 79, 70, -8, 5, 65, 3, 85, -11, -65, -32, -63, 8, -51, 78, 13, 74, 54, 48, -64, -67, -11, 21, -44, 73, 19, 63, -37, 20, 80, -23, -49, -17, -25, 46, -65, 59, 18, 85, -84, -76, 59, 67, -71, -81, 78, -55, -22, 81, 28, 64, 73, 58, 51, 57, -9, 57, 9, 82, -68, 80, -73, 49, -38, 59, -65, 37, 62, 47, 51, -55, 78, 17, -43, -44, 71, 1, 34, 4, 66, -13, -21, -80, -54, 52, 56, 9, -64, 37, -76, 83, -32, 67, 44, 22, 35, 67, 16, 38, -29, -75, 28, -22, 3, 6, 46, -5, -23, 64, 51, 78, -21, -60, -49, 34, -20, -8, -80, -17, 38, -64, 13, 24, -80, -49, 31, -33, 71, 55, 18, -64, -20, -21, 11, -58, 50, 36, -69, 40, 8, 40, -57, -20, -9, 11, -12, 45, -43, 8, 61, 85, -53, -61, 17, 29, -29, 11, 22, 81, -71, -29, -36, -19, -26, 70, 45, -58, 23, -82, -6, -11, 13, 48, -4, 22, -74, 59, -39, 84, -82, 21, 55, 33, -66, -76, -58, 84, -81, -80, 5, 55, 10, 21, -72, -61, 36, -55, -47, 49, 18, 65, -34, 21, 15, 57, -21, 51, 60, -56, 66, 28, 13, 59, -53, -9, -27, -44, -53, -9, -56, 17, -34, 27, 38, 1, -33, -3, 63, 30, -82, 4, 56, -78, -15, -78, 69, 31, -65, -67, 46, 8, 18, 46, -27, -70, 13, 3, 21, -44, 17, -14, 30, -85, -69, -71, 36, -1, -30, 85, -19, -63, -7, -51, -32, 85, 45, 25, -21, 21, -13, 47, 53, 38, 78, -5, 10, 74, -81, 74, 56, -22, -26, -81, -59, 49, 33, -44, 55, 19, 55, 69, -19, 15, 23, 51, 82, 63, 74, 18, -78, -41, -75, -63, 39, -31, 31, 47, 83, -74, -54, 70, -59, 28, -11, -78, 40, 76, -22, -16, -69, 46, -84, 36, -77, -76, 37, -32, -38, -38, -33, 29, 56, 49, -73, 71, 69, 12, -7, -51, -40, -16, -32, 50, -65, -26, -8, 15, -11, -8, 76, -83, 57, 32, -45, 45, 30, 12, 58, 34, -72, 64, -64, -4, -54, -40, -50, -53, -74, -77, 40, -61, 82, -13, 85, 39, 4, 39, -45, -46, -36, -56, 36, 60, -51, 48, 76, -67, -3, 52, 73, -36, 58, 29, -78, -14, -59, -53, -12, -50, -34, 11, 72, 9, -18, 33, 21, -74, -13, 15, -24, -78, 68, -80, 60, 53, -68, -68, -81, 80, 66, 41, -33, -75, 28, 80, 12, -32, -8, 18, 3, 26, 56, 64, -34, -82, -31, -19, 6, -12, -57, -20, 17, -34, -27, -27, 62, 46, 19, -6, -21, -58, 7, 17, 75, 57, -85, -73, -46, -47, -21, -44, 36, -18, -63, -72, 26, -39, -30, -36, 44, -41, -65, 84, -13, -5, -60, 13, 3, -42, 11, 7, -55, 11, 72, 84, -81, -20, -63, -59, -27, -1, -72, -19, 27, 6, 38, -57, 32, -64, -23, -57, -3, -10, 20, -45, -47, 0, -69, -65, 34, -11, -57, 45, -67, 71, -2, 26, -34, 22, 72, -12, -26, 12, -56, -33, 31, -43, -46, 9, -12, 4, -36, -76, 48, 41, -73, -44, -16, -10, 24, -52, -52, -69, -28, 57, -50, 21, 85, 9, -30, -1, 52}

#define ADD_CONV2D_KERNEL_0_SHIFT (9)

#define ADD_CONV2D_BIAS_0 {72, -22, 56, 73, 51, -74, 69, -74}

#define ADD_CONV2D_BIAS_0_SHIFT (7)

#define BATCH_NORMALIZATION_KERNEL_0 {102, 103, 102, 102, 103, 101, 103, 101}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {84, 86, 84, 84, 83, 86, 84, 87}

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
static int8_t nnom_output_data[8192];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(AddConv2D(8, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(8, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 8), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
