#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {55, -76, -4, 24, 20, -18, -10, 43, 53, 77, 34, 71, 68, -30, 10, -83, 74, 18, -3, -48, 84, 73, -14, -67, 54, -80, -37, -80, -32, -81, -23, 77, -84, 33, -75, 41, 63, -34, 28, -60, 59, -82, 82, -27, -6, -19, -75, 46, -70, 25, -53, -29, -22, 8, 60, 69, -44, -74, -7, 61, -13, -82, -2, 37, 53, -35, -33, -9, 8, -59, -64, 80, -21, -38, -68, 50, -5, 38, 20, -74, -69, 83, -56, -43, 18, -58, 69, -49, 43, 34, 75, 51, 83, -26, -28, -23, -6, 70, 63, -34, -48, 0, 76, 24, -79, -24, 32, 66, 13, 51, -48, 73, 10, 28, -5, -2, -57, 56, -36, 74, 66, -34, 69, 47, -24, -5, 40, 66, 30, -11, 71, -82, 33, 33, -11, -57, 48, 30, -60, 27, 82, 74, -34, -5, -23, -55, 56, -72, 27, 40, -41, -84, 77, -56, -6, -18, 85, 51, -7, 47, -45, -45, 53, 85, 55, -78, -29, 68, -68, 54, -37, 43, -34, -3, 81, 7, 9, 31, 12, -41, -60, -72, 31, -74, -55, -15, 19, 68, -57, -72, 39, 6, -19, -35, -84, -16, -41, -67, 27, 6, 2, 29, 46, 65, 31, 38, -36, 81, 84, 30, 15, -44, 51, -12, -34, 75, 74, -54, 42, -66, -39, -2, -43, -80, -70, 28, 52, 9, -82, 2, -44, -42, 11, -54, -42, -52, -40, -39, -23, 69, -16, 84, 52, 49, -2, -69, -6, -71, -60, 62, -66, 10, 30, -61, 83, -42, -2, -18, 73, -3, 18, 34, -73, 28, -26, 83, -48, -78, 76, 11, -4, 7, -78, 27, -61, -61, 56, -27, -12, 72, 49, 79, 78, 1, 76, -56, 23, 19, 56, -73, 20, 46, -52, 83, 2, 6, 60, -83, 2, 2, 79, 23, -52, -37, 9, -41, 48, 69, 58, 16, -45, 71, 50, 28, 68, 19, -63, -83, 19, -65, -40, 19, 44, -78, 19, -26, 81, 49, -83, 67, 6, 18, 55, -56, -37, 62, 57, -6, -64, 4, 17, -40, -1, 55, -23, 35, -45, 44, -75, -64, -10, 66, 29, 25, 51, -10, 9, -77, -25, -73, 39, 3, -33, 12, -8, 34, -38, 49, -78, -66, 21, -31, -50, -82, -13, 10, 60, -24, -6, 78, 9, -47, 47, -7, 1, -4, 39, -27, -31, 61, 47, -78, 28, 65, -16, -5, 41, 1, 21, -24, 32, 79, -63, 61, 66, 28, 54, -68, 46, -69, -36, 68, 72, -75, 36, 5, 1, -11, 16, 47, -52, 43, -58, -47, 40, 12, -7, -43, -22, -77, -49, 30, -20, 48, 36, -31, 2, 8, 84, 80, -18, 85, 50, -46, 71, 34, 47, 38, -2, -71, 16, 73, 26, 40, -79, 79, 81, -81, 6, -23, -20, 54, 84, 57, -70, 71, 55, 44, 9, 82, 16, -42, -10, -85, -72, -16, -77, 9, 69, 37, -15, -21, 68, 37, -6, 62, 24, -29, -15, 78, 67, 27, -13, -38, -42, -37, 39, 49, -53, 41, 57, 62, -30, 13, 27, -65, 43, -14, -26, -39, -48, 23, 48, -28, -56, -43, -70, -51, 70, 12, 6, -85, 30, -43, 37, 29, 59, -18, -23, -65, -36, 79, -71, 38, -40, 33, -18, 35, 3, 64, 25, 16, 45, -80, 53, -34, -19, -17, 60, 35, -38, 61, 67, 16, 61, 78, 46, 51, -53, 33, 69, 56, -53, -68, -57, 53, -71, 50, -44, -52, 63, -84, -35, 64, -22, 36, 62, 19, 76, -70, 41, 10, -30, -30, 34, -69, -34, 2, -6, 48, 19, -1, -63, 58, 80, 68, 66, 12, -33, 36, 48, 78, -65, -75, 51, -7, 62, -52, -17, 58, 16, 31, -73, -32, 80, -51, -56, 3, 49, 34, -55, -78, -62, 64, -80, -22, -79, 69, -71, 29, -18, 46, -12, -47, -5, -75, 17, -78, 21, 79, -55, 69, -41, -16, -51, 24, -59, 78, -42, -7, 57, 79, -70, -35, 7, -83, -51, 9, 37, 73, -34, -82, 71, 57, 70, 75, -22, 84, 29, 66, 14, -22, 16, -49, -33, 71, -65, -62, 42, -7, 69, -34, -3, 20, 54, 69, 44, -2, 73, 0, -40, 13, -52, -51, -52, -78, 32, 23, -26, 43, -16, 36, 21, 40, 32, -51, 7, -39, 29, 6, -33, -77, 10, 16, 85, -13, -57, 48, 5, 82, 48, -16, -4, 42, 43, 70, 58, 22, -45, -34, -49, 75, -30, 47, 59, 31, -46, 41, 18, -61, 73, 65, 53, 60, 43, -75, -30, -57, -49, -39, 50, -44, -71, -20, 16, -41, -84, -36, 39, 36, -7, 73, -12, 46, -7, 18, 62, -9, 62, -68, -9, 5, 36, -32, -35, -21, 31, 76, 36, -16, -80, 44, -46, 9, -71, 72, 16, -23, -25, 77, 84, 67, 39, 10, -31, 39, -1, 13, -57, 80, 45, 71, -27, -69, 59, -34, 85, 45, 50, -83, -23, 32, -3, -63, -69, -56, -5, -48, -41, 15, 81, -4, -1, 37, -21, -48, 19, -53, -3, -17, 46, -58, 36, -51, 15, 46, -75, 17, 59, 32, -4, 20, 43, -79, -82, -28, -3, 4, -11, 19, 67, -70, 79, 64, -35, 29, -42, 27, -35, 22, 77, -12, -79, -14, 1, -41, 69, -12, -61, -62, 75, -31, 59, -35, 21, -37, 42, -72, 25, -36, 15, -29, -10, -2, -42, -19, 79, 52, -41, -10, -63, 55, -13, 46, 45, -66, 49, 62, 78, 38, 41, -4, -27, 0, 64, 53, -66, 69, -46, -49, -36, -25, 15, -50, 15, 23, -67, 70, 71, -11, 11, -52, 40, -80, -2, 74, 69, 11, 68, 56, -7, -50, -67, -17, 62, -35, 22, 72, -6, -73, 0, -22, -55, -81, 76, 47, -61, -19, 1, -75, 39, 7, -16, 17, -61, -15, 74, -21, -1, -27, 77, -41, -80, -66, -84, 84, -16, -80, -44, -73, 55, -36, 1, -31, 15, -19, 12, 56, 1, -18, -4, -36, -10, 83, -6, 8, -74, 75, -43, 65, 64, 25, -46, 17, 46, -38, 75, -72, -22, -15, -31, 20, -5, 26, -7, -84, 80, -34, -10, -4, 48, 34, -29, -67, 37, 61, 41, -15, 39, 11, 83, -41, 50, 62, 75, -65, 61, 72, -75, 29, -40, 66, -52, 50, -28, -71, 61, -16, -22, 36, -26, -14, 78, -57, 9, 9, 75, 12, -53, 2, 69, -66, 6, -45, -7, 72, -70, -76, 17, -49, -59, -51, 51, 6, -70, -82, -5, 84, -85, 9, 2, -32, -28, 28, -74, 60, -27, -82, -55, 43, -34, 50, 6, 76, -34, -53, -21, -76, -71, 54, 41, -78, -19, 1, -18, -40, 66, -46, -51, -66, 22, -15, 21, 3, -79, 8, 42, 25, -9, 27, 32, 42, 79, 64, -76, -66, 71, 18, -42, -48, 26, 17, 66, -34, 42, -75, 16, -58, -79, -81, -47, -34, -54, 51, 41, 36, 57, 79, 77, 6, -73, -18, -15, 9, 45, 34, -41, -19, -20, 78, -20, -25}

#define ADD_CONV2D_KERNEL_0_SHIFT (9)

#define ADD_CONV2D_BIAS_0 {106, -21, -91, -96, -96, 99, -107, 95, -108, 22, -84, -108, -47, -48, -108, -92}

#define ADD_CONV2D_BIAS_0_SHIFT (8)

#define BATCH_NORMALIZATION_KERNEL_0 {119, 119, 119, 119, 119, 119, 120, 119, 119, 119, 120, 119, 119, 118, 119, 118}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {109, 107, 110, 111, 109, 107, 106, 108, 110, 107, 108, 110, 108, 111, 110, 110}

#define BATCH_NORMALIZATION_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ADD_CONV2D_OUTPUT_SHIFT 1
#define BATCH_NORMALIZATION_OUTPUT_SHIFT 1

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
static int8_t nnom_input_data[8192];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 8), nnom_input_data);
	layer[1] = model.hook(AddConv2D(16, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(16, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
