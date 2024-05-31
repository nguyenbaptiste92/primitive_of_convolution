#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {-89, 60, 41, -83, -29, -61, -15, 44, -41, -3, -68, -58, 4, -108, 86, 24, 16, 31, -55, -16, -91, -43, 88, -105, -77, -66, 58, 91, -82, 11, 26, -87, -106, -70, 80, 24, -77, 85, 11, 49, -74, -85, -6, 6, -43, -96, -61, 49, 100, 26, -104, -40, -74, -15, -7, -107, 39, 37, 35, -40, -101, 92, -96, -39, 74, -59, 52, -110, -51, 92, -9, 80, 78, -42, 40, 41, 91, -85, 44, 2, -54, 21, 44, -93, -25, -76, -95, -55, 30, -82, 18, -93, 31, 48, 108, 66, -99, 2, 80, 91, -30, -110, -45, 104, -65, -50, 91, 82, 37, -85, 20, 40, -49, -25, 44, 68, 13, 69, 21, 62, -90, 92, -105, 47, -98, -72, -34, -19, 51, 53, -70, 73, 61, -88, -36, 110, -14, -103, 63, -49, -32, 77, 41, 101, -40, -58, 97, 41, -104, -103, -6, 55, 0, -40, 50, -81, 46, 53, 47, 66, -67, -44, 87, 85, -10, -67, 98, 90, -42, 97, -97, -79, -51, -94, 106, -2, -77, -83, -95, -33, 67, 10, -74, 26, 110, 58, -5, -64, -71, 79, -106, -4, 56, 8, 97, 18, 98, -101, -69, 20, 65, -40, 107, -5, 18, -71, 76, 48, -79, -64, 84, -51, -22, -71, 20, -30, 49, -27, 29, 86, -41, 10, 89, -48, 47, -76, -78, 40, -92, -81, -9, -102, -81, -90, -106, -63, -40, -46, 99, 51, -20, 86, 87, 91, 110, 79, 51, -36, 9, -106, -78, -52, 13, -103, 60, 21}

#define ADD_CONV2D_KERNEL_0_SHIFT (8)

#define ADD_CONV2D_BIAS_0 {52, 29, 28, 97, 85, 81, 81, 84, -43, 83, 14, 14, -69, 74, 78, -104}

#define ADD_CONV2D_BIAS_0_SHIFT (8)

#define BATCH_NORMALIZATION_KERNEL_0 {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {98, 99, 99, 98, 89, 96, 93, 94, 92, 92, 96, 99, 91, 94, 104, 94}

#define BATCH_NORMALIZATION_BIAS_0_SHIFT (9)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ADD_CONV2D_OUTPUT_SHIFT 3
#define BATCH_NORMALIZATION_OUTPUT_SHIFT 3

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
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(AddConv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(16, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
