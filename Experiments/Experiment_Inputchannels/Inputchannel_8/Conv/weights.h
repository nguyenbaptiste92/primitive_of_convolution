#include "nnom.h"

#define CONV2D_KERNEL_0 {71, -21, 19, 62, -24, -30, -49, 70, -12, 39, -70, 55, 20, 9, -40, 27, 36, -77, -41, 29, -17, 58, -2, 15, 34, 51, 30, 36, 76, 22, 58, -12, 1, 76, -30, -71, -2, -55, 28, 51, 28, -5, 10, 57, -55, -68, -8, 1, -42, -53, 66, 21, -45, -39, -31, -81, -72, -70, 30, -82, 21, 27, 27, 57, -26, -26, 85, -31, 69, -64, -48, 18, -73, -51, -6, 57, -2, 66, 48, 66, 78, -76, 62, -18, -31, -47, 84, -59, 71, -44, 72, -73, -82, 22, 27, -58, -67, -39, 18, -29, -56, -17, 62, -35, -80, 80, 18, -16, 80, -26, 46, 81, -69, -60, 78, 39, 28, 54, 44, 57, -20, -31, 7, 45, -57, 30, 78, 19, 9, -72, -77, -13, -22, 37, 51, -65, 85, 46, -62, 9, 32, 43, -17, -8, -40, -85, -23, -14, 9, 80, -55, -51, 74, -49, -33, 33, -76, 0, -58, 76, 45, 17, -4, 29, -22, -41, 54, -18, 68, 19, 44, 55, 17, 69, 66, -34, 22, 47, 50, -46, -11, 8, -55, 41, -14, 1, -72, 72, -22, -42, -26, -26, 16, 26, -76, -42, -70, 3, -22, -56, -59, 27, 20, -8, 9, 31, 55, -51, 45, -1, 73, 66, 17, 76, 41, -47, -18, 77, -31, -32, -76, -18, -46, 56, -28, 68, 21, 53, 33, -20, -75, -85, -64, 60, -25, -24, 1, 60, 35, 59, 63, -81, 71, -64, 68, -1, 75, -80, -36, 4, -13, -82, 61, -3, -77, 72, 36, -23, -50, -27, 23, 49, 73, -56, 33, -32, 9, 21, -30, 41, -25, -6, 26, 35, -82, -17, 40, 73, 0, 22, -30, -59, 2, -17, 52, 27, 76, -39, 3, 40, -31, 4, 64, -28, -13, -33, 27, -22, -43, -76, 58, -26, -76, -41, -11, -26, -80, -57, -60, 22, -44, 25, 48, -34, -20, -85, 58, -53, -25, -80, 29, -85, 42, -70, 17, 6, 13, -38, 37, -28, 21, -49, 76, -58, 49, -39, 1, -59, -62, 2, 17, -14, 37, 51, 43, -32, -34, -52, -34, 74, -16, 36, -40, 16, -37, -82, -45, 53, 63, 77, -1, -64, -9, -58, -54, -11, -29, -40, -20, 25, -41, -64, 33, -34, 41, -19, -85, -38, -79, 31, -2, 17, -71, -15, 76, 68, 26, -10, -60, 7, 8, 32, 30, 82, -15, 43, 25, 51, -72, 15, -66, -81, -12, -32, 30, -57, 81, -83, 73, -14, 68, 13, 40, 37, -1, 58, -27, 11, -15, 1, -34, 12, 6, 11, 81, 1, -67, 38, -36, 59, 37, 71, 40, -80, 76, 22, -8, 50, -85, -8, 4, 13, -75, 26, 52, -75, -27, 77, -46, 15, 48, -49, -47, 38, -6, -20, -1, -16, 13, -27, -67, 35, -61, 80, 28, 64, -40, -36, -66, 68, -18, -78, -1, -85, 4, -30, 65, -1, -75, 67, -6, -81, 33, -15, -55, -66, -42, 65, 80, 64, 64, 81, 32, 11, 41, 63, -16, 14, -74, -2, 37, 9, -43, 59, -38, 69, -1, 66, 67, 69, 63, -38, -16, 80, 38, 42, -31, 84, 10, 55, -6, -2, -69, 11, 1, -20, -18, -75, 82, -52, 25, -82, 42, 69, -33, 22, 39, 62, -10, -59, 36, 0, -45, -71, -68, -60, 0, -7, -40, 27, -38, -73, -50, -57, 63, 72, 29, 42, -64, 66, 41, -64, 52, -10, -65, -67, 76, -69, -62, -30, -61, 7, -27, 18, -26, 62, -46, -77, 9, 10, 67, -81, 42, -53, -31, -56, -70, -83, 37, -84, -2, 69, 33, 81, 54, 23, -75, -18, 13, -78, -37, 48, -71, -22, -71, 40, 8, -8, -34, -73, -66, 79, 45, -79, 81, 36, -29, 18, 2, -3, -84, -65, -10, -6, 22, -45, 52, -12, 63, 8, 47, 34, -70, 78, -72, -58, -51, -69, 32, -71, -62, 40, 34, -31, 32, -58, 15, -35, 33, 31, -7, 56, 47, 64, 35, -15, -5, -29, 44, -16, -47, 64, 79, -38, -61, -24, -72, -68, -73, 21, 35, 6, -52, 15, 28, -80, -2, 14, 39, 80, 14, 61, 64, 34, 28, 61, -70, 57, -78, -20, -32, 37, -37, 30, -20, 8, 54, 63, -59, -55, 4, -6, -7, -81, 85, 80, 36, -11, -57, 73, 1, 29, 62, -5, 31, 46, 79, 38, -78, 81, 64, 43, 68, 54, -25, -49, 20, 7, 60, 0, -55, -79, -60, -58, 22, 66, -8, -30, 56, -54, 60, 10, 54, -78, -38, -16, -41, 21, 29, 5, -26, 72, -11, 70, 19, -36, -26, 50, 67, -81, -10, 29, 57, -10, 79, 35, 43, 26, -21, 72, -81, 85, 84, 82, 12, -79, -43, -31, -37, 1, -19, -48, 21, 76, -36, 25, 48, -80, -66, -79, 13, -68, 16, 31, 74, 1, -32, 63, 28, -70, -52, 29, 19, 26, -34, 60, 14, -63, 79, 3, 24, -48, 57, -16, 44, 53, -17, -10, 4, 3, -45, -61, -31, 68, 14, -38, 6, -29, 69, -61, -23, 7, 27, 74, -40, 65, 61, 81, -36, 85, 56, 36, -60, -76, 26, -30, -8, -8, -35, -24, -26, -10, -32, -65, 63, -1, -63, -77, 23, -41, 57, 58, 23, 6, -7, 0, -27, -4, 47, 40, 23, -40, 5, -6, -28, -54, -59, -11, -36, -5, -72, -26, 33, 16, 42, 16, -1, -49, -42, -19, -84, -19, 35, 5, -41, 78, 8, -16, -24, -26, -74, -12, -73, -48, -65, -11, -48, -19, 28, -33, 14, -38, -8, -79, 24, -45, 71, 72, 79, -35, 16, -64, -56, -80, -13, -42, 43, -13, -37, 5, 44, 4, -17, -75, 36, -63, 2, 72, -19, -23, -63, 84, 85, 25, -78, 74, 83, 72, -46, 7, 25, -13, 70, -28, 28, -21, 11, 74, -35, 56, -9, -33, -56, -53, -51, 50, -62, 29, 5, 72, -74, -22, 7, -22, -22, -39, -25, 44, -38, 76, -32, 71, -47, 71, 8, 8, -76, 10, -13, -36, 14, 53, 66, 67, -37, 33, 54, -21, 59, -55, -84, 80, 17, 23, -38, -75, 28, -73, -69, -78, 74, 38, 66, -33, 80, 75, -41, -80, -16, -31, -46, -8, -30, 82, -43, 45, 35, -7, 32, 71, 47, -23, 24, -3, 82, -6, -41, -47, 80, -9, 41, 28, 28, -71, -53, -58, -69, 12, -55, 79, 49, -30, 30, -3, 53, 49, 5, -37, 68, -82, 5, -84, 22, 71, 25, -37, 36, 14, 63, -29, 71, 85, -1, 84, -6, 55, -35, -42, -63, 48, 46, -55, 54, -85, -75, -17, -78, -47, -3, -81, 9, -52, 44, 54, 6, -27, 36, -19, -33, -50, -21, 30, -84, -2, -54, 6, 24, 25, -21, -21, -52, 14, 6, -10, -20, -40, 73, -54, 35, 33, -50, -59, 35, -24, -19, -34, -15, 51, -13, 21, -15, -76, -76, -32, -36, -56, 82, 82, 75, 63, 69, -14, 6, -15, -60, 23, -34, -80, -64, 37, 40, -75}

#define CONV2D_KERNEL_0_SHIFT (9)

#define CONV2D_BIAS_0 {99, 41, -100, -27, -1, -73, -62, -21, 78, 36, -90, -74, -80, -93, -95, -52}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 5

/* bias shift and output shift for each layer */
#define CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_OUTPUT_SHIFT)
#define CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_BIAS_0_SHIFT)
#if CONV2D_OUTPUT_RSHIFT < 0
#error CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if CONV2D_BIAS_LSHIFT < 0
#error CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
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
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 8), nnom_input_data);
	layer[1] = model.hook(Conv2D(16, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
