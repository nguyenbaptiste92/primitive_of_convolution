#ifndef __NNOM_LOCAL_ADDCONV2D_H__
#define __NNOM_LOCAL_ADDCONV2D_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "stdint.h"
#include "nnom_port.h"
#include "nnom_local.h"

#ifdef ARM_NN_TRUNCATE
#define NNOM_TRUNCATE
#endif

#define ABS(x) ((x) > 0 ? (x) : -(x))

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// ADD-CONVOLUTION WITHOUT FUSED BATCH NORMALIZATION WITHOUT DPS
//----------------------------------------------------------------------------------------------------------------------------------------------------------

void local_add_convolve_HWC_q7_nonsquare_0(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t dim_im_in_y,     // input image dimention y
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,    // filter kernel size x
	const uint16_t dim_kernel_y,    // filter kernel size y
	const uint16_t padding_x,       // padding sizes x
	const uint16_t padding_y,       // padding sizes y
	const uint16_t stride_x,        // stride x
	const uint16_t stride_y,        // stride y
  const uint16_t dilation_x,      // dilation x
	const uint16_t dilation_y,      // dilation y
	const q7_t *bias,               // bias
	const nnom_qformat_param_t *bias_shift,                            // bias shifts
  const nnom_qformat_param_t *out_shift,                             // output shift
  const nnom_qformat_param_t *inter_lshift,                             // intermediary shift
  const nnom_qtype_t q_type,      // per channel or per tensor
  q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	const uint16_t dim_im_out_y,    // output image dimension y
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
);

void local_add_convolve_HWC_q7_nonsquare_1(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t dim_im_in_y,     // input image dimention y
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,    // filter kernel size x
	const uint16_t dim_kernel_y,    // filter kernel size y
	const uint16_t padding_x,       // padding sizes x
	const uint16_t padding_y,       // padding sizes y
	const uint16_t stride_x,        // stride x
	const uint16_t stride_y,        // stride y
  const uint16_t dilation_x,      // dilation x
	const uint16_t dilation_y,      // dilation y
	const q7_t *bias,               // bias
	const nnom_qformat_param_t *bias_shift,                            // bias shifts
  const nnom_qformat_param_t *out_shift,                             // output shift
  const nnom_qformat_param_t *inter_lshift,                             // intermediary shift
  const nnom_qtype_t q_type,      // per channel or per tensor
  q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	const uint16_t dim_im_out_y,    // output image dimension y
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
);

void local_add_convolve_HWC_q7_nonsquare_2(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t dim_im_in_y,     // input image dimention y
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,    // filter kernel size x
	const uint16_t dim_kernel_y,    // filter kernel size y
	const uint16_t padding_x,       // padding sizes x
	const uint16_t padding_y,       // padding sizes y
	const uint16_t stride_x,        // stride x
	const uint16_t stride_y,        // stride y
  const uint16_t dilation_x,      // dilation x
	const uint16_t dilation_y,      // dilation y
	const q7_t *bias,               // bias
	const nnom_qformat_param_t *bias_shift,                            // bias shifts
  const nnom_qformat_param_t *out_shift,                             // output shift
  const nnom_qformat_param_t *inter_lshift,                             // intermediary shift
  const nnom_qtype_t q_type,      // per channel or per tensor
  q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	const uint16_t dim_im_out_y,    // output image dimension y
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
);

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// ADD-CONVOLUTION WITHOUT FUSED BATCH NORMALIZATION WITHOUT DPS : THE EQUIVALENT OF __SMLAD Instruction don't exist for signed integer absolute difference.
// The closest instructions is __SSUB16 or __USADA8 (https://www.keil.com/pack/doc/CMSIS/Core/html/group__intrinsic__SIMD__gr.html#gad032bd21f013c5d29f5fcb6b0f02bc3f)
//----------------------------------------------------------------------------------------------------------------------------------------------------------

q7_t* local_mat_mult_kernel_for_addconv_0(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut);

q7_t* local_mat_mult_kernel_for_addconv_1(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut);

q7_t* local_mat_mult_kernel_for_addconv_2(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut);

int local_addconvolve_HWC_q7_nonsquare_im2col(const q7_t *Im_in,
                                              const uint16_t dim_im_in_x,
                                              const uint16_t dim_im_in_y,
                                              const uint16_t ch_im_in,
                                              const q7_t *wt,
                                              const uint16_t ch_im_out,
                                              const uint16_t dim_kernel_x,
                                              const uint16_t dim_kernel_y,
                                              const uint16_t padding_x,
                                              const uint16_t padding_y,
                                              const uint16_t stride_x,
                                              const uint16_t stride_y,
                                              const q7_t *bias,
                                              const uint16_t bias_shift,
                                              const uint16_t out_shift,
											  const uint32_t mode,
											  const uint32_t inter_shift,// intermediary shift
                                              q7_t *Im_out,
                                              const uint16_t dim_im_out_x,
                                              const uint16_t dim_im_out_y,
                                              q7_t *bufferA,
                                              q7_t *bufferB);

#ifdef __cplusplus
}
#endif

#endif  /* __NNOM_LOCAL_ADDCONV2D_H__ */
