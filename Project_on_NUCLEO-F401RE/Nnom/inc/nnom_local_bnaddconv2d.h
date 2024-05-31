#ifndef __NNOM_LOCAL_BNADDCONV2D_H__
#define __NNOM_LOCAL_BNADDCONV2D_H__

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
// ADD-CONVOLUTION FUSED WITH BATCH NORMALIZATION
//----------------------------------------------------------------------------------------------------------------------------------------------------------

void local_bnadd_convolve_HWC_q7_nonsquare_0(const q7_t *Im_in,                // input image
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
  const q7_t * bn_factor,         // bn_factor
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

void local_bnadd_convolve_HWC_q7_nonsquare_1(const q7_t *Im_in,                // input image
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
  const q7_t * bn_factor,         // bn_factor
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

void local_bnadd_convolve_HWC_q7_nonsquare_2(const q7_t *Im_in,                // input image
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
  const q7_t * bn_factor,         // bn_factor
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




#ifdef __cplusplus
}
#endif

#endif  /* __NNOM_LOCAL_BNADDCONV2D_H__ */