#ifndef __NNOM_LOCAL_BATCHNORMALIZATION_H__
#define __NNOM_LOCAL_BATCHNORMALIZATION_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"
#include "nnom_port.h"
#include "nnom_local.h"

void local_batchnormalization_HWC_q7_nonsquare(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t dim_im_in_y,     // input image dimention y
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const q7_t *bias,               // bias
	const nnom_qformat_param_t *bias_shift,                            // bias shifts
  const nnom_qformat_param_t *out_shift,                             // output shift
  const nnom_qtype_t q_type,      // per channel or per tensor
  q7_t *Im_out,                   // output image
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_LOCAL_BATCHNORMALIZATION_H__ */