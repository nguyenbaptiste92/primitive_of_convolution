#ifndef __NNOM_LOCAL_SHIFTCONV1D_H__
#define __NNOM_LOCAL_SHIFTCONV1D_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "stdint.h"
#include "nnom_port.h"
#include "nnom_local.h"

#ifdef ARM_NN_TRUNCATE
#define NNOM_TRUNCATE
#endif

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

void local_shift_convolve1D_HWC_q7(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
 	const uint16_t stride_x,        // stride x
	const q7_t *bias,               // bias
 	const q7_t *shift,              // shift
	const nnom_qformat_param_t *bias_shift,  // bias shifts
	const nnom_qformat_param_t *out_shift,   // output shift
	const nnom_qtype_t q_type,      // per channel or per tensor
	q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
);

#ifdef NNOM_USING_CMSIS_NN
q31_t arm_nn_read_q7x4_for_shiftconv1D(const q7_t *Im_in,
	 uint16_t ch_im_in,
	 uint16_t dim_im_in_x,
	 uint16_t x,
	 uint16_t c_start,
	 q7_t *shift
);

void arm_q7_to_q15_reordered_for_shiftconv1D(q7_t *Im_in, 
	q15_t *pDst, 
	uint16_t ch_im_in, 
	uint16_t dim_im_in_x,  
	uint16_t x, 
	q7_t *shift
);

arm_status arm_shift_convolve1D_HWC_q7_fast(const q7_t *Im_in,
                                                 const uint16_t dim_im_in_x,
                                                 const uint16_t ch_im_in,
                                                 const q7_t *wt,
                                                 const uint16_t ch_im_out,
                                                 const uint16_t stride_x,
                                                 const q7_t *bias,
						                                     const q7_t *shift,
                                                 const uint16_t bias_shift,
                                                 const uint16_t out_shift,
                                                 q7_t *Im_out,
                                                 const uint16_t dim_im_out_x,
                                                 q15_t *bufferA,
                                                 q7_t *bufferB
);
#endif

#ifdef __cplusplus
}
#endif

#endif  /* __NNOM_LOCAL_SHIFTCONV1D_H__ */