#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_batchnormalization.h"

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
)
{
    int i, j, k;
    int batch_out;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;

    uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;
        
    for (i = 0, shift_idx = 0; i < ch_im_in; i++, shift_idx += shift_steps)
    {
        for (j = 0; j < dim_im_in_y; j++)
        {
            for (k = 0; k < dim_im_in_x; k++)
            {
                
                if(bias)
                    batch_out = ((q31_t)(bias[i]) << bias_shift[shift_idx]) + NNOM_ROUND(out_shift[shift_idx]);
                else
                    batch_out = (q31_t) NNOM_ROUND(out_shift[shift_idx]);
                    
                batch_out += Im_in[i + j * im_in_y_dec + k * ch_im_in] * wt[i];
                
                Im_out[i + j * im_in_y_dec + k * ch_im_in] = (q7_t)__NNOM_SSAT((batch_out >> out_shift[shift_idx]), 8);
            }
        }
    }   
}
