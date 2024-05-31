#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_bnaddconv2d.h"

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// ADD-CONVOLUTION FUSED WITH BATCH NORMALIZATION
//----------------------------------------------------------------------------------------------------------------------------------------------------------

void local_bnadd_convolve_HWC_q7_nonsquare_0(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,                                        // input image dimention x
	const uint16_t dim_im_in_y,                                        // input image dimention y
	const uint16_t ch_im_in,                                           // number of input image channels
	const q7_t *wt,                                                    // kernel weights
	const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,                                       // filter kernel size x
	const uint16_t dim_kernel_y,                                       // filter kernel size y
	const uint16_t padding_x,                                          // padding sizes x
	const uint16_t padding_y,                                          // padding sizes y
	const uint16_t stride_x,                                           // stride x
	const uint16_t stride_y,                                           // stride y
  const uint16_t dilation_x,                                         // dilation x
	const uint16_t dilation_y,                                         // dilation y
	const q7_t *bias,                                                  // bias
  const q7_t * bn_factor,         // bn_factor
	const nnom_qformat_param_t *bias_shift,                                        // bias shifts
  const nnom_qformat_param_t *out_shift,                                         // output shift
  const nnom_qformat_param_t *inter_lshift,// intermediary shift
  const nnom_qtype_t q_type,                                         // per channel or per tensor
  q7_t *Im_out,                                                      // output image
	const uint16_t dim_im_out_x,                                       // output image dimension x
	const uint16_t dim_im_out_y,                                       // output image dimension y
	q15_t *bufferA,                                                    //buffer space for input
	q7_t *bufferB                                                      //buffer space for output
)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;
        
    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            int32_t base_idx_y = stride_y * j - padding_y;
            for (k = 0; k < dim_im_out_x; k++)
            {
				        int32_t base_idx_x = stride_x * k - padding_x;
                int32_t ker_y_start = MAX(0, -(base_idx_y-(dilation_y-1))/dilation_y);
                int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
                int32_t ker_y_end = MIN(dim_kernel_y, (dim_im_in_y - base_idx_y + (dilation_y-1))/dilation_y);
                int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);
                
                conv_out = (q31_t) NNOM_ROUND(out_shift[shift_idx]);
                for (m = ker_y_start; m < ker_y_end; m++)
                {
                    for (n = ker_x_start; n < ker_x_end; n++)
                    {
                        in_row = stride_y * j + m * dilation_y - padding_y;
                        in_col = stride_x * k + n * dilation_x - padding_x;

                        // pre-calculate the pixel location and weight location to improve the performance.
                        in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
                        wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in;
                        
                        for (l = 0; l < ch_im_in; l++)
                        {    
                            conv_out -= ABS(Im_in[in_pix_loc + l] - wt[wt_loc + l]);
                        } 
                    }
                }
                
                conv_out *= ((q31_t)(bn_factor[i]));
                
                if(bias)
                    conv_out += ((q31_t)(bias[i]) << bias_shift[shift_idx]);
                    
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }   
}

void local_bnadd_convolve_HWC_q7_nonsquare_1(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,                                        // input image dimention x
	const uint16_t dim_im_in_y,                                        // input image dimention y
	const uint16_t ch_im_in,                                           // number of input image channels
	const q7_t *wt,                                                    // kernel weights
	const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,                                       // filter kernel size x
	const uint16_t dim_kernel_y,                                       // filter kernel size y
	const uint16_t padding_x,                                          // padding sizes x
	const uint16_t padding_y,                                          // padding sizes y
	const uint16_t stride_x,                                           // stride x
	const uint16_t stride_y,                                           // stride y
  const uint16_t dilation_x,                                         // dilation x
	const uint16_t dilation_y,                                         // dilation y
	const q7_t *bias,                                                  // bias
  const q7_t * bn_factor,         // bn_factor
	const nnom_qformat_param_t *bias_shift,                                        // bias shifts
  const nnom_qformat_param_t *out_shift,                                         // output shift
  const nnom_qformat_param_t *inter_lshift,// intermediary shift
  const nnom_qtype_t q_type,                                         // per channel or per tensor
  q7_t *Im_out,                                                      // output image
	const uint16_t dim_im_out_x,                                       // output image dimension x
	const uint16_t dim_im_out_y,                                       // output image dimension y
	q15_t *bufferA,                                                    //buffer space for input
	q7_t *bufferB                                                      //buffer space for output
)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;
        
    uint32_t weight_y_dec = dim_kernel_x * ch_im_in;
    uint32_t weight_ch_im_out_dec = ch_im_in * dim_kernel_y * dim_kernel_x;

    uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;

    uint32_t im_out_y_dec = dim_im_out_x * ch_im_out;

    uint8_t inter_shift = 0;
    uint8_t bias_lshift = 0;
    uint8_t output_rshift = 0;
        
    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
   	    inter_shift = inter_lshift[shift_idx];
    	  bias_lshift = bias_shift[shift_idx];
    	  output_rshift = out_shift[shift_idx];
        for (j = 0; j < dim_im_out_y; j++)
        {
            int32_t base_idx_y = stride_y * j - padding_y;
            int32_t ker_y_start = MAX(0, -(base_idx_y-(dilation_y-1))/dilation_y);
            int32_t ker_y_end = MIN(dim_kernel_y, (dim_im_in_y - base_idx_y + (dilation_y-1))/dilation_y);
            
            for (k = 0; k < dim_im_out_x; k++)
            {
				        int32_t base_idx_x = stride_x * k - padding_x;
                int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
                int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);
                
                conv_out = 0;
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        //If padding is used, if outside of image -> conv_out -= ABS(wt)
                        if ((m >= ker_y_start) && (n >= ker_x_start) && (m < ker_y_end) && (n < ker_x_end))
                        {
                        
                            in_row = stride_y * j + m * dilation_y - padding_y;
                            in_col = stride_x * k + n * dilation_x - padding_x;
                            
                            // pre-calculate the pixel location and weight location to improve the performance.
                            in_pix_loc = in_row * im_in_y_dec + in_col * ch_im_in;
                            wt_loc = i * weight_ch_im_out_dec + m * weight_y_dec + n * ch_im_in;
                            
                            for (l = 0; l < ch_im_in; l++)
                            {    
                                conv_out -= ABS(((q31_t)(Im_in[in_pix_loc + l]) << inter_shift) - wt[wt_loc + l]);
                            } 
                        }
                        else{
                            // pre-calculate the weight location to improve the performance.
                            wt_loc = i * weight_ch_im_out_dec + m * weight_y_dec + n * ch_im_in;
                        
                            for (l = 0; l < ch_im_in; l++)
                            {    
                                conv_out -= ABS(wt[wt_loc + l]);
                            } 
                        }
                    }
                }
                conv_out *= ((q31_t)(bn_factor[i]));
                
                if(bias)
                    conv_out += ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
                else
                    conv_out += (q31_t) NNOM_ROUND(output_rshift);
                    
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }
}

void local_bnadd_convolve_HWC_q7_nonsquare_2(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,                                        // input image dimention x
	const uint16_t dim_im_in_y,                                        // input image dimention y
	const uint16_t ch_im_in,                                           // number of input image channels
	const q7_t *wt,                                                    // kernel weights
	const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,                                       // filter kernel size x
	const uint16_t dim_kernel_y,                                       // filter kernel size y
	const uint16_t padding_x,                                          // padding sizes x
	const uint16_t padding_y,                                          // padding sizes y
	const uint16_t stride_x,                                           // stride x
	const uint16_t stride_y,                                           // stride y
  const uint16_t dilation_x,                                         // dilation x
	const uint16_t dilation_y,                                         // dilation y
	const q7_t *bias,                                                  // bias
  const q7_t * bn_factor,         // bn_factor
	const nnom_qformat_param_t *bias_shift,                                        // bias shifts
  const nnom_qformat_param_t *out_shift,                                         // output shift
  const nnom_qformat_param_t *inter_lshift,// intermediary shift
  const nnom_qtype_t q_type,                                         // per channel or per tensor
  q7_t *Im_out,                                                      // output image
	const uint16_t dim_im_out_x,                                       // output image dimension x
	const uint16_t dim_im_out_y,                                       // output image dimension y
	q15_t *bufferA,                                                    //buffer space for input
	q7_t *bufferB                                                      //buffer space for output
)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;
        
    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            int32_t base_idx_y = stride_y * j - padding_y;
            for (k = 0; k < dim_im_out_x; k++)
            {
				        int32_t base_idx_x = stride_x * k - padding_x;
                int32_t ker_y_start = MAX(0, -(base_idx_y-(dilation_y-1))/dilation_y);
                int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
                int32_t ker_y_end = MIN(dim_kernel_y, (dim_im_in_y - base_idx_y + (dilation_y-1))/dilation_y);
                int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);
                
                conv_out = (q31_t) NNOM_ROUND(out_shift[shift_idx]);
                for (m = ker_y_start; m < ker_y_end; m++)
                {
                    for (n = ker_x_start; n < ker_x_end; n++)
                    {
                        in_row = stride_y * j + m * dilation_y - padding_y;
                        in_col = stride_x * k + n * dilation_x - padding_x;

                        // pre-calculate the pixel location and weight location to improve the performance.
                        in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
                        wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in;
                        
                        for (l = 0; l < ch_im_in; l++)
                        {    
                            conv_out -= ABS(Im_in[in_pix_loc + l] - ((q31_t)(wt[wt_loc + l]) << inter_lshift[shift_idx]));
                        } 
                    }
                }
                
                conv_out *= ((q31_t)(bn_factor[i]));
                
                if(bias)
                    conv_out += ((q31_t)(bias[i]) << bias_shift[shift_idx]);

                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }   
}