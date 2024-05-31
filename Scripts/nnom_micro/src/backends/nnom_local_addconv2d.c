#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_addconv2d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// ADD-CONVOLUTION WITHOUT FUSED BATCH NORMALIZATION WITH NAIVE ALGORITHM
// THERE IS 3 FUNCTIONS DEPENDING IF THE INPUT SHIFT IS SUPERIOR TO WEIGHT SHIFT
//----------------------------------------------------------------------------------------------------------------------------------------------------------

void local_add_convolve_HWC_q7_nonsquare_0(const q7_t *Im_in,                // input image
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

    uint8_t bias_lshift = 0;
    uint8_t output_rshift = 0;
        
    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
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
                
                if(bias)
                    conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
                else
                    conv_out = (q31_t) NNOM_ROUND(output_rshift);
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
                                conv_out -= ABS(Im_in[in_pix_loc + l] - wt[wt_loc + l]);
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
                
                Im_out[i + j * im_out_y_dec + k * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
            }
        }
    }   
}

void local_add_convolve_HWC_q7_nonsquare_1(const q7_t *Im_in,                // input image
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
	const q7_t *bias,                                               // bias
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
                
                if(bias)
                    conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
                else
                    conv_out = (q31_t) NNOM_ROUND(output_rshift);
                    
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
                                //conv_out -= ABS(Im_in[in_pix_loc + l] - (wt[wt_loc + l]>> inter_shift));
                            } 
                        }
                        else{
                            // pre-calculate the weight location to improve the performance.
                            wt_loc = i * weight_ch_im_out_dec + m * weight_y_dec + n * ch_im_in;
                        
                            for (l = 0; l < ch_im_in; l++)
                            {    
                                conv_out -= ABS(wt[wt_loc + l]);
                                //conv_out -= ABS(wt[wt_loc + l]>> inter_shift);
                            } 
                        }
                    }
                }

                Im_out[i + j * im_out_y_dec + k * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
            }
        }
    }
}

void local_add_convolve_HWC_q7_nonsquare_2(const q7_t *Im_in,                // input image
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
                
                if(bias)
					conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
				else
					conv_out = (q31_t) NNOM_ROUND(output_rshift);

                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_y; n++)
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
                                conv_out -= ABS(Im_in[in_pix_loc + l] - ((q31_t)(wt[wt_loc + l]) << inter_shift));
                            } 
                        }
                        else{
                            // pre-calculate the weight location to improve the performance.
                        	wt_loc = i * weight_ch_im_out_dec + m * weight_y_dec + n * ch_im_in;
                        
                            for (l = 0; l < ch_im_in; l++)
                            {    
                                conv_out -= ABS(((q31_t)(wt[wt_loc + l]) << inter_shift));
                            } 
                        }
                    }
                }
                
                Im_out[i + j * im_out_y_dec + k * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
            }
        }
    }   
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// ADD-CONVOLUTION WITH IM2COL ALGORITHM BUT WITHOUT DSP : THE EQUIVALENT OF __SMLAD Instruction don't exist for signed integer absolute difference.
// The closest instructions is __SSUB16 or __USADA8 (https://www.keil.com/pack/doc/CMSIS/Core/html/group__intrinsic__SIMD__gr.html#gad032bd21f013c5d29f5fcb6b0f02bc3f)
// THERE IS 3 FUNCTIONS DEPENDING IF THE INPUT SHIFT IS SUPERIOR TO WEIGHT SHIFT
//----------------------------------------------------------------------------------------------------------------------------------------------------------

q7_t* local_mat_mult_kernel_for_addconv_0(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut)
{


    /* set up the second output pointers */
    q7_t *pOut2 = pOut + ch_im_out;
    int i;

    /* set up pointers and variable*/
    q7_t inA1,inB1,inA2,inB2;
    q31_t sum, sum2, sum3, sum4;

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        const q7_t *pB = pInBuffer;
        const q7_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const q7_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        sum = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum2 = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum3 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);
        sum4 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);

        uint16_t colCnt = numCol_A;
        /* accumulate over the vector */
        while (colCnt)
        {
      			inA1 = *pA++;
      			inB1 = *pB++;
      			inA2 = *pA2++;
      			inB2 = *pB2++;
      
      			sum -= ABS(inB1 - inA1);
      			sum2 -= ABS(inB2 - inA1);
      			sum3 -= ABS(inB1 - inA2);
      			sum4 -= ABS(inB2 - inA2);
  
            colCnt--;
        } /* while over colCnt */
        
        //----------------------------------------------------------------------------
        // NO LEFT-OVER CASE BECAUSE THE LOOP IS ON ch_im_in AND NOT ch_im_in >> 2
        //----------------------------------------------------------------------------
        
        *pOut++ = (q7_t)__NNOM_SSAT((sum >> out_shift), 8);
        *pOut++ = (q7_t)__NNOM_SSAT((sum3 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum2 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum4 >> out_shift), 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    } /* for over ch_im_out */

    pOut += ch_im_out;
    /* return the new output pointer with offset */
    return pOut;
}

q7_t* local_mat_mult_kernel_for_addconv_1(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut)
{


    /* set up the second output pointers */
    q7_t *pOut2 = pOut + ch_im_out;
    int i;

    /* set up pointers and variable*/
    q7_t inA1,inB1,inA2,inB2;
    q31_t sum, sum2, sum3, sum4;

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        const q7_t *pB = pInBuffer;
        const q7_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const q7_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        sum = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum2 = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum3 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);
        sum4 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);

        uint16_t colCnt = numCol_A;
        /* accumulate over the vector */
        while (colCnt)
        {
      			inA1 = *pA++;
      			inB1 = *pB++;
      			inA2 = *pA2++;
      			inB2 = *pB2++;
      
      			sum -= ABS(((q31_t)(inB1<< inter_shift)) - inA1);
      			sum2 -= ABS(((q31_t)(inB2<< inter_shift)) - inA1);
      			sum3 -= ABS(((q31_t)(inB1<< inter_shift)) - inA2);
      			sum4 -= ABS(((q31_t)(inB2<< inter_shift)) - inA2);
      
            colCnt--;
        } /* while over colCnt */
        
        //----------------------------------------------------------------------------
        // NO LEFT-OVER CASE BECAUSE THE LOOP IS ON ch_im_in AND NOT ch_im_in >> 2
        //----------------------------------------------------------------------------
        
        *pOut++ = (q7_t)__NNOM_SSAT((sum >> out_shift), 8);
        *pOut++ = (q7_t)__NNOM_SSAT((sum3 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum2 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum4 >> out_shift), 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    } /* for over ch_im_out */

    pOut += ch_im_out;
    /* return the new output pointer with offset */
    return pOut;
}

q7_t* local_mat_mult_kernel_for_addconv_2(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
											  const uint16_t inter_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut)
{


    /* set up the second output pointers */
    q7_t *pOut2 = pOut + ch_im_out;
    int i;

    /* set up pointers and variable*/
    q7_t inA1,inB1,inA2,inB2;
    q31_t sum, sum2, sum3, sum4;

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        const q7_t *pB = pInBuffer;
        const q7_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const q7_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        sum = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum2 = ((q31_t)(bias[i]) << bias_shift) + NNOM_ROUND(out_shift);
        sum3 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);
        sum4 = ((q31_t)(bias[i + 1]) << bias_shift) + NNOM_ROUND(out_shift);

        uint16_t colCnt = numCol_A;
        /* accumulate over the vector */
        while (colCnt)
        {
      			inA1 = *pA++;
      			inB1 = *pB++;
      			inA2 = *pA2++;
      			inB2 = *pB2++;
      
      			sum -= ABS(inB1 - ((q31_t)(inA1) << inter_shift));
      			sum2 -= ABS(inB2 - ((q31_t)(inA1) << inter_shift));
      			sum3 -= ABS(inB1 - ((q31_t)(inA2) << inter_shift));
      			sum4 -= ABS(inB2 - ((q31_t)(inA2) << inter_shift));

            colCnt--;
        } /* while over colCnt */
        
        //----------------------------------------------------------------------------
        // NO LEFT-OVER CASE BECAUSE THE LOOP IS ON ch_im_in AND NOT ch_im_in >> 2
        //----------------------------------------------------------------------------
        
        *pOut++ = (q7_t)__NNOM_SSAT((sum >> out_shift), 8);
        *pOut++ = (q7_t)__NNOM_SSAT((sum3 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum2 >> out_shift), 8);
        *pOut2++ = (q7_t)__NNOM_SSAT((sum4 >> out_shift), 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    } /* for over ch_im_out */

    pOut += ch_im_out;
    /* return the new output pointer with offset */
    return pOut;
}

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
                                              q7_t *bufferB)
{
    (void)bufferB;

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    uint32_t size2patches = 2 * ch_im_in * dim_kernel_x * dim_kernel_y;
    uint32_t sizepatch = ch_im_in * dim_kernel_x * dim_kernel_y;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q7_t *pBuffer = bufferA;
    q7_t *pOut = Im_out;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return -3;
    }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < padding_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                 i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                     i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q7_t) * ch_im_in);
                    }
                    else
                    {
                    	memcpy(pBuffer, (q7_t *)Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, sizeof(q7_t) * ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + size2patches)
            {
            	if (mode==0){
            		pOut = local_mat_mult_kernel_for_addconv_0(
            				wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else if (mode==1){
            		pOut = local_mat_mult_kernel_for_addconv_1(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else{
            		pOut = local_mat_mult_kernel_for_addconv_2(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out_y - padding_y; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < padding_x; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                 i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                     i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q7_t) * ch_im_in);
                    }
                    else
                    {
                        memcpy(pBuffer, (q7_t *)Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, sizeof(q7_t) * ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + size2patches)
            {
            	if (mode==0){
            		pOut = local_mat_mult_kernel_for_addconv_0(
            				wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else if (mode==1){
            		pOut = local_mat_mult_kernel_for_addconv_1(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else{
            		pOut = local_mat_mult_kernel_for_addconv_2(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out_x - padding_x; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                 i_ker_y++)
            {
                memcpy(pBuffer, (q7_t *)Im_in + (i_ker_y * dim_im_in_x + i_out_x * stride_x - padding_x) * ch_im_in, sizeof(q7_t) * ch_im_in * dim_kernel_x);
                pBuffer += ch_im_in * dim_kernel_x;
            }

            if (pBuffer == bufferA + size2patches)
            {
            	if (mode==0){
            		pOut = local_mat_mult_kernel_for_addconv_0(
            				wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else if (mode==1){
            		pOut = local_mat_mult_kernel_for_addconv_1(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else{
            		pOut = local_mat_mult_kernel_for_addconv_2(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* right part */
        for (; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                 i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                     i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q7_t) * ch_im_in);
                    }
                    else
                    {
                        memcpy(pBuffer, (q7_t *)Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, sizeof(q7_t) * ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + size2patches)
            {
            	if (mode==0){
            		pOut = local_mat_mult_kernel_for_addconv_0(
            				wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else if (mode==1){
            		pOut = local_mat_mult_kernel_for_addconv_1(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else{
            		pOut = local_mat_mult_kernel_for_addconv_2(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    for (; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y;
                 i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
                     i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q7_t) * ch_im_in);
                    }
                    else
                    {
                        memcpy(pBuffer, (q7_t *)Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, sizeof(q7_t) * ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + size2patches)
            {
            	if (mode==0){
            		pOut = local_mat_mult_kernel_for_addconv_0(
            				wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else if (mode==1){
            		pOut = local_mat_mult_kernel_for_addconv_1(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
            	else{
            		pOut = local_mat_mult_kernel_for_addconv_2(
            		        wt, bufferA, ch_im_out, sizepatch, bias_shift, inter_shift, out_shift, bias, pOut);
            	}
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }
    
    //----------------------------------------------------------------------------
    // TO DO: LEFT-OVER CASE (IF (dim_im_out_x*dim_im_out_y)%2!=0)
    //----------------------------------------------------------------------------


    /* Return to application */
    return 0;
}
