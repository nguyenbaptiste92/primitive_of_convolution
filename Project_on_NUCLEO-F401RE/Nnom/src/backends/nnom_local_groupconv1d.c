#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_conv1d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// CONVOLUTION WITHOUT DSP
//----------------------------------------------------------------------------------------------------------------------------------------------------------


void local_groupconvolve1D_HWC_q7(const q7_t *Im_in,                // input time series
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters
  const uint16_t groups,
	const uint16_t dim_kernel_x,    // filter kernel size x
	const uint16_t padding_x,       // padding sizes x
	const uint16_t stride_x,        // stride x
  const uint16_t dilation_x,      // dilation x
	const q7_t *bias,               // bias
	const nnom_qformat_param_t *bias_shift,                            // bias shifts
  const nnom_qformat_param_t *out_shift,                             // output shift
  const nnom_qtype_t q_type,      // per channel or per tensor
  q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
)
{
    int i, j, l, n;
    int conv_out;
    int in_row;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;
        
    uint16_t filter_by_group = ch_im_out/groups;
    uint16_t channel_by_group = ch_im_in/groups;

    uint32_t weight_y_dec = dim_kernel_x * channel_by_group;
    
    uint8_t bias_lshift = 0;
    uint8_t output_rshift = 0;

    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
      uint16_t group = i/filter_by_group;
      uint16_t base_channel = group * channel_by_group;
    	bias_lshift = bias_shift[shift_idx];
    	output_rshift = out_shift[shift_idx];
        for (j = 0; j < dim_im_out_x; j++)
        {
            int32_t base_idx_x = stride_x * j - padding_x;
            int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
            int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);
                
            if(bias)
                conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
            else
                conv_out = (q31_t) NNOM_ROUND(output_rshift);
                
            for (n = ker_x_start; n < ker_x_end; n++)
            {
                in_row = stride_x * j + n * dilation_x - padding_x;

                // pre-calculate the pixel location and weight location to improve the performance.
                in_pix_loc = in_row * ch_im_in;
                wt_loc = i * weight_y_dec + n * channel_by_group;
                for (l = 0; l < channel_by_group; l++)
                {
                    conv_out += Im_in[in_pix_loc + base_channel + l] * wt[wt_loc + l];
                } 
            }
            
            Im_out[i + j * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// CONVOLUTION1D WITH DSP AND IM2COL
//----------------------------------------------------------------------------------------------------------------------------------------------------------
/*
 * Custom mat mult for group conv
 */

#ifdef NNOM_USING_CMSIS_NN
/**
 * @brief Fast Q7 convolution function 
 * @param[in]       Im_in        pointer to input tensor (1D)
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       padding_x    padding size x
 * @param[in]       stride_x     convolution stride x
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @param[in,out]   bufferB      pointer to buffer space for output
 * @return     The function returns either
 * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * This function is the version with full list of optimization tricks, but with
 * some constraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 */

arm_status arm_groupconvolve1D_HWC_q7_fast(const q7_t *Im_in,
                                             const uint16_t dim_im_in_x,
                                             const uint16_t ch_im_in,
                                             const q7_t *wt,
                                             const uint16_t ch_im_out,
                                             const uint16_t groups,
                                             const uint16_t dim_kernel_x,
                                             const uint16_t padding_x,
                                             const uint16_t stride_x,
                                             const q7_t *bias,
                                             const uint16_t bias_shift,
                                             const uint16_t out_shift,
                                             q7_t *Im_out,
                                             const uint16_t dim_im_out_x,
                                             q15_t *bufferA,
                                             q7_t *bufferB
)
{
    (void)bufferB;
#if defined(ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_x, i_ker_x;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t *pBuffer = bufferA;
    q7_t *pOut = Im_out;

   if (ch_im_in % groups != 0 || ch_im_out % groups != 0)
   {
	   /* check if the input dimension meets the constraints */
	   return -3;
   }
   else{
	   if ((ch_im_in/groups) % 4 != 0 || (ch_im_out/groups) % 2 != 0)
	   {
	          /* check if the input dimension meets the constraints */
	          return -3;
	   }
   }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Left: i_out_x from 0 to padding - 1
     * Middle: i_out_x from padding to dim_im_out-padding-1
     * Right: i_out_x from dim_im_out-padding to dim_im_out-1
     */
     
     uint32_t size2patches = 2 * ch_im_in * dim_kernel_x;

    /* left part */
    for (i_out_x = 0; i_out_x < padding_x; i_out_x++)
    {
        for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
             i_ker_x++)
        {
            if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
            {
                /* arm_fill_q15(0, pBuffer, ch_im_in); */
                memset(pBuffer, 0, sizeof(q15_t) * ch_im_in);
            }
            else
            {
                arm_q7_to_q15_reordered_no_shift((q7_t *)Im_in + i_ker_x * ch_im_in, pBuffer, ch_im_in);
            }
            pBuffer += ch_im_in;
        }

        if (pBuffer == bufferA + size2patches)
        {
            pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x, bias_shift, out_shift, bias, pOut);
            /* counter reset */
            pBuffer = bufferA;
        }
    }

    /* mid part */
    for (; i_out_x < dim_im_out_x - padding_x; i_out_x++)
    {
        /* This part implements the im2col function */
        arm_q7_to_q15_reordered_no_shift((q7_t *)Im_in + (i_out_x * stride_x - padding_x) * ch_im_in, pBuffer, ch_im_in * dim_kernel_x);
        pBuffer += ch_im_in * dim_kernel_x;

        if (pBuffer == bufferA + size2patches)
        {
            pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x, bias_shift, out_shift, bias, pOut);
            /* counter reset */
            pBuffer = bufferA;
        }
    }

    /* right part */
    for (; i_out_x < dim_im_out_x; i_out_x++)
    {
        /* This part implements the im2col function */
        for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
             i_ker_x++)
        {
            if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
            {
                /* arm_fill_q15(0, pBuffer, ch_im_in); */
                memset(pBuffer, 0, sizeof(q15_t) * ch_im_in);
            }
            else
            {
                arm_q7_to_q15_reordered_no_shift((q7_t *)Im_in + i_ker_x * ch_im_in, pBuffer, ch_im_in);
            }
            pBuffer += ch_im_in;
        }

        if (pBuffer == bufferA + size2patches)
        {
            pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in * dim_kernel_x, bias_shift, out_shift, bias, pOut);
            /* counter reset */
            pBuffer = bufferA;
        }
    }


    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        int i;
        for (i = 0; i < ch_im_out; i++)
        {
            q31_t sum = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
            const q15_t *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t colCnt = ch_im_in * dim_kernel_x >> 2;

            while (colCnt)
            {

                q31_t inA1, inA2;
                q31_t inB1, inB2;

                pA = read_and_pad_reordered(pA, &inA1, &inA2);

                inB1 = arm_nn_read_q15x2_ia(&pB);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = arm_nn_read_q15x2_ia(&pB);
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = (ch_im_in * dim_kernel_x) & 0x3;
            while (colCnt)
            {
                q7_t inA1 = *pA++;
                q15_t inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut = (q7_t)__SSAT((sum >> out_shift), 8);
            pOut++;
        }
    }
    
    /* Return to application */
    return ARM_MATH_SUCCESS;

#else
    return -1;

#endif /* ARM_MATH_DSP */
}

#endif