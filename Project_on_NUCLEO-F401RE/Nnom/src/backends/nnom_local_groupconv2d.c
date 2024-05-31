#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_groupconv2d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// GROUP-CONVOLUTION  WITH NAIVE ALGORITHM AND WITHOUT SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------


void local_groupconvolve_HWC_q7_nonsquare(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,                                        // input image dimention x
	const uint16_t dim_im_in_y,                                        // input image dimention y
	const uint16_t ch_im_in,                                           // number of input image channels
	const q7_t *wt,                                                    // kernel weights
	const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
  const uint16_t groups,                                             // number of groups
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
        
    uint16_t filter_by_group = ch_im_out/groups;
    uint16_t channel_by_group = ch_im_in/groups;

    uint32_t weight_y_dec = dim_kernel_x * channel_by_group;
    uint32_t weight_ch_im_out_dec = channel_by_group * dim_kernel_y * dim_kernel_x;

    uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;

    uint32_t im_out_y_dec = dim_im_out_x * ch_im_out;

    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        uint16_t group = i/filter_by_group;
        uint16_t base_channel = group * channel_by_group;
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
                    conv_out = ((q31_t)(bias[i]) << bias_shift[shift_idx]) + NNOM_ROUND(out_shift[shift_idx]);
                else
                    conv_out = (q31_t) NNOM_ROUND(out_shift[shift_idx]);

                for (m = ker_y_start; m < ker_y_end; m++)
                {
                    for (n = ker_x_start; n < ker_x_end; n++)
                    {
                        in_row = stride_y * j + m * dilation_y - padding_y;
                        in_col = stride_x * k + n * dilation_x - padding_x;

                        // pre-calculate the pixel location and weight location to improve the performance.
                        in_pix_loc = in_row * im_in_y_dec + in_col * ch_im_in;
                        wt_loc = i * weight_ch_im_out_dec + m * weight_y_dec + n * channel_by_group;
                        for (l = 0; l < channel_by_group; l++)
                        {
                            conv_out += Im_in[in_pix_loc + base_channel + l] * wt[wt_loc + l];
                        } 
                    }
                }
                Im_out[i + j * im_out_y_dec + k * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
//GROUP-CONVOLUTION  WITH IM2COL ALGORITHM AND NO SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------
q7_t* local_mat_mult_kernel_for_group_conv(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut,
											  const int16_t dec_pOut)
{


    /* set up the second output pointers */
    q7_t *pOut2 = pOut + dec_pOut;
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
      
      			sum += inA1 * inB1;
      			sum2 += inA1 * inB2;
      			sum3 += inA2 * inB1;
      			sum4 += inA2 * inB2;

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

    /* return the new output pointer with offset */
    return pOut;
}

int local_groupconvolve_HWC_q7_nonsquare_im2col(const q7_t *Im_in,
                                             const uint16_t dim_im_in_x,
                                             const uint16_t dim_im_in_y,
                                             const uint16_t ch_im_in,
                                             const q7_t *wt,
                                             const uint16_t ch_im_out,
											 const uint16_t groups,                                             // number of groups
                                             const uint16_t dim_kernel_x,
                                             const uint16_t dim_kernel_y,
                                             const uint16_t padding_x,
                                             const uint16_t padding_y,
                                             const uint16_t stride_x,
                                             const uint16_t stride_y,
                                             const q7_t *bias,
                                             const uint16_t bias_shift,
                                             const uint16_t out_shift,
                                             q7_t *Im_out,
                                             const uint16_t dim_im_out_x,
                                             const uint16_t dim_im_out_y,
                                             q7_t *bufferA,
                                             q7_t *bufferB)
{
   (void)bufferB;
   /* Run the following code for Cortex-M4 and Cortex-M7 */

   int16_t i_out_y, i_out_x, i_ker_y, i_ker_x, i_group;

   /* -----------------------
    *  Here we use bufferA as q15_t internally as computation are done with q15_t level
    *  im2col are done to output in q15_t format from q7_t input
    */

   q7_t *pBuffer = bufferA;
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

   uint16_t filter_by_group = ch_im_out/groups;
   uint16_t channel_by_group = ch_im_in/groups;

   uint32_t weight_y_dec = dim_kernel_x * channel_by_group;
   uint32_t weight_ch_im_out_dec = channel_by_group * dim_kernel_y * dim_kernel_x;
   uint32_t group_dec = channel_by_group * dim_kernel_y * dim_kernel_x * filter_by_group;

   uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;
   uint32_t im_out_y_dec = dim_im_out_x * ch_im_out;

   uint32_t size2patches = 2 * channel_by_group * dim_kernel_x * dim_kernel_y;
   uint32_t dec_output = (groups-1) * filter_by_group + ch_im_out;

   /*
    *  Here we split the entire matrix into three regions depending on the padding situation
    *    Top: i_out_y from 0 to padding - 1
    * Middle: i_out_y from padding to dim_im_out-padding-1
    * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
    */
   /* For each group */
   for (i_group = 0; i_group < groups; i_group++)
   {
	   pOut = Im_out + i_group*filter_by_group;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q7_t) * channel_by_group);
					   }
					   else
					   {
						   memcpy(pBuffer, (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, sizeof(q7_t) * channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q7_t) * channel_by_group);
					   }
					   else
					   {
						   memcpy(pBuffer, (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, sizeof(q7_t) * channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
				   for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
						i_ker_x++)
				   {
					   memcpy(pBuffer, (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, sizeof(q7_t) * channel_by_group);
             pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q7_t) * channel_by_group);
					   }
					   else
					   {
						   memcpy(pBuffer, (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, sizeof(q7_t) * channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q7_t) * channel_by_group);
					   }
					   else
					   {
						   memcpy(pBuffer, (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, sizeof(q7_t) * channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
				   pBuffer = bufferA;
			   }
		   }
	   }
   }
   
   //----------------------------------------------------------------------------
   // TO DO: LEFT-OVER CASE (IF (dim_im_out_x*dim_im_out_y)%2!=0) Verify if group don't mess with it
   //----------------------------------------------------------------------------

   /* Return to application */
   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// GROUP-CONVOLUTION WITH IM2COL ALGORITHM AND SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------
/*
 * Custom mat mult for group conv
 */

#ifdef NNOM_USING_CMSIS_NN
/*
 * Custom mat mult for group conv
 */
q7_t *arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(const q7_t *pA,
                                              const q15_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
                                              const uint16_t out_shift,
                                              const q7_t *bias,
                                              q7_t *pOut,
					      const int16_t dec_pOut)
{

#if defined(ARM_MATH_DSP)
    /* set up the second output pointers */
    q7_t *pOut2 = pOut + dec_pOut;
    int i;

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        const q15_t *pB = pInBuffer;
        const q15_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const q7_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        q31_t sum = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum2 = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum3 = ((q31_t)(bias[i + 1]) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum4 = ((q31_t)(bias[i + 1]) << bias_shift) + NN_ROUND(out_shift);

        uint16_t colCnt = numCol_A >> 2;
        /* accumulate over the vector */
        while (colCnt)
        {
            q31_t inA11, inA12, inA21, inA22;

            q31_t inB1 = arm_nn_read_q15x2_ia(&pB);
            q31_t inB2 = arm_nn_read_q15x2_ia(&pB2);

            pA = read_and_pad_reordered(pA, &inA11, &inA12);
            pA2 = read_and_pad_reordered(pA2, &inA21, &inA22);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = arm_nn_read_q15x2_ia(&pB);
            inB2 = arm_nn_read_q15x2_ia(&pB2);

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            colCnt--;
        } /* while over colCnt */
        
        //----------------------------------------------------------------------------
        // TO DO: LEFT-OVER CASE (IF (channel_by_group * dim_kernel_y * dim_kernel_x)%4!=0)
        //----------------------------------------------------------------------------
        *pOut++ = (q7_t)__SSAT((sum >> out_shift), 8);
        *pOut++ = (q7_t)__SSAT((sum3 >> out_shift), 8);
        *pOut2++ = (q7_t)__SSAT((sum2 >> out_shift), 8);
        *pOut2++ = (q7_t)__SSAT((sum4 >> out_shift), 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    } /* for over ch_im_out */

    /* return the new output pointer with offset */
    return pOut;
#else
    (void)pA;
    (void)pInBuffer;
    (void)ch_im_out;
    (void)numCol_A;
    (void)bias_shift;
    (void)out_shift;
    (void)bias;
    (void)pOut;
    /* To be completed */
    return NULL;
#endif /* ARM_MATH_DSP */
}

arm_status arm_group_convolve_HWC_q7_fast_nonsquare(const q7_t *Im_in,
                                             const uint16_t dim_im_in_x,
                                             const uint16_t dim_im_in_y,
                                             const uint16_t ch_im_in,
                                             const q7_t *wt,
                                             const uint16_t ch_im_out,
											 const uint16_t groups,                                             // number of groups
                                             const uint16_t dim_kernel_x,
                                             const uint16_t dim_kernel_y,
                                             const uint16_t padding_x,
                                             const uint16_t padding_y,
                                             const uint16_t stride_x,
                                             const uint16_t stride_y,
                                             const q7_t *bias,
                                             const uint16_t bias_shift,
                                             const uint16_t out_shift,
                                             q7_t *Im_out,
                                             const uint16_t dim_im_out_x,
                                             const uint16_t dim_im_out_y,
                                             q15_t *bufferA,
                                             q7_t *bufferB)
{
   (void)bufferB;
#if defined(ARM_MATH_DSP)
   /* Run the following code for Cortex-M4 and Cortex-M7 */

   int16_t i_out_y, i_out_x, i_ker_y, i_ker_x, i_group;

   /* -----------------------
    *  Here we use bufferA as q15_t internally as computation are done with q15_t level
    *  im2col are done to output in q15_t format from q7_t input
    */

   q15_t *pBuffer = bufferA;
   q7_t *pOut = Im_out;
   if (ch_im_in % groups != 0 || ch_im_out % groups != 0)
   {
	   /* check if the input dimension meets the constraints */
	   return ARM_MATH_SIZE_MISMATCH;
   }
   else{
	   if ((ch_im_in/groups) % 4 != 0 || (ch_im_out/groups) % 2 != 0)
	   {
	          /* check if the input dimension meets the constraints */
	          return ARM_MATH_SIZE_MISMATCH;
	   }
   }

   uint16_t filter_by_group = ch_im_out/groups;
   uint16_t channel_by_group = ch_im_in/groups;

   uint32_t weight_y_dec = dim_kernel_x * channel_by_group;
   uint32_t weight_ch_im_out_dec = channel_by_group * dim_kernel_y * dim_kernel_x;
   uint32_t group_dec = channel_by_group * dim_kernel_y * dim_kernel_x * filter_by_group;

   uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;
   uint32_t im_out_y_dec = dim_im_out_x * ch_im_out;

   uint32_t size2patches = 2 * channel_by_group * dim_kernel_x * dim_kernel_y;
   uint32_t dec_output = (groups-1) * filter_by_group + ch_im_out;

   /*
    *  Here we split the entire matrix into three regions depending on the padding situation
    *    Top: i_out_y from 0 to padding - 1
    * Middle: i_out_y from padding to dim_im_out-padding-1
    * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
    */
   /* For each group */
   for (i_group = 0; i_group < groups; i_group++)
   {
	   pOut = Im_out + i_group*filter_by_group;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q15_t) * channel_by_group);
					   }
					   else
					   {
						   arm_q7_to_q15_reordered_no_shift(
							   (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, pBuffer, channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q15_t) * channel_by_group);
					   }
					   else
					   {
						   arm_q7_to_q15_reordered_no_shift(
							   (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, pBuffer, channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
				   for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
						i_ker_x++)
				   {
					   arm_q7_to_q15_reordered_no_shift(
							   (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, pBuffer, channel_by_group);
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q15_t) * channel_by_group);
					   }
					   else
					   {
						   arm_q7_to_q15_reordered_no_shift(
							   (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, pBuffer, channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
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
						   /* arm_fill_q15(0, pBuffer, channel_by_group); */
						   memset(pBuffer, 0, sizeof(q15_t) * channel_by_group);
					   }
					   else
					   {
						   arm_q7_to_q15_reordered_no_shift(
							   (q7_t *)Im_in + i_ker_y * im_in_y_dec + i_ker_x * ch_im_in + i_group * channel_by_group, pBuffer, channel_by_group);
					   }
					   pBuffer += channel_by_group;
				   }
			   }
			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered_for_group_conv(
						   (q7_t *)wt + (group_dec * i_group), bufferA, filter_by_group, weight_ch_im_out_dec, bias_shift, out_shift, (q7_t *)bias + (filter_by_group * i_group), pOut, ch_im_out);
				   /* counter reset */
				   pOut = pOut + dec_output;
				   pBuffer = bufferA;
			   }
		   }
	   }
   }
   
   //----------------------------------------------------------------------------
   // TO DO: LEFT-OVER CASE (IF (dim_im_out_x*dim_im_out_y)%2!=0) Verify if group don't mess with it
   //----------------------------------------------------------------------------

#else
   return ARM_MATH_SIZE_MISMATCH;
#endif /* ARM_MATH_DSP */

   /* Return to application */
   return ARM_MATH_SUCCESS;
}

#endif
