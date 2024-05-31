#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_shiftconv1d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// SHIFT-CONVOLUTION WITHOUT SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------


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
){
    int i, j, l;
    
    int conv_out;
    int32_t shift_idx_x;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;

    uint8_t bias_lshift = 0;
    uint8_t output_rshift = 0;


    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        wt_loc = i * ch_im_in;
      	bias_lshift = bias_shift[shift_idx];
      	output_rshift = out_shift[shift_idx];
  
        for (j = 0; j < dim_im_out_x; j++)
        {
            if(bias)
                conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
            else
                conv_out = (q31_t) NNOM_ROUND(output_rshift);
   
            for (l = 0; l < ch_im_in; l++)
            {   
                shift_idx_x = stride_x * j + shift[l];
                if ((shift_idx_x >= 0) && (shift_idx_x < dim_im_in_x))
                {
                    in_pix_loc = shift_idx_x * ch_im_in;
                    conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
                }
            }
            Im_out[i + j * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// SHIFT-CONVOLUTION WITH SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------

#ifdef NNOM_USING_CMSIS_NN
q31_t arm_nn_read_q7x4_for_shiftconv1D(const q7_t *Im_in, uint16_t ch_im_in, uint16_t dim_im_in_x, uint16_t x, uint16_t c_start, q7_t *shift){
	q31_t val;
	q7_t table[4]={0,0,0,0};
	uint16_t current_x;
	int i;
	for (i = 0; i < 4; i++){
		current_x = x + shift[i+c_start];
		if ((current_x >= 0) && (current_x < dim_im_in_x)){
			table[i] = Im_in[current_x * ch_im_in + (i+c_start)];
		}
	}
	memcpy(&val, table, 4);
	return (val);
}

void arm_q7_to_q15_reordered_for_shiftconv1D(q7_t *Im_in, q15_t *pDst, uint16_t ch_im_in, uint16_t dim_im_in_x, uint16_t x, q7_t *shift)
{

    uint32_t blkCnt;        /* loop counter */
    uint16_t c_start = 0; //Channel count
    uint16_t current_x;

    q31_t in;
    q31_t in1, in2;

    /* Run the below code for Cortex-M4 and Cortex-M3 */

    /*loop Unrolling */
    blkCnt = ch_im_in >> 2u;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
     ** a second loop below computes the remaining 1 to 3 samples. */
    while (blkCnt > 0u)
    {
        /* C = (q15_t) A << 8 */
        /* convert from q7 to q15 and then store the results in the destination buffer */
        in = arm_nn_read_q7x4_for_shiftconv(Im_in, ch_im_in, dim_im_in_x, x, c_start, shift);

        /* rotatate in by 8 and extend two q7_t values to q15_t values */
        in1 = __SXTB16(__ROR((uint32_t)in, 8));

        /* extend remainig two q7_t values to q15_t values */
        in2 = __SXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
        *__SIMD32(pDst)++ = in2;
        *__SIMD32(pDst)++ = in1;
#else
        *__SIMD32(pDst)++ = in1;
        *__SIMD32(pDst)++ = in2;
#endif

        /* Decrement the loop counter */
        blkCnt--;
        c_start+=4;
    }

    /* check if there is left-over for compute */
    blkCnt = blockSize % 0x4u;
    
    while (blkCnt > 0u)
    {
        /* C = (q15_t) A << 8 */
        /* convert from q7 to q15 and then store the results in the destination buffer */
        current_x = x + shift[i+c_start];
        *pDst++ = (q15_t)Im_in[current_x * ch_im_in + c_start];

        /* Decrement the loop counter */
        blkCnt--;
        c_start+=1;
    }
}

/* This function is the version with full list of optimization tricks, but with
* some constraints:
*   ch_im_in is multiple of 4
*   ch_im_out is multiple of 2
*
*/

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
                                                 q7_t *bufferB)
{
   (void)bufferB;
#if defined(ARM_MATH_DSP)
   /* Run the following code for Cortex-M4 and Cortex-M7 */
   int16_t i_out_x, i_ch_out, i_ker_x;

   /* -----------------------
    *  Here we use bufferA as q15_t internally as computation are done with q15_t level
    *  im2col are done to output in q15_t format from q7_t input
    */

   q15_t *pBuffer = bufferA;
   q7_t *pOut = Im_out;

   uint32_t size2patches = 2 * ch_im_in;

   if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
   {
       /* check if the input dimension meets the constraints */
       return ARM_MATH_SIZE_MISMATCH;
   }

   for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
   {
       /* This part implements the im2col function */
       i_ker_x = i_out_x * stride_x
       arm_q7_to_q15_reordered_for_shiftconv1D(Im_in, pBuffer, ch_im_in, dim_im_in_x, i_ker_x * ch_im_in, shift);
       pBuffer += ch_im_in;

	     if (pBuffer == bufferA + size2patches)
	     {
  			   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in, bias_shift, out_shift, bias, pOut);
  			   /* counter reset */
  			   pBuffer = bufferA;
       }
   }

   /* check if there is left-over for compute */
   if (pBuffer != bufferA)
   {
       const q7_t *pA = wt;
       for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
       {
           q31_t sum = ((q31_t)(bias[i_ch_out]) << bias_shift) + NN_ROUND(out_shift);
           const q15_t *pB = bufferA;
           /* basically each time it process 4 entries */
           uint16_t colCnt = ch_im_in  >> 2;

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
           colCnt = ch_im_in  & 0x3;
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
#else
   return ARM_MATH_SIZE_MISMATCH; //Create other flag dor this case
#endif /* ARM_MATH_DSP */

   /* Return to application */
   return ARM_MATH_SUCCESS;
}
#endif