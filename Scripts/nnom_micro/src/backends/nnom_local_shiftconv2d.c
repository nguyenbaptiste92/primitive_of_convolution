#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_shiftconv2d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

int iter_number=0;

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// SHIFT-CONVOLUTION WITH NAIVE ALGORITHM AND WITHOUT SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------


void local_shift_convolve_HWC_q7_nonsquare(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t dim_im_in_y,     // input image dimention y
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
	const q7_t *bias,               // bias
	const q7_t *shift,              // shift
	const nnom_qformat_param_t *bias_shift,    // bias shifts
	const nnom_qformat_param_t *out_shift,    // output shift
	const nnom_qtype_t q_type,      // per channel or per tensor
	q7_t *Im_out,                   // output image
	const uint16_t dim_im_out_x,    // output image dimension x
	const uint16_t dim_im_out_y,    // output image dimension y
	q15_t *bufferA,                 //buffer space for input
	q7_t *bufferB                   //buffer space for output
){
    int i, j, k, l;
    
    int conv_out;
    int32_t shift_idx_x,shift_idx_y;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == NNOM_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;

    uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;

    uint32_t im_out_y_dec = dim_im_out_x * ch_im_out;

    uint8_t bias_lshift = 0;
    uint8_t output_rshift = 0;


    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        wt_loc = i * ch_im_in;
    	bias_lshift = bias_shift[shift_idx];
    	output_rshift = out_shift[shift_idx];
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                if(bias)
                    conv_out = ((q31_t)(bias[i]) << bias_lshift) + NNOM_ROUND(output_rshift);
                else
                    conv_out = (q31_t) NNOM_ROUND(output_rshift);
       
                for (l = 0; l < ch_im_in; l++)
                {   
                    shift_idx_x = k + shift[2*l];
                    shift_idx_y = j + shift[2*l+1];
                    if ((shift_idx_x >= 0) && (shift_idx_y >= 0) && (shift_idx_x < dim_im_in_x) && (shift_idx_y < dim_im_in_y))
                    {
                        in_pix_loc = shift_idx_y * im_in_y_dec + shift_idx_x * ch_im_in;
                        conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
                    }
                }
                Im_out[i + j * im_out_y_dec + k * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> output_rshift), 8);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// SHIFT-CONVOLUTION WITH IM2COL ALGORITHM AND NO SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------

q7_t* local_mat_mult_kernel_for_shiftconv(const q7_t *pA,
                                              const q7_t *pInBuffer,
                                              const uint16_t ch_im_out,
                                              const uint16_t numCol_A,
                                              const uint16_t bias_shift,
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

    pOut += ch_im_out;
    /* return the new output pointer with offset */
    return pOut;
}

int local_shiftconvolve_HWC_q7_nonsquare_im2col(const q7_t *Im_in,
        const uint16_t dim_im_in_x,
        const uint16_t dim_im_in_y,
        const uint16_t ch_im_in,
        const q7_t *wt,
        const uint16_t ch_im_out,
        const q7_t *bias,
		 const q7_t *shift,
        const uint16_t bias_shift,
        const uint16_t out_shift,
        q7_t *Im_out,
        const uint16_t dim_im_out_x,
        const uint16_t dim_im_out_y,
        q7_t *bufferA,
        q7_t *bufferB)
{
	int16_t i_out_y, i_out_x, i_ch_out;

	   /* -----------------------
	    *  Here we use bufferA as q15_t internally as computation are done with q15_t level
	    *  im2col are done to output in q15_t format from q7_t input
	    */

	   q7_t *pBuffer = bufferA;
	   q7_t *pOut = Im_out;

	   q7_t table[ch_im_in];

	   uint16_t current_x;
	   uint16_t current_y;
	   int i;

	   uint32_t size2patches = 2 * ch_im_in;
	   uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;

	   if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
	   {
	       /* check if the input dimension meets the constraints */
	       return -3;
	   }

	   for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
	   {
	       for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
	       {
	           /* This part implements the im2col function */
	    	   for (i = 0; i < ch_im_in; i++){
	    	   		current_x = i_out_x + shift[2*i];
	    	   		current_y = i_out_y + shift[2*i+1];
	    	   		if ((current_x >= 0) && (current_y >= 0) && (current_x < dim_im_in_x) && (current_y < dim_im_in_y)){
	    	   			table[i] = Im_in[current_y  * im_in_y_dec + current_x * ch_im_in + i];
	    	   		}
	    	   		else{
	    	   			table[i] = 0;
	    	   		}
	    	   }
	    	   memcpy(pBuffer, table, ch_im_in);
	           pBuffer += ch_im_in;

			   if (pBuffer == bufferA + size2patches)
			   {
				   pOut = local_mat_mult_kernel_for_shiftconv(wt, bufferA, ch_im_out, ch_im_in, bias_shift, out_shift, bias, pOut);
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
//----------------------------------------------------------------------------------------------------------------------------------------------------------
// SHIFT-CONVOLUTION WITH IM2COL ALGORITHM AND SIMD (DSP)
//----------------------------------------------------------------------------------------------------------------------------------------------------------

#ifdef NNOM_USING_CMSIS_NN
q31_t arm_nn_read_q7x4_for_shiftconv(const q7_t *Im_in, uint16_t ch_im_in, uint16_t dim_im_in_x, uint16_t dim_im_in_y, uint16_t x, uint16_t y, uint16_t c_start, q7_t *shift){
	q31_t val;
	q7_t table[4]={0,0,0,0};
	uint16_t current_x;
	uint16_t current_y;
	int i;
	uint32_t im_in_y_dec = dim_im_in_x * ch_im_in;
	for (i = 0; i < 4; i++){
		current_x = x + shift[2*(i+c_start)];
		current_y = y + shift[2*(i+c_start)+1];
		if ((current_x >= 0) && (current_y >= 0) && (current_x < dim_im_in_x) && (current_y < dim_im_in_y)){
			table[i] = Im_in[current_y  * im_in_y_dec + current_x * ch_im_in + (i+c_start)];
		}
	}
	memcpy(&val, table, 4);
	return (val);
}

void arm_q7_to_q15_reordered_for_shiftconv(q7_t *Im_in, q15_t *pDst, uint16_t ch_im_in, uint16_t dim_im_in_x, uint16_t dim_im_in_y, uint16_t x, uint16_t y, q7_t *shift)
{

    uint32_t blkCnt;        /* loop counter */
    uint16_t c_start = 0; //Channel count

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
        in = arm_nn_read_q7x4_for_shiftconv(Im_in, ch_im_in, dim_im_in_x, dim_im_in_y, x, y, c_start, shift);

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
    //----------------------------------------------------------------------------
    // NO LEFT-OVER CASE BECAUSE ch_im_in%4==0
    //----------------------------------------------------------------------------

}

/* This function is the version with full list of optimization tricks, but with
* some constraints:
*   ch_im_in is multiple of 4
*   ch_im_out is multiple of 2
*
*/

arm_status arm_shift_convolve_HWC_q7_fast_nonsquare(const q7_t *Im_in,
                                                 const uint16_t dim_im_in_x,
                                                 const uint16_t dim_im_in_y,
                                                 const uint16_t ch_im_in,
                                                 const q7_t *wt,
                                                 const uint16_t ch_im_out,
                                                 const q7_t *bias,
												 const q7_t *shift,
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
   int16_t i_out_y, i_out_x, i_ch_out;

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

   for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
   {
       for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
       {
           /* This part implements the im2col function */
           arm_q7_to_q15_reordered_for_shiftconv(Im_in, pBuffer, ch_im_in, dim_im_in_x, dim_im_in_y, i_out_x, i_out_y, shift);
           pBuffer += ch_im_in;

		   if (pBuffer == bufferA + size2patches)
		   {
			   pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in, bias_shift, out_shift, bias, pOut);
			   /* counter reset */
			   pBuffer = bufferA;
           }
       }
   }

   //----------------------------------------------------------------------------
   // LEFT-OVER CASE (IF (dim_im_out_x*dim_im_out_y)%2!=0)
   //----------------------------------------------------------------------------
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
