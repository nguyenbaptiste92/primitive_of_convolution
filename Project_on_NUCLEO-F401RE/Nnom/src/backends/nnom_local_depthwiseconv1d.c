#include "nnom.h"
#include "nnom_local.h"
#include "nnom_local_depthwiseconv1d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// DEPTHWISECONVOLUTION WITHOUT DSP
//----------------------------------------------------------------------------------------------------------------------------------------------------------

void local_depthwise_conv1D_HWC_q7(const q7_t *Im_in,// input image
	const uint16_t dim_im_in_x,     // input image dimention x
	const uint16_t ch_im_in,        // number of input image channels
	const q7_t *wt,                 // kernel weights
	const uint16_t ch_im_out,       // number of filters, i.e., output image channels
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
    int i_out_x, i_ch_out, i_ch_in, i_ch_mult;
    int i_ker_x;
    int i_out = 0;
    int shift_idx, shift_steps;
    int ch_mult = ch_im_out / ch_im_in;
    q31_t conv_out;

    for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
    {
        const int32_t base_idx_x = stride_x * i_out_x - padding_x;
        int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
        int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);
        for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
        {
            for(i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
            {
                i_ch_out = i_ch_mult + i_ch_in * ch_mult;

                shift_idx = q_type == NNOM_QTYPE_PER_AXIS ? i_ch_out : 0;
                if (bias)
                    conv_out = ((q31_t)(bias[i_ch_out]) << bias_shift[shift_idx]) + NNOM_ROUND(out_shift[shift_idx]);
                else
                    conv_out = (q31_t)NNOM_ROUND(out_shift[shift_idx]);

                for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                {
                    const int32_t idx_x = base_idx_x + i_ker_x * dilation_x;
                    int32_t in_pix_loc = idx_x * ch_im_in + i_ch_in;
                    int32_t wt_loc = i_ker_x * (ch_im_in * ch_mult) + i_ch_out;
                    conv_out += Im_in[in_pix_loc] * wt[wt_loc];
                }
                Im_out[i_out++] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------
// DEPTHWISECONVOLUTION1D WITH DSP AND IM2COL
//----------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef NNOM_USING_CMSIS_NN
/**
 * @brief Q7 depthwise separable convolution function (non-square shape)
 * @param[in]       Im_in         pointer to input tensor
 * @param[in]       dim_im_in_x   input tensor dimension x
 * @param[in]       dim_im_in_y   input tensor dimension y
 * @param[in]       ch_im_in      number of input tensor channels
 * @param[in]       wt            pointer to kernel weights
 * @param[in]       ch_im_out     number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x  filter kernel size x
 * @param[in]       dim_kernel_y  filter kernel size y
 * @param[in]       padding_x     padding sizes x
 * @param[in]       padding_y     padding sizes y
 * @param[in]       stride_x      convolution stride x
 * @param[in]       stride_y      convolution stride y
 * @param[in]       bias          pointer to bias
 * @param[in]       bias_shift    amount of left-shift for bias
 * @param[in]       out_shift     amount of right-shift for output
 * @param[in,out]   Im_out        pointer to output tensor
 * @param[in]       dim_im_out_x  output tensor dimension x
 * @param[in]       dim_im_out_y  output tensor dimension y
 * @param[in,out]   bufferA       pointer to buffer space for input
 * @param[in,out]   bufferB       pointer to buffer space for output
 * @return     The function returns either
 * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * This function is the version with full list of optimization tricks, but with
 * some constraints:
 *   ch_im_in is equal to ch_im_out
 *
 */

arm_status arm_depthwise_conv1D_HWC_q7(const q7_t *Im_in,
   const uint16_t dim_im_in_x,
   const uint16_t ch_im_in,
   const q7_t *wt,
   const uint16_t ch_im_out,
   const uint16_t dim_kernel_x,
   const uint16_t padding_x,
   const uint16_t stride_x,
   const q7_t *bias,
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

    /*
     * Implementation:
     * There are 3 nested loop here:
     * Inner loop: calculate each output value with MAC instruction over an accumulator
     * Mid   loop: loop over different output channel
     * Outer loop: loop over different output (x, y)
     *
     */

    int16_t i_out_x;
    int16_t i_ker_x;
    q7_t *colBuffer = (q7_t *)bufferA;
    q7_t *pBuffer = colBuffer;
    const q7_t *pBias = bias;
    q7_t *pOut = Im_out;
    uint16_t rowCnt;
    uint16_t row_shift;

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
    {
        /* we first do im2col here */
        for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x;
             i_ker_x++)
        {
            if (i_ker_x < 0 || i_ker_x >= dim_im_in_x)
            {
                /* arm_fill_q7(0, pBuffer, ch_im_in); */
                memset(pBuffer, 0, ch_im_in);
            }
            else
            {
                /* arm_copy_q7((q7_t *) Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in, pBuffer,
                 * ch_im_in); */
                memcpy(pBuffer, (q7_t *)Im_in + i_ker_x * ch_im_in, ch_im_in);
            }
            pBuffer += ch_im_in;
        }

        /* we will do the computation here for each channel */
        rowCnt = ch_im_out >> 2;
        row_shift = 0;
        pBias = bias;

        while (rowCnt)
        {
            q31_t sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
            q31_t sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
            q31_t sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
            q31_t sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

            uint16_t colCnt = dim_kernel_x >> 1;
            q7_t *pB = colBuffer + row_shift;
            const q7_t *pA = wt + row_shift;
            row_shift += 4;

#ifdef USE_INTRINSIC

#ifndef ARM_MATH_BIG_ENDIAN

            while (colCnt)
            {
                q31_t inA1, inA2, inB1, inB2, opA, opB;

                inB1 = arm_nn_read_q7x4(pB);
                pB += ch_im_in;
                opB = arm_nn_read_q7x4(pB);
                pB += ch_im_in;
                inB2 = __PKHTB(opB, inB1, 16);
                inB1 = __PKHBT(inB1, opB, 16);
                inA1 = arm_nn_read_q7x4(pA);
                pA += ch_im_in;
                opB = arm_nn_read_q7x4(pA);
                pA += ch_im_in;
                inA2 = __PKHTB(opB, inA1, 16);
                inA1 = __PKHBT(inA1, opB, 16);
                opA = __SXTB16(inA1);
                opB = __SXTB16(inB1);
                sum = __SMLAD(opA, opB, sum);
                opA = __SXTB16(__ROR(inA1, 8));
                opB = __SXTB16(__ROR(inB1, 8));
                sum2 = __SMLAD(opA, opB, sum2);
                opA = __SXTB16(inA2);
                opB = __SXTB16(inB2);
                sum3 = __SMLAD(opA, opB, sum3);
                opA = __SXTB16(__ROR(inA2, 8));
                opB = __SXTB16(__ROR(inB2, 8));
                sum4 = __SMLAD(opA, opB, sum4);
                colCnt--;
            }
#else

            while (colCnt)
            {
                q31_t inA1, inA2, inB1, inB2, opA, opB;

                inB1 = arm_nn_read_q7x4(pB);
                pB += ch_im_in;
                opB = arm_nn_read_q7x4(pB);
                pB += ch_im_in;
                inB2 = __PKHBT(opB, inB1, 16);
                inB1 = __PKHTB(inB1, opB, 16);
                inA1 = arm_nn_read_q7x4(pA);
                pA += ch_im_in;
                opB = arm_nn_read_q7x4(pA);
                pA += ch_im_in;
                inA2 = __PKHBT(opB, inA1, 16);
                inA1 = __PKHTB(inA1, opB, 16);
                opA = __SXTB16(inA1);
                opB = __SXTB16(inB1);
                sum2 = __SMLAD(opA, opB, sum2);
                opA = __SXTB16(__ROR(inA1, 8));
                opB = __SXTB16(__ROR(inB1, 8));
                sum = __SMLAD(opA, opB, sum);
                opA = __SXTB16(inA2);
                opB = __SXTB16(inB2);
                sum4 = __SMLAD(opA, opB, sum4);
                opA = __SXTB16(__ROR(inA2, 8));
                opB = __SXTB16(__ROR(inB2, 8));
                sum3 = __SMLAD(opA, opB, sum3);
                colCnt--;
            }

#endif /* ARM_MATH_BIG_ENDIAN */

#else

#ifndef ARM_MATH_BIG_ENDIAN
            //  r0    r1    r2    r3    r4   r5
            // inA1, inA2, inB1, inB2, opA, opB
            asm volatile("COL_LOOP:\n"
                         "ldr.w r2, [%[pB], #0]\n"
                         "add.w %[pB], %[pB], %[ch_im_in]\n"
                         "ldr.w r5, [%[pB], #0]\n"
                         "add.w %[pB], %[pB], %[ch_im_in]\n"
                         "pkhtb r3, r5, r2, ASR #16\n"
                         "pkhbt r2, r2, r5, LSL #16\n"
                         "ldr.w r0, [%[pA], #0]\n"
                         "add.w %[pA], %[pA], %[ch_im_in]\n"
                         "ldr.w r5, [%[pA], #0]\n"
                         "add.w %[pA], %[pA], %[ch_im_in]\n"
                         "pkhtb r1, r5, r0, ASR #16\n"
                         "pkhbt r0, r0, r5, LSL #16\n"
                         "sxtb16 r4, r0\n"
                         "sxtb16 r5, r2\n"
                         "smlad %[sum], r4, r5, %[sum]\n"
                         "mov.w r4, r0, ror #8\n"
                         "mov.w r5, r2, ror #8\n"
                         "sxtb16 r4, r4\n"
                         "sxtb16 r5, r5\n"
                         "smlad %[sum2], r4, r5, %[sum2]\n"
                         "sxtb16 r4, r1\n"
                         "sxtb16 r5, r3\n"
                         "smlad %[sum3], r4, r5, %[sum3]\n"
                         "mov.w r4, r1, ror #8\n"
                         "mov.w r5, r3, ror #8\n"
                         "sxtb16 r4, r4\n"
                         "sxtb16 r5, r5\n"
                         "smlad %[sum4], r4, r5, %[sum4]\n"
                         "subs %[colCnt], #1\n"
                         "bne COL_LOOP\n"
                         : [ sum ] "+r"(sum),
                           [ sum2 ] "+r"(sum2),
                           [ sum3 ] "+r"(sum3),
                           [ sum4 ] "+r"(sum4),
                           [ pB ] "+r"(pB),
                           [ pA ] "+r"(pA)
                         : [ colCnt ] "r"(colCnt), [ ch_im_in ] "r"(ch_im_in)
                         : "r0", "r1", "r2", "r3", "r4", "r5");
#else
            //  r0    r1    r2    r3    r4   r5
            // inA1, inA2, inB1, inB2, opA, opB
            asm volatile("COL_LOOP:\n"
                         "ldr.w r2, [%[pB], #0]\n"
                         "add.w %[pB], %[pB], %[ch_im_in]\n"
                         "ldr.w r5, [%[pB], #0]\n"
                         "add.w %[pB], %[pB], %[ch_im_in]\n"
                         "pkhbt r3, r5, r2, LSL #16\n"
                         "pkhtb r2, r2, r5, ASR #16\n"
                         "ldr.w r0, [%[pA], #0]\n"
                         "add.w %[pA], %[pA], %[ch_im_in]\n"
                         "ldr.w r5, [%[pA], #0]\n"
                         "add.w %[pA], %[pA], %[ch_im_in]\n"
                         "pkhbt r1, r5, r0, LSL #16\n"
                         "pkhtb r0, r0, r5, ASR #16\n"
                         "sxtb16 r4, r0\n"
                         "sxtb16 r5, r2\n"
                         "smlad %[sum2], r4, r5, %[sum2]\n"
                         "mov.w r4, r0, ror #8\n"
                         "mov.w r5, r2, ror #8\n"
                         "sxtb16 r4, r4\n"
                         "sxtb16 r5, r5\n"
                         "smlad %[sum], r4, r5, %[sum]\n"
                         "sxtb16 r4, r1\n"
                         "sxtb16 r5, r3\n"
                         "smlad %[sum4], r4, r5, %[sum4]\n"
                         "mov.w r4, r1, ror #8\n"
                         "mov.w r5, r3, ror #8\n"
                         "sxtb16 r4, r4\n"
                         "sxtb16 r5, r5\n"
                         "smlad %[sum3], r4, r5, %[sum3]\n"
                         "subs %[colCnt], #1\n"
                         "bne COL_LOOP\n"
                         : [ sum ] "+r"(sum),
                           [ sum2 ] "+r"(sum2),
                           [ sum3 ] "+r"(sum3),
                           [ sum4 ] "+r"(sum4),
                           [ pB ] "+r"(pB),
                           [ pA ] "+r"(pA)
                         : [ colCnt ] "r"(colCnt), [ ch_im_in ] "r"(ch_im_in)
                         : "r0", "r1", "r2", "r3", "r4", "r5");
#endif /*ARM_MATH_BIG_ENDIAN */

#endif /* USE_INTRINSIC */

            colCnt = dim_kernel_x & 0x1;
            while (colCnt)
            {
                union arm_nnword inA, inB;
                inA.word = arm_nn_read_q7x4(pA);
                pA += ch_im_in;
                inB.word = arm_nn_read_q7x4(pB);
                pB += ch_im_in;
                sum += inA.bytes[0] * inB.bytes[0];
                sum2 += inA.bytes[1] * inB.bytes[1];
                sum3 += inA.bytes[2] * inB.bytes[2];
                sum4 += inA.bytes[3] * inB.bytes[3];
                colCnt--;
            }

            *pOut++ = (q7_t)__SSAT((sum >> out_shift), 8);
            *pOut++ = (q7_t)__SSAT((sum2 >> out_shift), 8);
            *pOut++ = (q7_t)__SSAT((sum3 >> out_shift), 8);
            *pOut++ = (q7_t)__SSAT((sum4 >> out_shift), 8);

            rowCnt--;
        }

        rowCnt = ch_im_out & 0x3;
        while (rowCnt)
        {
            q7_t *pB = colBuffer + row_shift;
            const q7_t *pA = wt + row_shift;
            q31_t sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
            uint16_t colCnt = dim_kernel_x;

            row_shift += 1;

            while (colCnt)
            {
                q7_t A1 = *pA;
                q7_t B1 = *pB;
                pA += ch_im_in;
                pB += ch_im_in;
                sum += A1 * B1;

                colCnt--;
            }
            *pOut++ = (q7_t)__SSAT((sum >> out_shift), 8);
            rowCnt--;
        }

        // clear counter and pointers
        pBuffer = colBuffer;
    }
    
    /* Return to application */
    return ARM_MATH_SUCCESS;

#else
    return -1;

#endif /* ARM_MATH_DSP */
}
#endif