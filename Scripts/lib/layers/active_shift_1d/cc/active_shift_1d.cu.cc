/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "active_shift_1d.h"
#include "cuda.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <algorithm>
#include <cstring>
#include <vector>

// bottom: entree, top: sortie

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef std::vector<int32> TShape;

template <typename DType>
__global__ void ASL1D_Forward_GPUKernel(const int count, const int num, const int channels,
		const int top_width,
		const int bottom_width,
		const DType* xshift,
    const int pad_w,
    const int stride_w,
		const DType* bottom_data, DType* top) {
   //count : number_of_thread = nombre element de la sortie dans notre cas
   //index : index de base du calcul
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_width;
		const int bottom_sp_dim = bottom_width;
		const int n = index/(channels * top_sp_dim); //n :numero de l'echantillon du batch
		const int idx = index%(channels * top_sp_dim);//idx : position dans l'ehantillon
		const int c = idx/top_sp_dim;//c : numero de channel de la position
		const int w = idx%top_sp_dim;//sp_idx et w: position dans le channel
		const DType* data_im_ptr = bottom_data + n*channels*bottom_sp_dim + c*bottom_sp_dim;//pointeur sur le channel d'entrée de l'échantillon

		const int w_offset = w * stride_w - pad_w;

		{
			DType val = 0;
			const DType x = xshift[c];

			int w_im;

			int x1 = floorf(x);
			int x2 = x1+1;

			w_im = w_offset + x1;
			DType q1 = (w_im >= 0 && w_im < bottom_width) ? data_im_ptr[w_im] : 0;

			w_im = w_offset + x2;
			DType q2 = (w_im >= 0 && w_im < bottom_width) ? data_im_ptr[w_im] : 0;

			DType dx = x-x1;

			val = q1*(1-dx) + q2*dx;

			top[index] = val;
		}
	}
}



template <typename DType>
__global__ void ASL1D_ShiftBackwardMerged_GPUKernel(const int count, const int num, const int channels,
		const int top_width,
		const int bottom_width,
		const DType* xshift,
		const int pad_w,
		const int stride_w,
		const DType* bottom_data, const DType* top_diff, DType* shift_temp_buff_x) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		//n,c,i
		const int top_sp_dim = top_width;
		const int bottom_sp_dim = bottom_width;
		const int n = index/(channels * top_sp_dim);
		const int idx = index%(channels * top_sp_dim);
		const int c = idx/top_sp_dim;
		const int w = idx%top_sp_dim;
		const DType* data_im_ptr = bottom_data + n*channels*bottom_sp_dim + c*bottom_sp_dim;

		const int w_offset = w * stride_w - pad_w;

		//output : 2*(C) x (H*W)
		DType* out_ptr_x = shift_temp_buff_x + c*top_sp_dim + w; //(c,h,w)

		{
			DType val_x = 0;

			const DType shiftX = xshift[c];


			//Calc
			const int ix1 = floorf(shiftX);
			const int ix2 = ix1+1;
			const DType dx = shiftX-ix1;

			const int w_im1 = w_offset + ix1;
			const int w_im2 = w_offset + ix2;
			const int w_im1a = w_im1 + ((dx==0)?-1:0);

			const DType q1 = (w_im1 >= 0 && w_im1 < bottom_width) ? data_im_ptr[w_im1] : 0;
			const DType q2 = (w_im2 >= 0 && w_im2 < bottom_width) ? data_im_ptr[w_im2] : 0;


			const DType q1a = (dx==0)?((w_im1a >= 0 && w_im1a < bottom_width) ? data_im_ptr[w_im1a] : 0):q1;

			val_x = q2-q1a;


			//Summary
			const DType diff_scale = top_diff[index];
			const DType shiftX_diff_sum = val_x * diff_scale;

			//reduce along batch dimension
			CudaAtomicAdd(out_ptr_x, shiftX_diff_sum);
		}
	}
}


template <typename DType>
__global__ void ASL1D_BottomBackward_Stride1_GPUKernel(const int count, const int channels,
		const int top_width, //top
		const int bottom_width, //bottom
		const DType* xshift,
		const int pad_w,
		const DType* top_diff, DType* bottom_diff) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_width;
		const int bottom_sp_dim = bottom_width;
		const int n = index/(channels * bottom_sp_dim);
		const int idx = index%(channels * bottom_sp_dim);
		const int c = idx/bottom_sp_dim;
		const int w_col = idx%bottom_sp_dim;
		const DType* top_diff_ptr = top_diff + n*channels*top_sp_dim + c*top_sp_dim;

		const int w_offset = w_col + pad_w;


		{
			DType val = 0;
			const DType x = -xshift[c];  //reverse shift

			int w_im;

			if(x==0)
			{
				w_im = w_offset;
				val = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;
			}
			else
			{
				int x1 = floorf(x);
				int x2 = x1+1;

				//q1
				DType q1 = 0;

				w_im = (w_offset + x1);
				q1 = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;

				//q2
				DType q2 = 0;

				w_im = (w_offset + x2);
				q2 = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;

				DType dx = x-x1;

				val = q1*(1-dx)+ q2*dx;
				//printf("%d:%d, col(%d,%d), im(%d,%d) / %f,(%f,%f,%f,%f),(%f,%f)\n", flip, index, w_col, h_col, w_offset,h_offset, val, q11,q21,q12,q22, x,y);
			}

			bottom_diff[index] = val;
		}
	}
}


template <typename DType>
__global__ void ASL1D_BottomBackward_GPUKernel(const int count, const int channels,
		const int top_width, //top
		const int bottom_width, //bottom
		const DType* xshift,
		const int pad_w,
		const int stride_w,
		const DType* top_diff, DType* bottom_diff) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_width;
		const int bottom_sp_dim = bottom_width;
		const int n = index/(channels * bottom_sp_dim);
		const int idx = index%(channels * bottom_sp_dim);
		const int c = idx/bottom_sp_dim;
		const int w_col = idx%bottom_sp_dim;
		const DType* top_diff_ptr = top_diff + n*channels*top_sp_dim + c*top_sp_dim;

		const int w_offset = w_col + pad_w;


		{
			DType val = 0;
			const DType x = -xshift[c];  //reverse shift

			int w_im;

			if(x==0)
			{
				w_im = w_offset;
				if(w_im%stride_w == 0)
				{
					w_im=w_im/stride_w;

					val = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;
				}
			}
			else
			{
				int x1 = floorf(x);
				int x2 = x1+1;

				//q1
				DType q1 = 0;

				w_im = (w_offset + x1);
				if(w_im%stride_w == 0)
				{
					w_im=w_im/stride_w;

					q1 = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;
				}

				//q2
				DType q2 = 0;

				w_im = (w_offset + x2);
				if(w_im%stride_w == 0)
				{
					w_im=w_im/stride_w;

					q2 = (w_im >= 0 && w_im < top_width) ? top_diff_ptr[w_im] : 0;
				}

				DType dx = x-x1;

				val = q1*(1-dx) + q2*dx;
				//printf("%d:%d, col(%d,%d), im(%d,%d) / %f,(%f,%f,%f,%f),(%f,%f)\n", flip, index, w_col, h_col, w_offset,h_offset, val, q11,q21,q12,q22, x,y);
			}

			bottom_diff[index] = val;
		}
	}
}


template <typename DType>
__global__ void Apply1D_ShiftConstraint_GPUKernel(const int n, const DType* xshift_data, DType* xshift_diff, const bool normalize, const float clip_gradient) {
	CUDA_1D_KERNEL_LOOP(index, n)
  {
		const DType xi = xshift_data[index];
		const DType ri = xi;

		if(normalize) //normalize
		{
			const DType dx = xshift_diff[index];
			const DType dr = dx;

			if(dr!=0)
			{
				xshift_diff[index] = dx/dr;
			}
		}
		else if(clip_gradient!=0)
		{
			const DType dx = xshift_diff[index];
			const DType dr = dx;

			if(dr>clip_gradient)
			{
				xshift_diff[index] = dx/dr*clip_gradient;
			}
		}
	}
}


template <typename DType>
__global__ void setValueKernel(const int n, const DType val, DType* result_data)
{

  CUDA_1D_KERNEL_LOOP(index, n) {
      *(result_data+index) = val;
  }
  
}


namespace functor {

template <typename DType>
struct ASL1D_Forward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_width,
			const int bottom_width,
			const DType* xshift,
			const int pad_w,
			const int stride_w,
			const DType* bottom_data, DType* top_data)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		ASL1D_Forward_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
				count, num, channels,
				top_width,
				bottom_width,
				xshift,
				pad_w,
				stride_w,
				bottom_data, top_data);
	}
};


template <typename DType>
struct ASL1D_ShiftBackward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_width,
			const int bottom_width,
			const DType* xshift,
			const int pad_w,
			const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr, DType* temp_buf_ptr)
	{

		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		//const int temp_buf_offset = channels * top_height * top_width;	//(C*K) x (1*H*W)

		DType* shift_temp_buff_x = temp_buf_ptr;
    //DType* shift_temp_buff_y = temp_buf_ptr + temp_buf_offset;

		// shift diff
		ASL1D_ShiftBackwardMerged_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
				count, num, channels,
				top_width,
			  bottom_width,
				xshift,
				pad_w,
				stride_w,
				bottom_data, top_diff,
				shift_temp_buff_x);
	}
};




template <typename DType>
struct ASL1D_BottomBackward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_width,
			const int bottom_width,
			const DType* xshift,
			const int pad_w,
			const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		//bottom diff
		if(stride_w==1)
		{
			ASL1D_BottomBackward_Stride1_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					count, channels,
					top_width,
					bottom_width,
					xshift,
					pad_w,
					top_diff, bottom_backprop_ptr);
		}
		else
		{
			ASL1D_BottomBackward_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					count, channels,
					top_width,
					bottom_width,
					xshift,
					pad_w,
					stride_w,
					top_diff, bottom_backprop_ptr);
		}
	}
};



template <typename DType>
struct Apply1D_ShiftConstraint<GPUDevice, DType>{
	void operator()(const GPUDevice& d, const int count,
			const DType* xshift_data,
			DType* xshift_diff,
			const bool normalize, const float clip_gradient)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

	    Apply1D_ShiftConstraint_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
	    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	    		count, xshift_data, xshift_diff, normalize, clip_gradient);

	}
};



// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

template <typename DType>
struct setZero<GPUDevice, DType>{
    void operator() (const GPUDevice& d, const int n, DType* out){
    	CUDA_CHECK(cudaMemset(out, 0, sizeof(DType) * n));
    }
};


template <typename DType>
struct setValue<GPUDevice, DType>{
    void operator() (const GPUDevice& d, const int n, DType val, DType* out){
        CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
        setValueKernel<DType> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, val, out);
    }
};


}  // namespace functor

/*#define DECLARE_GPU_SPEC(DType)                                  \
    template struct functor::ASL1D_Forward<GPUDevice, DType>; \
    template struct functor::ASL1D_ShiftBackward<GPUDevice, DType>; \
    template struct functor::ASL1D_BottomBackward<GPUDevice, DType>; \
    template struct functor::Apply1D_ShiftConstraint<GPUDevice, DType>; \
    template struct functor::setZero<GPUDevice, DType>; \
    template struct functor::setValue<GPUDevice, DType>;
    
// extern template struct Copy<GPUDevice, T>;
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);

// TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC*/

// Explicitly instantiate functors for the types of OpKernels registered.

template struct functor::ASL1D_Forward<GPUDevice, float>;
template struct functor::ASL1D_ShiftBackward<GPUDevice, float>;
template struct functor::ASL1D_BottomBackward<GPUDevice, float>;
template struct functor::Apply1D_ShiftConstraint<GPUDevice, float>;
template struct functor::setZero<GPUDevice, float>;
template struct functor::setValue<GPUDevice, float>;

template struct functor::ASL1D_Forward<GPUDevice, double>;
template struct functor::ASL1D_ShiftBackward<GPUDevice, double>;
template struct functor::ASL1D_BottomBackward<GPUDevice, double>;
template struct functor::Apply1D_ShiftConstraint<GPUDevice, double>;
template struct functor::setZero<GPUDevice, double>;
template struct functor::setValue<GPUDevice, double>;


}  // namespace tensorflow

// #endif  // GOOGLE_CUDA