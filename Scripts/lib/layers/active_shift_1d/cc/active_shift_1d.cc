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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include <cfloat>
#include <vector>
#include <array>

#include "absl/strings/string_view.h"


#include <unsupported/Eigen/CXX11/Tensor>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/tensor_format.h"
//#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/bounds_check.h"

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

#include "format_1d.h"
#include "active_shift_1d.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
} //namespace

namespace functor {
Status compute_output_shape(const int64 input_size, const int64 stride, const int64 padding, int64* output_size) {
	if (stride <= 0) {
		return errors::InvalidArgument("Stride must be > 0, but got ", stride);
	}

	*output_size = (input_size + 2 * padding - 1) / stride + 1;


	if (*output_size < 0) {
		return errors::InvalidArgument(
				"Computed output size would be negative: ", *output_size,
				" [input_size: ", input_size,
				", stride: ", stride,
				", padding: ", padding, "]");
	}

	return OkStatus();
}


template <typename T>
struct LaunchBlasGemv {
	static void Launch(OpKernelContext* ctx,
			const bool isTrans, const int m, const int n, const T alpha,
			const T* in_a_ptr, const int lda,
			const T* in_x_ptr, const int incx,
			const T beta, T* out_y_ptr, const int incy)
	{
		auto blas_trans = isTrans? perftools::gputools::blas::Transpose::kTranspose
								 : perftools::gputools::blas::Transpose::kNoTranspose;

		auto* stream = ctx->op_device_context()->stream();
		OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

		auto a_ptr = AsDeviceMemory(in_a_ptr);
		auto x_ptr = AsDeviceMemory(in_x_ptr);
		auto y_ptr = AsDeviceMemory(out_y_ptr);

		bool blas_launch_status =
				stream
				->ThenBlasGemv(blas_trans, m, n, alpha, a_ptr, lda, x_ptr, incx,
						beta, &y_ptr, incy)
						.ok();

		if (!blas_launch_status) {
			ctx->SetStatus(
					errors::Internal("Blas GEMV launch failed:  m=", m, ", n=", n));
		}
	}
};

} // namespace functor

REGISTER_OP("Active_Shift_1D")
.Input("x: T")		// [ batch, in_channels, in_rows]
.Input("shift: T")	// [in_channels]
.Output("output: T")
.Attr("T: {float, double}")
.Attr("strides: list(int)")
.Attr("paddings: list(int)")
.Attr("normalize: bool = false")
.Attr("data_format: { 'NWC', 'NCW' } = 'NCW' ")
.SetShapeFn([](InferenceContext* c) {
    ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
    ShapeHandle shift_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shift_shape));

    string data_format;
	TensorFormat1D data_format_;
	TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
	Format1DFromString(data_format, &data_format_);
	CHECK(data_format_ == FORMAT_NCW) << "ACU implementation only supports NCW tensor format for now.";


    std::vector<int32> strides;
    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
    if (strides.size() != 3) {
        return errors::InvalidArgument(
                   "ActiveShift1D requires the stride attribute to contain 3 values, but "
                   "got: ", strides.size());
    }
    const int32 stride_w = GetTensor1DDim(strides, data_format_, 'W');


    std::vector<int32> paddings;
	TF_RETURN_IF_ERROR(c->GetAttr("paddings", &paddings));
	if (paddings.size() != 3) {
		return errors::InvalidArgument(
				   "ActiveShift1D requires the padding attribute to contain 3 values, but "
				   "got: ", paddings.size());
	}
	const int32 pad_w = GetTensor1DDim(paddings, data_format_, 'W');

    DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
    DimensionHandle in_channels_dim = c->Dim(input_shape, 1);
    DimensionHandle in_rows_dim = c->Dim(input_shape, 2);

    DimensionHandle output_depth_dim = in_channels_dim;


    auto in_rows = c->Value(in_rows_dim);

    int64 output_rows;
    functor::compute_output_shape(in_rows, stride_w, pad_w, &output_rows);

    ShapeHandle output_shape = c->MakeShape(
    {batch_size_dim, output_depth_dim, output_rows});
    c->set_output(0, output_shape);
    return OkStatus();
})
.Doc(R"doc(
only support NCW now
)doc");


REGISTER_OP("Active_Shift_1D_Backprop")
.Input("x: T")		// [ batch, in_channels, in_rows, in_cols]
.Input("shift: T")	// [ 2, in_channels]
.Input("out_grad: T")
.Output("x_grad: T")
.Output("shift_grad: T")
.Attr("T: {float, double}")
.Attr("strides: list(int)")
.Attr("paddings: list(int)")
.Attr("normalize: bool = false")
.Attr("data_format: { 'NWC', 'NCW' } = 'NCW' ")
.SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return OkStatus();    
})
.Doc(R"doc(
only support NCW now
)doc");

typedef std::vector<int32> TShape;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ActiveShift1DOp : public OpKernel {


private:
	TensorFormat1D data_format_;

	//parameters
	int32 stride_w_;
	int32 pad_w_;

	//dimensions
	int32 nums_;
	int32 in_channels_;
	int32 bottom_width_;

	int32 out_channels_;
	int32 top_width_;

	//output tensor
	Tensor* output_;

private:
	void LayerSetUp(OpKernelContext* context, const Tensor& bottom_data, const Tensor& shift_data)
	{
		// Dimension Check for inputs
		OP_REQUIRES(context, bottom_data.dims() == 3,
				errors::InvalidArgument("input must be 3-dimensional",
						bottom_data.shape().DebugString()));
		OP_REQUIRES(context, shift_data.dims() == 1,
				errors::InvalidArgument("shift must be 1-dimensional: ",
						shift_data.shape().DebugString()));

		// Batch size
		const int64 nums_raw = GetTensor1DDim(bottom_data, data_format_, 'N');
		OP_REQUIRES(context, FastBoundsCheck(nums_raw,
				std::numeric_limits<int>::max()),
				errors::InvalidArgument("batch is too large"));
		nums_ = static_cast<int>(nums_raw);

		// Number of input channels
		const int64 in_channels_raw = GetTensor1DDim(bottom_data, data_format_, 'C');
		OP_REQUIRES(context, FastBoundsCheck(in_channels_raw,
				std::numeric_limits<int>::max()),
				errors::InvalidArgument("Input channels too large"));
		in_channels_ = static_cast<int>(in_channels_raw);

		// Sptial dimension for input

		const int64 bottom_width_raw = GetTensor1DDim(bottom_data, data_format_, 'W');
		OP_REQUIRES(context, FastBoundsCheck(bottom_width_raw,
				std::numeric_limits<int>::max()),
				errors::InvalidArgument("Input cols too large"));
		bottom_width_ = static_cast<int>(bottom_width_raw);

		// Number of output channels
		out_channels_ = in_channels_; //same as input channel

		// Sptial dimension for output
	    int64 top_width_raw;
	    functor::compute_output_shape(bottom_width_, stride_w_, pad_w_, &top_width_raw);
	    top_width_ = static_cast<int>(top_width_raw);

		// Setup Output
		const TensorShape out_shape = ShapeFromFormat1D(data_format_, nums_, top_width_, out_channels_);
		if (out_shape.num_elements() == 0)
			// If there is nothing to compute, return.
			return;

		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_));
	}

public:
	explicit ActiveShift1DOp(OpKernelConstruction* context) : OpKernel(context) {
		//data format
		string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, Format1DFromString(data_format, &data_format_),
				errors::InvalidArgument("Invalid data format"));

		//strides

		std::vector<int32> strides;
		OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
		OP_REQUIRES(context, strides.size() == 3,
				errors::InvalidArgument("Sliding window strides field must specify 3 dimensions"));
		const int32 stride_n = GetTensor1DDim(strides, data_format_, 'N');
		const int32 stride_c = GetTensor1DDim(strides, data_format_, 'C');
		OP_REQUIRES(context, stride_n == 1 && stride_c == 1,
				errors::InvalidArgument("Current implementation does not yet support "
						"strides in the batch and depth dimensions."));

		stride_w_ = GetTensor1DDim(strides, data_format_, 'W');

		//paddings
		std::vector<int32> paddings;
		OP_REQUIRES_OK(context, context->GetAttr("paddings", &paddings));
		OP_REQUIRES(context, paddings.size() == 3,
				errors::InvalidArgument("Paddings field must specify 3 dimensions"));
		const int32 padding_n = GetTensor1DDim(paddings, data_format_, 'N');
		const int32 padding_c = GetTensor1DDim(paddings, data_format_, 'C');
		OP_REQUIRES(context, padding_n == 0 && padding_c == 0,
				errors::InvalidArgument("Current implementation does not yet support "
						"paddings in the batch and depth dimensions."));

		pad_w_ = GetTensor1DDim(paddings, data_format_, 'W');

	}

	void Compute(OpKernelContext* context) override {
		const Tensor& bottom_data 	= context->input(0);	// [ batch, in_channels, in_rows]
		const Tensor& shift_data 	= context->input(1);	// [ in_channels]

		//Setup Layers
		LayerSetUp(context, bottom_data, shift_data);

		//Setup pointers
		const Device& d = context->eigen_device<Device>();
		const T* bottom_data_ptr 	= bottom_data.template flat<T>().data();
		const T* shift_data_ptr 	= shift_data.template flat<T>().data();

		//Forward pass
		T* top_data = output_->template flat<T>().data();
		const int count = output_->NumElements();

		functor::ASL1D_Forward<Device, T>()(d,
				count, nums_, in_channels_,
				top_width_,
				bottom_width_,
				shift_data_ptr,
				pad_w_,
				stride_w_,
				bottom_data_ptr, top_data);

		VLOG(2) << "Conv1D: in_channels = " << in_channels_
				<< ", bottom_width = " << bottom_width_
				<< ", stride_w = " << stride_w_
				<< ", out_channels = " << out_channels_;
	}
};



template <typename Device, typename T>
class ActiveShift1DBackpropOp : public OpKernel {

private:
	TensorFormat1D data_format_;

	// parameters
	int32 stride_w_;
	int32 pad_w_;
	bool normalize_;	//normalize shift gradient

	// dimensions
	int32 nums_;
	int32 in_channels_;
	int32 bottom_width_;

	int32 out_channels_;
	int32 top_width_;

	// output tensor
    Tensor* bottom_diff_;
    Tensor* shift_diff_;

	//temporary buffer
	Tensor temp_buf_;
	Tensor diff_muliplier_;

private:
	void LayerSetUp(OpKernelContext* context, const Tensor& bottom_data, const Tensor& shift_data, const Tensor& top_diff)
	{
		// batch size
		nums_ = static_cast<int32>(GetTensor1DDim(bottom_data, data_format_, 'N'));

		// input dimension
		in_channels_ = static_cast<int32>(GetTensor1DDim(bottom_data, data_format_, 'C'));
		bottom_width_ = static_cast<int32>(GetTensor1DDim(bottom_data, data_format_, 'W'));

		// output dimensions
		out_channels_ = in_channels_;
		top_width_ = static_cast<int32>(GetTensor1DDim(top_diff, data_format_, 'W'));

		//configure outputs
	    OP_REQUIRES_OK(context, context->allocate_output(0, bottom_data.shape(), &bottom_diff_));
	    OP_REQUIRES_OK(context, context->allocate_output(1, shift_data.shape(), &shift_diff_));


		//temporary buffer for backprop : x //A tester [2*in_channel, top_width] ou [in_channel, top_width]
		auto temp_buf_shape = TensorShape({2*in_channels_, top_width_});
		OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, temp_buf_shape, &temp_buf_));

	    //temporary buffer for reduce
		auto diff_muliplier_shape = TensorShape({top_width_});
		OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, diff_muliplier_shape, &diff_muliplier_));
	}

public:
	explicit ActiveShift1DBackpropOp(OpKernelConstruction* context)
	: OpKernel(context) {
		//data format
		string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, Format1DFromString(data_format, &data_format_),
					errors::InvalidArgument("Invalid data format"));

		//strides
		std::vector<int32> strides;
		OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
		stride_w_ = GetTensor1DDim(strides, data_format_, 'W');

		//paddings
		std::vector<int32> paddings;
		OP_REQUIRES_OK(context, context->GetAttr("paddings", &paddings));
		pad_w_ = GetTensor1DDim(paddings, data_format_, 'W');

		//normalize
		OP_REQUIRES_OK(context, context->GetAttr("normalize", &normalize_));
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& bottom_data = context->input(0);	// [ batch, in_channels, in_rows]
		const Tensor& shift_data = context->input(1);	// [in_channels ]
		const Tensor& top_diff = context->input(2);

		LayerSetUp(context, bottom_data, shift_data, top_diff);


	    //
		const Device& d = context->eigen_device<Device>();
		const T* bottom_data_ptr 	= bottom_data.template flat<T>().data();
		const T* shift_data_ptr 	= shift_data.template flat<T>().data();
		const T* top_diff_ptr 		= top_diff.template flat<T>().data();

		// outputs pointers
	    auto bottom_diff_ptr 	= bottom_diff_->template flat<T>().data();
	    auto shift_diff_ptr 	= shift_diff_->template flat<T>().data();

	    //temporary buffer for backprop : x, y
	    auto temp_buf_ptr = temp_buf_.template flat<T>().data();
	    functor::setZero<Device, T>()(d, temp_buf_.NumElements(), temp_buf_ptr);

	    //TODO : inefficient initialization for diff_muliplier
		auto diff_muliplier_ptr = diff_muliplier_.template flat<T>().data();
		functor::setValue<Device, T>()(d, diff_muliplier_.NumElements(), T(1.), diff_muliplier_ptr);	//set one

		//Start Calculation
		int count = top_diff.NumElements();
		const int temp_buf_offset = in_channels_ * top_width_;	//(C*1) x (1*W)


		// shift diff
		{
			//calc temporary diff for shift
			functor::ASL1D_ShiftBackward<Device, T>()(d,
					count, nums_, in_channels_,
					top_width_,
					bottom_width_,
					shift_data_ptr,
					pad_w_,
					stride_w_,
					bottom_data_ptr, top_diff_ptr,
					bottom_diff_ptr, shift_diff_ptr, temp_buf_ptr);

			//shift_diff : //2*(C*1) x (1*W)
			{
				const int M = in_channels_;
				const int N = top_width_;

				functor::LaunchBlasGemv<T>::Launch(context,
						        						true, N, 2*M, 1.,
						        						temp_buf_ptr, N,
														diff_muliplier_ptr, 1,
														0., shift_diff_ptr, 1);
			}
		}

		//Apply Constraint
		auto xShiftDiff = shift_diff_ptr;

		count = in_channels_;
		functor::Apply1D_ShiftConstraint<Device, T>()(d, count,
				shift_data_ptr,
				xShiftDiff, normalize_, 0.f);

		//bottom diff
		count = bottom_data.NumElements();
		functor::ASL1D_BottomBackward<Device, T>()(d,
							count, nums_, in_channels_,
							top_width_,
							bottom_width_,
							shift_data_ptr,
							pad_w_,
							stride_w_,
							bottom_data_ptr, top_diff_ptr,
							bottom_diff_ptr, shift_diff_ptr);
	}
};

#if GOOGLE_CUDA


#define REGISTER_GPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Active_Shift_1D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ActiveShift1DOp<GPUDevice, T>);                                    \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Active_Shift_1D_Backprop").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ActiveShift1DBackpropOp<GPUDevice, T>);
                                                                    
// TF_CALL_GPU_NUMBER_TYPES(REGISTER);
REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER


#endif  // GOOGLE_CUDA


}  // namespace tensorflow