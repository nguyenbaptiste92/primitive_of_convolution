#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local_bnaddconv2d.h"
#include "nnom_layers.h"
#include "layers/nnom_bnaddconv2d.h"

// a machine friendly api, with suffix _s for structured configuration.  
nnom_layer_t *bnaddconv2d_s(const nnom_bnaddconv2d_config_t *config)
{
	nnom_bnaddconv2d_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	size_t mem_size;

	// allocate a block memory for all the sub handles and shifts.
	mem_size = sizeof(nnom_bnaddconv2d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;
	
	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_bnaddconv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_BNADDCONV_2D;
	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	#ifdef NNOM_USING_CMSIS_NN
	layer->super.comp = comp;
	#endif
	// set run method & output shape
	layer->super.run = bnaddconv2d_run;
	layer->super.build = bnaddconv2d_build;
	layer->super.free = bnaddconv2d_free;

	// save the config
	layer->super.config = (void*) config;

	// get the private parameters
	// test: for 1d input, expend h = 1
	if(config->weight->num_dim == 3)
	{
		layer->kernel = kernel(1, config->kernel_size[0]);
		layer->stride = stride(1, config->stride_size[0]);
		layer->dilation = dilation(1, config->dilation_size[0]);
	}
	else
	{
		layer->kernel = kernel(config->kernel_size[0], config->kernel_size[1]);
		layer->stride = stride(config->stride_size[0], config->stride_size[1]);
		layer->dilation = dilation(config->dilation_size[0], config->dilation_size[1]);
	}

	layer->filter_mult = config->filter_size; // for convs, this means filter number
	layer->padding_type = config->padding_type;

	// get bias and weight tensor and bn_factor, this should be created by script. 
	layer->weight = config->weight;
	layer->bias = config->bias;
  layer->bn_factor = config->bn_factor;
	
	// get shifts
	layer->output_rshift = (nnom_qformat_param_t *)config->output_shift;
	layer->bias_lshift = (nnom_qformat_param_t *)config->bias_shift;
  layer->inter_lshift = (nnom_qformat_param_t *)config->inter_lshift;
  
  // get mode (if shift of weight is superior to  the shift of input)
  layer->mode = config->mode;

	// padding
	if (layer->padding_type == PADDING_SAME)
	{
		layer->pad.h = layer->dilation.h * (layer->kernel.h - 1) / 2;
		layer->pad.w = layer->dilation.w * (layer->kernel.w - 1) / 2;
		layer->pad.c = (1 - 1) / 2;
	}

	return (nnom_layer_t *)layer;
}


// Conv2D
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
nnom_layer_t *BnAddConv2D(uint32_t filters, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_3d_shape_t d,  nnom_padding_t pad_type,
					 const nnom_weight_t *w, const nnom_bias_t *b, const nnom_addconv_parameter_t *add_param)
{
	nnom_bnaddconv2d_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_bnaddconv2d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_bnaddconv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_BNADDCONV_2D;
	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	#ifdef NNOM_USING_CMSIS_NN
	layer->super.comp = comp;
	#endif
	// set run method & output shape
	layer->super.run = bnaddconv2d_run;
	layer->super.build = bnaddconv2d_build;

	// get the private parameters
	layer->kernel = k;
	layer->stride = s;
	layer->dilation = d; 	
	layer->filter_mult = filters; 		// for convs, this means filter number
	layer->padding_type = pad_type;
  layer->mode = add_param->mode;

	// create weight and bias tensor and bn_factor
	layer->weight = new_tensor(NNOM_QTYPE_PER_TENSOR, 4, filters);
	layer->bias = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, filters);
  layer->bn_factor = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, filters);

	// configure weight tensor manually to support new tensor based backends. 
	// needs to be very careful
	{
		// config weight 
		nnom_shape_data_t dim[4] = {k.h, k.w, k.c, filters};
		*(layer->weight->q_offset) = 0;			// we have no support of offset here
		*(layer->weight->q_dec) = 0;		// not using it
		layer->weight->p_data = (void*)w->p_value;
		layer->weight->bitwidth = 8;
		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(nnom_shape_data_t));

		// config bias 
		dim[0] = filters;
		*(layer->bias->q_offset) = 0;			// we have no support of offset here
		*(layer->bias->q_dec) = 0;		// not using it
		layer->bias->p_data = (void*) b->p_value;
		layer->bias->bitwidth = 8;
		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(nnom_shape_data_t));
   
    // config bn_factor
    dim[0] = filters;
		*(layer->bn_factor->q_offset) = 0;			// we have no support of offset here
		*(layer->bn_factor->q_dec) = 0;		// not using it
		layer->bn_factor->p_data = (void*) add_param->bn_factor;
		layer->bn_factor->bitwidth = 8;
		layer->bn_factor->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->bn_factor->dim, dim, layer->bn_factor->num_dim * sizeof(nnom_shape_data_t));
		
		// output shift and bias shift
		layer->output_rshift = (nnom_qformat_param_t *)&w->shift;
		layer->bias_lshift = (nnom_qformat_param_t *)&b->shift;
    layer->inter_lshift = (nnom_qformat_param_t *)&add_param->shift;
	}

	return (nnom_layer_t *)layer;
}

// keras's implementation. 
// source: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L85
uint32_t bnaddconv_output_length(uint32_t input_length, uint32_t filter_size, nnom_padding_t padding, uint32_t stride, uint32_t dilation)
{
    if (input_length == 0)
        return 0;
    uint32_t dilated_filter_size = (filter_size - 1) * dilation + 1;
	uint32_t output_length;
    if(padding == PADDING_SAME)
        output_length = input_length;
    else
        output_length = input_length - dilated_filter_size + 1;
    return (output_length + stride - 1) / stride;
}

nnom_status_t bnaddconv2d_build(nnom_layer_t *layer)
{
	nnom_bnaddconv2d_layer_t *cl = (nnom_bnaddconv2d_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for the output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// calculate the output tensor q format, only support per tensor quantise now
  if (cl->mode==2){
      layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->bn_factor->q_dec[0] - cl->output_rshift[0]; // need some modification for 16bit.
  }
  else{
      layer->out->tensor->q_dec[0] = cl->weight->q_dec[0] + cl->bn_factor->q_dec[0] - cl->output_rshift[0]; // need some modification for 16bit.
  }
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
	
	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = bnaddconv_output_length(layer->in->tensor->dim[0], cl->kernel.h, cl->padding_type, cl->stride.h, cl->dilation.h);
	layer->out->tensor->dim[1] = bnaddconv_output_length(layer->in->tensor->dim[1], cl->kernel.w, cl->padding_type, cl->stride.w, cl->dilation.w);
	layer->out->tensor->dim[2] = cl->filter_mult; // channel stays the same
	
	// fill padding
	if (cl->padding_type == PADDING_SAME)
	{
		cl->pad.w = cl->dilation.w * (cl->kernel.w - 1) / 2;
		cl->pad.h = cl->dilation.h * (cl->kernel.h - 1) / 2;
		cl->pad.c = 0;
	}

	#ifdef NNOM_USING_CMSIS_NN
	// bufferA size: (1D shape)
	// 2*ch_im_in*dim_kernel*dim_kernel
	layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] * cl->kernel.w * cl->kernel.h;
	#endif
	// computational cost: K x K x Cin x Hour x Wout x Cout (modify for add conv)
	layer->stat.macc = cl->kernel.w * cl->kernel.h * layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t bnaddconv2d_free(nnom_layer_t *layer)
{
	// free weight and bias tensor and bn_factor when we are not initialised from structured configuration. 
	if(!layer->config)
	{
		nnom_bnaddconv2d_layer_t* cl = (nnom_bnaddconv2d_layer_t*)layer;
		delete_tensor(cl->weight);
		delete_tensor(cl->bias);
    delete_tensor(cl->bn_factor);
	}
	return NN_SUCCESS;
}


nnom_status_t bnaddconv2d_run(nnom_layer_t *layer)
{
	nnom_bnaddconv2d_layer_t *cl = (nnom_bnaddconv2d_layer_t *)layer;


	// HWC format
	if (cl->mode==0){
      local_bnadd_convolve_HWC_q7_nonsquare_0(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
				cl->bias->p_data,cl->bn_factor->p_data, cl->bias_lshift, cl->output_rshift,cl->inter_lshift, cl->weight->qtype,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
  }
  else if (cl->mode==1){
      local_bnadd_convolve_HWC_q7_nonsquare_1(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
				cl->bias->p_data,cl->bn_factor->p_data, cl->bias_lshift, cl->output_rshift,cl->inter_lshift, cl->weight->qtype,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
  }
  else{
      local_bnadd_convolve_HWC_q7_nonsquare_2(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
				cl->bias->p_data,cl->bn_factor->p_data, cl->bias_lshift, cl->output_rshift,cl->inter_lshift, cl->weight->qtype,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
  }

	return NN_SUCCESS;
}