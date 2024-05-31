#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_depthwiseconv1d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_layer_t *DepthwiseConv1D(uint32_t multiplier, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_3d_shape_t d, nnom_padding_t pad_type,
						const nnom_weight_t *w, const nnom_bias_t *b)
{
    nnom_depthwiseconv1d_layer_t *layer;
  	nnom_buf_t *comp;
  	nnom_layer_io_t *in, *out;
  	// apply a block memory for all the sub handles.
  	size_t mem_size = sizeof(nnom_depthwiseconv1d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
  	layer = nnom_mem(mem_size);
  	if (layer == NULL)
  		return NULL;
  
  	// distribut the memory to sub handles.
  	in = (void *)((uint8_t*)layer + sizeof(nnom_depthwiseconv1d_layer_t));
  	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
  	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));
   
     
    
  
  	// set type in layer parent
  	layer->super.type = NNOM_DEPTHWISECONV_1D;
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
  	layer->super.run = depthwiseconv1d_run;
  	layer->super.build = depthwiseconv1d_build;
    layer->super.free = depthwiseconv1d_free;
  
  	// get the private parameters
  	layer->kernel = k;
  	layer->stride = s;
  	layer->dilation = d; 	
  	layer->filter_mult = multiplier; 		// for convs, this means filter number
  	layer->padding_type = pad_type;
  
  	// create weight and bias tensor
  	layer->weight = new_tensor(NNOM_QTYPE_PER_TENSOR, 2, multiplier);
  	layer->bias = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, multiplier);
  
    
  	// configure weight tensor manually to support new tensor based backends. 
  	// needs to be very careful
  	{
  		// config weight 
  		nnom_shape_data_t dim[2] = {k.w, multiplier};
  		*(layer->weight->q_offset) = 0;			// we have no support of offset here
  		*(layer->weight->q_dec) = 0;		// not using it
  		layer->weight->p_data = (void*)w->p_value;
  		layer->weight->bitwidth = 8;
  		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
  		nnom_memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(nnom_shape_data_t));
  
  		// config bias 
  		dim[0] = multiplier;
  		*(layer->bias->q_offset) = 0;			// we have no support of offset here
  		*(layer->bias->q_dec) = 0;		// not using it
  		layer->bias->p_data = (void*) b->p_value;
  		layer->bias->bitwidth = 8;
  		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
  		nnom_memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(nnom_shape_data_t));
  		
  		// output shift and bias shift
  		layer->output_rshift = (nnom_qformat_param_t *)&w->shift;
  		layer->bias_lshift = (nnom_qformat_param_t *)&b->shift;
  	}
    
  	return (nnom_layer_t *)layer;
}

uint32_t depthwiseconv1d_output_length(uint32_t input_length, uint32_t kernel_size, nnom_padding_t padding, uint32_t stride, uint32_t dilation)
{
    if (input_length == 0)
        return 0;
    uint32_t dilated_kernel_size = (kernel_size - 1) * dilation + 1;
	uint32_t output_length;
    if(padding == PADDING_SAME)
        output_length = input_length;
    else
        output_length = input_length - dilated_kernel_size + 1;
    return (output_length + stride - 1) / stride;
}

nnom_status_t depthwiseconv1d_build(nnom_layer_t *layer)
{
	nnom_depthwiseconv1d_layer_t *cl = (nnom_depthwiseconv1d_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor) * cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

	// calculate the output tensor q format, only support per tensor quantise now
	layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0];
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);

	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = 1;
	layer->out->tensor->dim[1] = depthwiseconv1d_output_length(layer->in->tensor->dim[1], cl->kernel.w, cl->padding_type, cl->stride.w, cl->dilation.w);
	layer->out->tensor->dim[2] = layer->in->tensor->dim[2] * cl->filter_mult; // channel stays the same

	// fill padding
	if (cl->padding_type == PADDING_SAME)
	{
		cl->pad.w = cl->dilation.w * (cl->kernel.w - 1) / 2;
		cl->pad.c = 0;
	}
	
	// bufferA size: 
	#ifdef NNOM_USING_CMSIS_NN
	layer->comp->size = 2 * 2 * (layer->in->tensor->dim[2] / cl->filter_mult) * cl->kernel.w;
	#endif

	// computational cost: K x K x Cin x Hout x Wout x Multiplier
	// or                : K x K x Cout x Hout x Wout
	layer->stat.macc = cl->kernel.w * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t depthwiseconv1d_free(nnom_layer_t *layer)
{
	// free weight and bias tensor when we are not initialised from structured configuration. 
	if(!layer->config)
	{
		nnom_depthwiseconv1d_layer_t* cl = (nnom_depthwiseconv1d_layer_t*)layer;
		delete_tensor(cl->weight);
		delete_tensor(cl->bias);
	}
	return NN_SUCCESS;
}

nnom_status_t depthwiseconv1d_run(nnom_layer_t *layer)
{
	nnom_status_t status = NN_SUCCESS;
	nnom_depthwiseconv1d_layer_t *cl = (nnom_depthwiseconv1d_layer_t *)layer;


	#ifdef NNOM_USING_CMSIS_NN
	// Current CMSIS-NN does not support dilation
	if(cl->dilation.w ==1 && cl->weight->qtype == NNOM_QTYPE_PER_TENSOR && cl->filter_mult == 1)
	{
  		// CMSIS-NN only support 1 mulplipier in depthwise conv
  		if (layer->in->tensor->dim[2] % 2 != 0 || layer->out->tensor->dim[2] % 2)
  			return NN_ARGUMENT_ERROR;
        
  		status = (nnom_status_t)arm_status arm_depthwise_conv1D_HWC_q7(
  				layer->in->tensor->p_data,
  				layer->in->tensor->dim[1], layer->in->tensor->dim[2],
  				cl->weight->p_data,
  				layer->out->tensor->dim[2],
  				cl->kernel.w,
  				cl->pad.w,
  				cl->stride.w,
  				cl->bias->p_data,
  				cl->bias_lshift[0], cl->output_rshift[0],
  				layer->out->tensor->p_data,
  				layer->out->tensor->dim[1], (q15_t *)(layer->comp->mem->blk), NULL);
          
  		if (status==0){
  				return status;
  		}
	}
	else
	#endif
	local_depthwise_conv1D_HWC_q7(
		layer->in->tensor->p_data,
		layer->in->tensor->dim[1], layer->in->tensor->dim[2],
		cl->weight->p_data,
		layer->out->tensor->dim[2],
		cl->kernel.w,
		cl->pad.w,
		cl->stride.w,
		cl->dilation.w,
		cl->bias->p_data,
		cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
		layer->out->tensor->p_data,
		layer->out->tensor->dim[1], NULL, NULL);
	return NN_SUCCESS;
}