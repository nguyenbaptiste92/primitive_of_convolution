#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_shiftconv2d.h"
#include "nnom_local_shiftconv2d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// a machine friendly api, with suffix _s for structured configuration.  
nnom_layer_t *shiftconv2d_s(const nnom_shiftconv2d_config_t *config)
{
	nnom_shiftconv2d_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	size_t mem_size;

	// allocate a block memory for all the sub handles and shifts.
	mem_size = sizeof(nnom_shiftconv2d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;
	
	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_shiftconv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_SHIFTCONV_2D;
	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	#ifdef NNOM_USING_CMSIS_NN
		layer->super.comp = comp;
	#else
		#ifdef USE_IM2COL
		layer->super.comp = comp;
		#endif
	#endif
	// set run method & output shape
	layer->super.run = shiftconv2d_run;
	layer->super.build = shiftconv2d_build;
	layer->super.free = shiftconv2d_free;

	// save the config
	layer->super.config = (void*) config;

	// get the private parameters
	// test: for 1d input, expend h = 1
	if(config->weight->num_dim == 3)
	{
		layer->kernel = kernel(1, 1);
	}
	else
	{
		layer->kernel = kernel(1, 1);
	}

	layer->filter_mult = config->filter_size; // for convs, this means filter number

	// get bias and weight tensor, this should be created by script. 
	layer->weight = config->weight;
	layer->bias = config->bias;
	layer->shift = config->shift;
	
	// get shifts
	layer->output_rshift = (nnom_qformat_param_t *)config->output_shift;
	layer->bias_lshift = (nnom_qformat_param_t *)config->bias_shift;


	return (nnom_layer_t *)layer;
}


// ShiftConv2D
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
nnom_layer_t *ShiftConv2D(uint32_t filters, const nnom_weight_t *w, const nnom_bias_t *b, const nnom_shift_t *shift)
{
	nnom_shiftconv2d_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_shiftconv2d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_shiftconv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_SHIFTCONV_2D;
	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	#ifdef NNOM_USING_CMSIS_NN
		layer->super.comp = comp;
	#else
		#ifdef USE_IM2COL
		layer->super.comp = comp;
		#endif
	#endif
	// set run method & output shape
	layer->super.run = shiftconv2d_run;
	layer->super.build = shiftconv2d_build;

	// get the private parameters
	layer->kernel = kernel(1, 1);
	layer->filter_mult = filters; 		// for convs, this means filter number

	// create weight and bias tensor
	layer->weight = new_tensor(NNOM_QTYPE_PER_TENSOR, 4, filters);
	layer->bias = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, filters);
  layer->shift = new_tensor(NNOM_QTYPE_PER_TENSOR, 2, filters);

	// configure weight tensor manually to support new tensor based backends. 
	// needs to be very careful
	{
		// config weight 
		nnom_shape_data_t dim[4] = {1, 1, 1, filters};
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
   
    // config shift 
		dim[0] = 2;
    		dim[1] = filters;
		*(layer->shift->q_offset) = 0;			// we have no support of offset here
		*(layer->shift->q_dec) = 0;		// not using it
		layer->shift->p_data = (void*) shift->p_value;
		layer->shift->bitwidth = 8;
		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->shift->dim, dim, layer->shift->num_dim * sizeof(nnom_shape_data_t));
		
		// output shift and bias shift
		layer->output_rshift = (nnom_qformat_param_t *)&w->shift;
		layer->bias_lshift = (nnom_qformat_param_t *)&b->shift;
	}

	return (nnom_layer_t *)layer;
}

nnom_status_t shiftconv2d_build(nnom_layer_t *layer)
{
	nnom_shiftconv2d_layer_t *cl = (nnom_shiftconv2d_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for the output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// calculate the output tensor q format, only support per tensor quantise now
	layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0]; // need some modification for 16bit. 
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
	
	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = layer->in->tensor->dim[0];
	layer->out->tensor->dim[1] = layer->in->tensor->dim[1];
	layer->out->tensor->dim[2] = cl->filter_mult; // channel stays the same

	#ifdef NNOM_USING_CMSIS_NN
		layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] ;
	#else
		#ifdef USE_IM2COL
			layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] ;
		#endif
	#endif
	// computational cost: K x K x Cin x Hour x Wout x Cout
	layer->stat.macc = layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t shiftconv2d_free(nnom_layer_t *layer)
{
	// free weight and bias tensor when we are not initialised from structured configuration. 
	if(!layer->config)
	{
		nnom_shiftconv2d_layer_t* cl = (nnom_shiftconv2d_layer_t*)layer;
		delete_tensor(cl->weight);
		delete_tensor(cl->bias);
		delete_tensor(cl->shift);
	}
	return NN_SUCCESS;
}


nnom_status_t shiftconv2d_run(nnom_layer_t *layer)
{
	nnom_shiftconv2d_layer_t *cl = (nnom_shiftconv2d_layer_t *)layer;
	nnom_status_t status;
	#ifdef NNOM_USING_CMSIS_NN
	// 8 bit cmsis nn : does not support per channel quantization
	if(layer->in->tensor->bitwidth == 8 && cl->weight->qtype == NNOM_QTYPE_PER_TENSOR)
	{
		// check if can use optimized function : ch_im_in is multiple of 4 and ch_im_out is multiple of 2
		if ((layer->in->tensor->dim[2] % 4 == 0) && (layer->out->tensor->dim[2] % 2 == 0))
		{
			status = (nnom_status_t)arm_shift_convolve_HWC_q7_fast_nonsquare(
			                        layer->in->tensor->p_data,
			                        layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
			                        cl->weight->p_data, layer->out->tensor->dim[2],
			                        cl->bias->p_data, cl->shift->p_data, cl->bias_lshift[0], cl->output_rshift[0],
			                        layer->out->tensor->p_data,
			                        layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);

			if (status==0){
				return status;
			}
		}
	}
    #endif // End of NNOM_USING_CMSIS_NN
	#ifdef USE_IM2COL
	local_shiftconvolve_HWC_q7_nonsquare_im2col(
				                        layer->in->tensor->p_data,
				                        layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				                        cl->weight->p_data, layer->out->tensor->dim[2],
				                        cl->bias->p_data, cl->shift->p_data, cl->bias_lshift[0], cl->output_rshift[0],
				                        layer->out->tensor->p_data,
				                        layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q7_t *)(layer->comp->mem->blk), NULL);
	#else
		local_shift_convolve_HWC_q7_nonsquare(
		layer->in->tensor->p_data,
		layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
		cl->weight->p_data, layer->out->tensor->dim[2],
		cl->bias->p_data,cl->shift->p_data, cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
		layer->out->tensor->p_data,
		layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
	#endif
	return NN_SUCCESS;
}
