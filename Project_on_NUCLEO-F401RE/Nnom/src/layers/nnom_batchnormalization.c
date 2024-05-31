#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local_batchnormalization.h"
#include "nnom_layers.h"
#include "layers/nnom_batchnormalization.h"

// a machine friendly api, with suffix _s for structured configuration.  
nnom_layer_t *batchnormalization_s(const nnom_batchnormalization_config_t *config)
{
	nnom_batchnormalization_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	size_t mem_size;

	// allocate a block memory for all the sub handles and shifts.
	mem_size = sizeof(nnom_batchnormalization_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;
	
	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_batchnormalization_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_BATCHNORMALIZATION;
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
	layer->super.run = batchnormalization_run;
	layer->super.build = batchnormalization_build;
	layer->super.free = batchnormalization_free;

	// save the config
	layer->super.config = (void*) config;

	// get bias and weight tensor and bn_factor, this should be created by script. 
	layer->weight = config->weight;
	layer->bias = config->bias;
	
	// get shifts
	layer->output_rshift = (nnom_qformat_param_t *)config->output_shift;
	layer->bias_lshift = (nnom_qformat_param_t *)config->bias_shift;

	return (nnom_layer_t *)layer;
}


// BatchNormalization
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
nnom_layer_t *BatchNormalization(size_t output_unit, const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_batchnormalization_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;
	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_batchnormalization_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_batchnormalization_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_BATCHNORMALIZATION;
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
	layer->super.run = batchnormalization_run;
	layer->super.build = batchnormalization_build;

	// create weight and bias tensor
	layer->weight = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, output_unit);
	layer->bias = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, output_unit);

	// configure weight tensor manually to support new tensor based backends. 
	// needs to be very careful
	{
		// config weight 
		nnom_shape_data_t dim[1] = {output_unit}; // the first dim doesnt matter here. will be file in later. 
		*(layer->weight->q_offset) = 0;			// we have no support of offset here
		*(layer->weight->q_dec) = 0;		// this is not even correct
		layer->weight->p_data = (void*)w->p_value;
		layer->weight->bitwidth = 8;
		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(nnom_shape_data_t));

		// config bias 
		dim[0] = output_unit;
		*(layer->bias->q_offset) = 0;			// we have no support of offset here
		*(layer->bias->q_dec) = 0;		// this is not even correct
		layer->bias->p_data = (void*)b->p_value;
		layer->bias->bitwidth = 8;
		layer->weight->qtype = NNOM_QTYPE_PER_TENSOR;
		nnom_memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(nnom_shape_data_t));
	}
		
	// output shift and bias shift
	layer->output_rshift = (nnom_qformat_param_t *)&w->shift;
	layer->bias_lshift = (nnom_qformat_param_t *)&b->shift;

	return (nnom_layer_t *)layer;
}


nnom_status_t batchnormalization_build(nnom_layer_t *layer)
{
	nnom_batchnormalization_layer_t *cl = (nnom_batchnormalization_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for the output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// calculate the output tensor q format, only support per tensor quantise now
  layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0]; // need some modification for 16bit.
      
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
	
	// now we set up the tensor shape, always HWC format (dimension stay the same
	layer->out->tensor->dim[0] = layer->in->tensor->dim[0];
	layer->out->tensor->dim[1] = layer->in->tensor->dim[1];
	layer->out->tensor->dim[2] = layer->in->tensor->dim[2];
	

	#ifdef NNOM_USING_CMSIS_NN
	// vec_buffer size: dim_vec (*2, q7->q15) ? I am not sure this is right
	layer->comp->size = tensor_size(layer->in->tensor)*2;
	#endif
	// computational cost: K x K x Cin x Hour x Wout x Cout (modify for add conv)
	layer->stat.macc = 2*tensor_size(layer->in->tensor);
	return NN_SUCCESS;
}

nnom_status_t batchnormalization_free(nnom_layer_t *layer)
{
	// free weight and bias tensor when we are not initialised from structured configuration. 
	if(!layer->config)
	{
		nnom_batchnormalization_layer_t* cl = (nnom_batchnormalization_layer_t*)layer;
		delete_tensor(cl->weight);
		delete_tensor(cl->bias);
	}

	return NN_SUCCESS;
}


nnom_status_t batchnormalization_run(nnom_layer_t *layer)
{
	nnom_batchnormalization_layer_t *cl = (nnom_batchnormalization_layer_t *)layer;


	// HWC format
  local_batchnormalization_HWC_q7_nonsquare(layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, 
				cl->bias->p_data, cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
				layer->out->tensor->p_data, NULL, NULL);

	return NN_SUCCESS;
}