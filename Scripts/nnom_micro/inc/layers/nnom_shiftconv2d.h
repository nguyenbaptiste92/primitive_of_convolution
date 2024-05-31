#ifndef __NNOM_SHIFTCONV2D_H__
#define __NNOM_SHIFTCONV2D_H__

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_layers.h"
#include "nnom_local.h"
#include "nnom_tensor.h"


// child layers parameters
typedef struct _nnom_shiftconv2d_layer_t
{
	nnom_layer_t super;
	nnom_3d_shape_t kernel;
	uint32_t filter_mult; 							// filter size (for conv) or multilplier (for depthwise)

	nnom_tensor_t *weight; 
	nnom_tensor_t *bias;
  nnom_tensor_t *shift;

	// test
	nnom_qformat_param_t * output_rshift;			
	nnom_qformat_param_t * bias_lshift;
} nnom_shiftconv2d_layer_t;

// a machine interface for configuration
typedef struct _nnom_shiftconv2d_config_t
{
	nnom_layer_config_t super;
	nnom_qtype_t qtype; 	//quantisation type(per channel or per layer)
	nnom_tensor_t *weight;
	nnom_tensor_t *bias;
  nnom_tensor_t *shift;
	nnom_qformat_param_t *output_shift;   
	nnom_qformat_param_t *bias_shift;   
	uint32_t filter_size;  
	int8_t kernel_size[2];
} nnom_shiftconv2d_config_t;

// method
nnom_status_t shiftconv2d_run(nnom_layer_t *layer);
nnom_status_t shiftconv2d_build(nnom_layer_t *layer);
nnom_status_t shiftconv2d_free(nnom_layer_t *layer);

// API
nnom_layer_t *shiftconv2d_s(const nnom_shiftconv2d_config_t *config);
nnom_layer_t *ShiftConv2D(uint32_t filters, const nnom_weight_t *w, const nnom_bias_t *b, const nnom_shift_t *shift);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_SHIFTCONV2D_H__ */