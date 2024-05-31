#ifndef __NNOM_SHIFTCONV1D_H__
#define __NNOM_SHIFTCONV1D_H__

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
typedef struct _nnom_shiftconv1d_layer_t
{
	nnom_layer_t super;
	nnom_3d_shape_t kernel;
  nnom_3d_shape_t stride;
	uint32_t filter_size; 							// filter size (for conv) or multilplier (for depthwise)

	nnom_tensor_t *weight; 
	nnom_tensor_t *bias;
  nnom_tensor_t *shift;

	// test
	nnom_qformat_param_t * output_rshift;			
	nnom_qformat_param_t * bias_lshift;
} nnom_shiftconv1d_layer_t;

// method
nnom_status_t shiftconv1d_run(nnom_layer_t *layer);
nnom_status_t shiftconv1d_build(nnom_layer_t *layer);
nnom_status_t shiftconv1d_free(nnom_layer_t *layer);

// utils
uint32_t shiftconv1d_output_length(uint32_t input_length, uint32_t stride);

// API
nnom_layer_t *ShiftConv1D(uint32_t filters, nnom_3d_shape_t s, const nnom_weight_t *w, const nnom_bias_t *b, const nnom_shift_t *shift);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_SHIFTCONV1D_H__ */