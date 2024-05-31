#ifndef __NNOM_GROUPCONV1D_H__
#define __NNOM_GROUPCONV1D_H__

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
typedef struct _nnom_groupconv1d_layer_t
{
	nnom_layer_t super;
	nnom_3d_shape_t kernel;
	nnom_3d_shape_t stride;
	nnom_3d_shape_t pad;
	nnom_3d_shape_t dilation;
	nnom_padding_t padding_type;
	uint32_t filters;
  uint32_t groups;

	nnom_tensor_t *weight; 
	nnom_tensor_t *bias;

	// test
	nnom_qformat_param_t * output_rshift;			
	nnom_qformat_param_t * bias_lshift;
} nnom_groupconv1d_layer_t;

// method
nnom_status_t groupconv1d_run(nnom_layer_t *layer);
nnom_status_t groupconv1d_build(nnom_layer_t *layer);
nnom_status_t groupconv1d_free(nnom_layer_t *layer);

// utils
uint32_t groupconv1d_output_length(uint32_t input_length, uint32_t filter_size, nnom_padding_t padding, uint32_t stride, uint32_t dilation);

// API
nnom_layer_t *GroupConv1D(uint32_t filters, uint32_t groups, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_3d_shape_t d,  nnom_padding_t pad_type, const nnom_weight_t *w, const nnom_bias_t *b);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_GROUPCONV1D_H__ */