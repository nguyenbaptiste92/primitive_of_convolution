#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "nnom.h"
#include "weights.h"
#include "test_data.h"

#define IMAGE_SIZE (50*7)//Adjust to the input size
#define MAX_LINE_LENGTH (IMAGE_SIZE*5) 
#define OUTPUT_SIZE (5) //Adjust to the output size

#ifdef NNOM_USING_STATIC_MEMORY
uint8_t static_buf[1024 * 500]; //Sometime you need to modify it
#endif
 
int main(int argc, char* argv[])
{
	nnom_model_t* model;
	nnom_predict_t * pre;
	float prob;
	uint32_t label;
	size_t size = 0;
 
  #ifdef NNOM_USING_STATIC_MEMORY
	// when use static memory buffer, we need to set it before create
	nnom_set_static_buf(static_buf, sizeof(static_buf)); 
  #endif
  
  model = nnom_model_create();				// create NNoM model

  FILE *out_file = fopen("test_result.txt", "w");
  FILE *out_shift = fopen("test_shift.txt", "w");
  
  int size_liste_image = sizeof(liste_image)/sizeof(liste_image[0]);
  int size_image = sizeof(liste_image[0]);
  printf("number_of_images : %d\n",size_liste_image);
  printf("size_of_images : %d\n",size_image);
  int i;
  int j;
  
  for (i = 0; i < size_liste_image; i++){
      for (j = 0; j < size_image; j++){
          nnom_input_data[j] = liste_image[i][j];
      }
      model_run(model);
      for(int i = 0; i < OUTPUT_SIZE; i++){
          fprintf(out_file, "%d ", nnom_output_data[i]);
      }
      fprintf(out_file, "\n");
  }
  
  fprintf(out_shift, "%d\n", ACTIVATION_1_OUTPUT_SHIFT); //Adjust to the name of the output shift variable
  
  if (fclose(out_file)){
    return EXIT_FAILURE;
  }
	return 0;
}
