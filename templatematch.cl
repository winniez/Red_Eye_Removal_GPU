__kernel void template(__global float *d_output,
    __global float *d_inputImage,
    __global float * d_templateImage,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
    )
{

  int  ny             = num_pixels_y;
  int  nx             = num_pixels_x;
  int  knx            = template_width;
  int  glosizex	      = get_local_size(0);
  int  glosizey	      = get_local_size(1);
  int tx 	      = get_local_id(0);
  int ty 	      = get_local_id(1);
  int blockx 	      = get_group_id(0);
  int blocky 	      = get_group_id(1);

int2 image_index_2d = (int2)( ( blockx* glosizex ) + tx, ( blocky * glosizey ) + ty );
  int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

  if ( image_index_2d.x < nx && image_index_2d.y < ny )
  {
    //
    // compute image mean
    //
    float image_sum = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = (int2)( image_index_2d.x + x, image_index_2d.y + y );
 
 	 int2 image_offset_index_2d_clamped = (int2)( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
  	int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
 
	  unsigned char image_offset_value = d_inputImage[ image_offset_index_1d_clamped ];

        image_sum += (float)image_offset_value;
      }
    }
  float image_mean = image_sum / (float)template_size;

    //
    // compute sums
    //
    float sum_of_image_template_diff_products = 0.0f;
    float sum_of_squared_image_diffs          = 0.0f;
    float sum_of_squared_template_diffs       = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = (int2)( image_index_2d.x + x, image_index_2d.y + y );
        int2 image_offset_index_2d_clamped = (int2)( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
        float image_offset_value = d_inputImage[ image_offset_index_1d_clamped ];
        float image_diff = image_offset_value - image_mean;

        int2 template_index_2d = (int2)( x + template_half_width, y + template_half_height );
        int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

        float template_value = d_templateImage[ template_index_1d ];
        float template_diff  = template_value - template_mean;

        float image_template_diff_product = image_offset_value   * template_diff;
        float squared_image_diff          = image_diff           * image_diff;
        float squared_template_diff       = template_diff        * template_diff;

        sum_of_image_template_diff_products += image_template_diff_product;
        sum_of_squared_image_diffs          += squared_image_diff;
        sum_of_squared_template_diffs       += squared_template_diff;
      }
    }


    //
    // compute final result
    //
    float result_value = 0.0f;

    if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 )
    {
      result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
    }

    d_output[ image_index_1d ] = result_value;
  }
}


  
