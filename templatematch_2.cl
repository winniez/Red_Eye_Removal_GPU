#define BLOCKSIZE 16

__kernel void combined_response(
    __global float *d_response,
    __global float *d_r_response,
    __global float *d_g_response,
    __global float *d_b_response,
    int            num_pixels_y,
    int            num_pixels_x)
{
    int ny          = num_pixels_y;
    int nx          = num_pixels_x;
    int tx          = get_local_id(0);
    int ty          = get_local_id(1);
    int bx          = get_group_id(0);
    int by          = get_group_id(1);

    __local float sub_r[BLOCKSIZE];
    __local float sub_g[BLOCKSIZE];
    __local float sub_b[BLOCKSIZE];
    

    int2 image_index_2d = (int2) (get_global_id(0), get_global_id(1));
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;
    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        d_response[ image_index_1d ] = d_r_response[ image_index_1d ] * d_g_response[ image_index_1d ] * d_b_response[ image_index_1d ];
    }
}


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
    
    __local float local_sub_img[48][48];
    __local float local_temp[33][33];
   
    
    // each thread loads 2*2 pixel values to local_temp, thread(15,15) load extra template_width+template_height pixel values 
    local_temp[tx*2][ty*2] = d_templateImage[ty*2*template_width+tx*2];
    local_temp[tx*2+1][ty*2] = d_templateImage[ty*2*template_width+tx*2+1];
    local_temp[tx*2][ty*2+1] = d_templateImage[(ty*2+1)*template_width+tx*2];
    local_temp[tx*2+1][ty*2+1] = d_templateImage[(ty*2+1)*template_width+tx*2+1];
    
    if (tx == 15 && ty == 15)
    {
        for (int a = 0; a < template_height; a++)
        {
            local_temp[tx*2+2][a] = d_templateImage[a*template_width+tx*2+2];
        }
        for (int b = 0; b < template_width; b++)
        {
            local_temp[b][ty*2+2] = d_templateImage[(ty*2+2)*template_width+b];
        }
    }
    
    // each thread loads 3*3 pixel values to local_sub_img
    int2 image_index_2d = (int2) (get_global_id(0), get_global_id(1));
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

  
    if ( image_index_2d.x < nx && image_index_2d.y < ny )
  {
    // each thread loads 3*3 pixel values to local_sub_img
    // local_sub_img loads [3*tx - 16 ~ 3*tx - 14][3*ty - 16 ~ 3*ty -14 ] from block perspective
    // get_global_id(0) = get_group_id(0) * BLOCKSIZE + get_local_id(0)
    // get_global_id(1) = get_group_id(1) * BLOCKSIZE + get_local_id(1)
    int2 load_img_index_2d_clamped;
    int load_img_index_1d_clamped;
    for (int a = 0; a < 3; a++)
    {
        for (int b = 0; b < 3; b++)
        {
            load_img_index_2d_clamped = (int2)(min( nx - 1, max( 0,image_index_2d.x + 2*tx-16+a)), min( ny - 1, max( 0,image_index_2d.y + 2*ty-16+b)));
            load_img_index_1d_clamped = (nx * load_img_index_2d_clamped.y) + load_img_index_2d_clamped.x;
            local_sub_img[tx*3+a][ty*3+b] = d_inputImage[load_img_index_1d_clamped];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //
    // compute image mean
    //
    float image_sum = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
        for ( int x = -template_half_width; x <= template_half_width; x++ )
        {
            //int2 image_offset_index_2d = (int2)( image_index_2d.x + x, image_index_2d.y + y );
            //int2 image_offset_index_2d_clamped = (int2)( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
            //int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
            float image_offset_value = local_sub_img[16+tx+x][16+ty+y]; 
            //float image_offset_value = d_inputImage[ image_offset_index_1d_clamped ];

            image_sum += image_offset_value;
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
        //int2 image_offset_index_2d         = (int2)( image_index_2d.x + x, image_index_2d.y + y );
        //int2 image_offset_index_2d_clamped = (int2)( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        //int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
        
        //float image_offset_value = d_inputImage[ image_offset_index_1d_clamped ];
        float image_offset_value = local_sub_img[16+tx+x][16+ty+y];
        float image_diff = image_offset_value - image_mean;

        //int2 template_index_2d = (int2)( x + template_half_width, y + template_half_height );
        //int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

        //float template_value = d_templateImage[ template_index_1d ];
        float template_value = local_temp[x + template_half_width][y + template_half_height];
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


  
