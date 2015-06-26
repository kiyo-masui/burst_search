#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "dedisperse_gbt.h"
#include "dedisperse.h"


int main(int argc, char *argv[])
{
  float nu1=900;
  int nnu=4096;
  float dnu=(700-900.0)/nnu;
  float tsamp=1e-3;
  

  float max_dm=2000;
  //if (argc>1)
  // max_dm=atof(argv[1]);
  int depth=burst_depth_for_max_dm(max_dm,nnu,nu1,dnu,tsamp);
  max_dm=get_diagonal_dm_simple(nu1,nu1+dnu*nnu,tsamp,depth);
  printf("depth required is %d, good to DM %12.4f\n",depth,max_dm);
  int nchan_out=burst_get_num_dispersions(nnu,nu1,dnu,depth);
  printf("have %d output channels.\n",nchan_out);
  int *chan_map=ivector(nnu);
  setup_channel_mapping(nnu,nu1,dnu,depth,chan_map);
  printf("first channel maps to %d\n",chan_map[0]);
  
  Data *dat;
  if (argc>1)
    dat=read_gbt(argv[1]);
  else
    dat=read_gbt("GBT11B_wigglez1hr_01_0123_dump.dat");
  printf("first and last channels from disk are %12.4f %12.4f\n",dat->raw_chans[0],dat->raw_chans[dat->raw_nchan-1]);
  
  int chunk_size=3*nchan_out;
  //the bonus nchan_out padding is to make sure there's enough space to handle the overflow from the end of the second chunk pasted in.
  size_t noutbuf=nchan_out*(chunk_size+nchan_out);
  float *outbuf=(float *)malloc(sizeof(float)*noutbuf);
  for (int i=0;i<10;i++) {
    printf("ptr %2d is %ld %ld\n",i,dat->raw_data[i], dat->raw_data[i+chunk_size]);
  }
  return 0;
  for (int i=0;i<(dat->ndata-chunk_size);i+=chunk_size) {
    printf ("\n\n\nworking on chunk %d with size %d out of %d\n",i,chunk_size,dat->ndata);
    float *dat1=dat->raw_data[i];
    float *dat2=dat->raw_data[i]+chunk_size*dat->raw_nchan;
    printf("distance is %ld %ld\n",(size_t)(dat1),(size_t)dat2);
    

    int n1=chunk_size;
    int n2=dat->ndata-chunk_size-i;

    if (n2>chunk_size)
      n2=chunk_size;
    printf("n1 and n2 are %d %d\n",n1,n2);

    double t1=omp_get_wtime();
    my_burst_dm_transform(dat1,dat2,outbuf,n1,n2,tsamp,nnu,chan_map,depth);
    
    printf("did a transform.\n");

  }
  
  
}
