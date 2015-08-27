//    Copyright Jonathan Sievers, 2015.  All rights reserved.  This code may only be used with permission of the owner.

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#ifndef max
  #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define DM0 4148.8
//OLD DM0 4000.0
#define NOISE_PERIOD 64
#define SIG_THRESH 30.0
#define THREAD 8
#define OMP_THREADS 8

#include "dedisperse_gbt.h"


/*--------------------------------------------------------------------------------*/
float *vector(int n)
{
  float *vec=(float *)malloc(sizeof(float)*n);
  assert(vec);
  memset(vec,0,n*sizeof(float));
  return vec;
}
/*--------------------------------------------------------------------------------*/
int *ivector(int n)
{
  int *vec=(int *)malloc(sizeof(int)*n);
  assert(vec);
  memset(vec,0,n*sizeof(int));
  return vec;
}

/*--------------------------------------------------------------------------------*/
size_t *stvector(int n)
{
  size_t *vec=(size_t *)malloc(sizeof(size_t)*n);
  assert(vec);
  memset(vec,0,n*sizeof(size_t));
  return vec;
}

/*--------------------------------------------------------------------------------*/

long get_file_size(const char *fname)

{
  struct stat buf;
  if (stat(fname,&buf)==0)
    return (long)buf.st_size;
  else
    return 0;
}
/*--------------------------------------------------------------------------------*/
char **cmatrix(int n, int m)
{

  char *vec=(char *)malloc(n*m*sizeof(char));
  char **mat=(char **)malloc(n*sizeof(char *));
  for (int i=0;i<n;i++) 
    mat[i]=vec+i*m;
  return mat;
}
/*--------------------------------------------------------------------------------*/
float **matrix(long n, long m)
{

  float *vec=(float *)malloc(n*m*sizeof(float));
  float **mat=(float **)malloc(n*sizeof(float *));
  for (long i=0;i<n;i++) 
    mat[i]=vec+i*m;
  return mat;
}
/*--------------------------------------------------------------------------------*/
int get_nchan_from_depth(int depth)
{
  int i=1;
  return i<<depth;
}

/*--------------------------------------------------------------------------------*/
float get_diagonal_dm_simple(float nu1, float nu2, float dt, int depth)
{
  float d1=1.0/nu1/nu1;
  float d2=1.0/nu2/nu2;
  int nchan=get_nchan_from_depth(depth);
  //printf("nchan is %d from %d\n",nchan,depth);
  //printf("freqs are %12.4f %12.4f\n",nu1,nu2);
  float dm_max=dt/DM0/( (d2-d1)/nchan);
  //printf("current dm is %12.4f\n",dm_max);
  return fabs(dm_max);
  
}

//Does not work
float alex_diag_dm(float nu1,float nu2,float dt)
{
  float d1 = 1.0/nu1/nu1;
  float d2 = 1.0/nu2/nu2;
  float dm_max = dt/DM0/(d2 - d1);
  return fabs(dt/DM0/(d2 - d1));
}
/*--------------------------------------------------------------------------------*/

float get_diagonal_dm(Data *dat) {
  //diagonal DM is when the delay between adjacent channels
  //is equal to the sampling time
  //delay = dm0*dm/nu^2
  // delta delay = dm0*dm*(1//nu1^2 - 1/nu2^2) = dt
  float d1=1.0/dat->chans[0]/dat->chans[0];
  float d2=1.0/dat->chans[1]/dat->chans[1];
  float dm_max=dat->dt/DM0/(d2-d1);
  return dm_max;
}
/*--------------------------------------------------------------------------------*/
Data *read_gbt(const char *fname)
{
  Data *dat=(Data *)malloc(sizeof(Data));
  memset(dat,0,sizeof(Data));
  int nchan=4096;
  int npol=4;
  long ndat=get_file_size(fname);
  if (ndat<=0) {
    printf("FILE %s unavailable for reading.\n",fname);
    return NULL;
  }
  int nsamp=ndat/npol/nchan;
  printf("have %d samples.\n",nsamp);
  char **mat=cmatrix(nsamp,nchan);
  char *tmp=(char *)malloc(sizeof(char)*nchan*npol);
  FILE *infile=fopen(fname,"r");
  for (int i=0;i<nsamp;i++) {
    size_t nread=fread(tmp,sizeof(char),nchan*npol,infile);
    memcpy(mat[i],tmp,sizeof(char)*nchan);
  }
  fclose(infile);
  free(tmp);
  dat->raw_nchan=nchan;
  dat->ndata=nsamp;
  //dat->raw_data=mat;
  dat->raw_data=matrix(nchan,nsamp);
  for (int i=0;i<nchan;i++)
    for (int j=0;j<nsamp;j++)
      dat->raw_data[i][j]=mat[j][i];
  
  dat->raw_chans=(float *)malloc(sizeof(float)*dat->raw_nchan);
  dat->dt=1e-3;
  float dnu=(900-700.0)/dat->raw_nchan;
  for (int i=0;i<dat->raw_nchan;i++) {
    dat->raw_chans[i]=900-(0.5+i)*dnu;
  }

  free(mat[0]);
  free(mat);
  return dat;
}


/*--------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------*/
Data *map_chans(Data *dat, int depth)
{
  double t0=omp_get_wtime();
  int nchan=(1<<depth);
  //printf("expecting %d channels.\n",nchan);
  //Data *dat=(Data *)malloc(sizeof(Data));
  dat->nchan=nchan;
  dat->data=matrix(dat->nchan,dat->ndata);
  memset(dat->data[0],0,sizeof(dat->data[0][0])*dat->nchan*dat->ndata);

  dat->chans=vector(dat->nchan);
  float nu1=dat->raw_chans[0];
  float nu2=dat->raw_chans[dat->raw_nchan-1];
  double l1=1.0/nu1/nu1;
  double l2=1.0/nu2/nu2;
  double dlam=(l2-l1)/(dat->nchan-1);
  for (int i=0;i<dat->nchan;i++)  {
    //dat->chans[i]=1.0/sqrt(l1+(double)i*dlam);
    dat->chans[i]=1.0/sqrt(l2-(double)i*dlam);
  }
  dat->chan_map=stvector(dat->raw_nchan);
  
  for (int i=0;i<dat->raw_nchan;i++)  {
    float min_err=1e30;
    int best=-1;
    for (int j=0;j<dat->nchan;j++) {
      float delt=fabs(1.0/(dat->chans[j]*dat->chans[j])-1.0/(dat->raw_chans[i]*dat->raw_chans[i]));
      if (delt<min_err) {
        min_err=delt;
        best=j;
      }

    }
    dat->chan_map[i]=best;
  }
  printf("took %12.5f seconds to map channels.\n",omp_get_wtime()-t0);
  printf("max one-pass dispersion measure is %12.4f\n",get_diagonal_dm(dat));
  return NULL;
}
/*--------------------------------------------------------------------------------*/
void remap_data( Data *dat)
{        					
  //double t0=omp_get_wtime();
  assert(dat->chan_map);
  memset(dat->data[0],0,sizeof(dat->data[0][0])*dat->nchan*dat->ndata);  
  for (int i=0;i<dat->raw_nchan;i++) {
    int ii=dat->chan_map[i];
    for (int j=0;j<dat->ndata;j++)
      dat->data[ii][j]+=dat->raw_data[i][j];
  }
  //printf("took %12.5f seconds to remap data.\n",omp_get_wtime()-t0);
}
/*--------------------------------------------------------------------------------*/

int get_npass(int n)
{
  int nn=0;
  while (n>1) {
    nn++;
    n/=2;
  }
  return nn;
}

/*--------------------------------------------------------------------------------*/

static void dedisperse_block_subkernel_2pass(const float *in1, const float *in2, float *out)
{
  out[0]=in1[0]+in1[4]+in1[8]+in1[12];
  out[1]=in1[1]+in1[5]+in1[9]+in1[13];
  out[2]=in1[2]+in1[6]+in1[10]+in1[14];
  out[3]=in1[3]+in1[7]+in1[11]+in1[15];

  out[4]=in1[0]+in1[4]+in1[9]+in1[13];
  out[5]=in1[1]+in1[5]+in1[10]+in1[14];
  out[6]=in1[2]+in1[6]+in1[11]+in1[15];
  out[7]=in1[3]+in1[7]+in2[8]+in2[12];

  out[8] =in1[0]+in1[5]+in1[9]+in1[14];
  out[9] =in1[1]+in1[6]+in1[10]+in1[15];
  out[10]=in1[2]+in1[7]+in1[11]+in2[12];
  out[11]=in1[3]+in2[4]+in2[8]+in2[13];

  out[12] =in1[0]+in1[5]+in1[10]+in1[15];
  out[13] =in1[1]+in1[6]+in1[11]+in2[12];
  out[14] =in1[2]+in1[7]+in2[8]+in2[13];
  out[15] =in1[3]+in2[4]+in2[9]+in2[14];
}
/*--------------------------------------------------------------------------------*/
void dedisperse_block_kernel_2pass(const float **in, float **out, int n, int m) 
{
  
  int nset=n/4;
  float *oo=out[0];

  float ii1[16],ii2[16],myoo[16];

  

  for (int jj=0;jj<nset;jj++) {


    const float *i0=in[4*jj];
    const float *i1=in[4*jj+1];
    const float *i2=in[4*jj+2];
    const float *i3=in[4*jj+3];
    ii1[0]=i0[0];
    ii1[1]=i0[1];
    ii1[2]=i0[2];
    ii1[3]=i0[3];
    ii1[4]=i1[0];
    ii1[5]=i1[1];
    ii1[6]=i1[2];
    ii1[7]=i1[3];
    ii1[8]=i2[0];
    ii1[9]=i2[1];
    ii1[10]=i2[2];
    ii1[11]=i2[3];
    ii1[12]=i3[0];
    ii1[13]=i3[1];
    ii1[14]=i3[2];
    ii1[15]=i3[3];

    
  //#pragma omp parallel for

    for (int i=4*jj;i<m-8;i+=4) {
      ii2[0]=i0[4+i];
      ii2[1]=i0[5+i];
      ii2[2]=i0[6+i];
      ii2[3]=i0[7+i];
      ii2[4]=i1[4+i];
      ii2[5]=i1[5+i];
      ii2[6]=i1[6+i];
      ii2[7]=i1[7+i];
      ii2[8]=i2[4+i];
      ii2[9]=i2[5+i];
      ii2[10]=i2[6+i];
      ii2[11]=i2[7+i];
      ii2[12]=i3[4+i];
      ii2[13]=i3[5+i];
      ii2[14]=i3[6+i];
      ii2[15]=i3[7+i];
      dedisperse_block_subkernel_2pass(ii1,ii2,myoo);
            
      
      oo[jj*m+i+0]=myoo[0];
      oo[jj*m+i+1]=myoo[1];
      oo[jj*m+i+2]=myoo[2];
      oo[jj*m+i+3]=myoo[3];
      
      oo[(jj+nset)*m+i-jj]=myoo[4];
      oo[(jj+nset)*m+i-jj+1]=myoo[5];
      oo[(jj+nset)*m+i-jj+2]=myoo[6];
      oo[(jj+nset)*m+i-jj+3]=myoo[7];
      
      oo[(jj+2*nset)*m+i-2*jj]=myoo[8];
      oo[(jj+2*nset)*m+i-2*jj+1]=myoo[9];
      oo[(jj+2*nset)*m+i-2*jj+2]=myoo[10];
      oo[(jj+2*nset)*m+i-2*jj+3]=myoo[11];
      
      oo[(jj+3*nset)*m+i-3*jj]=myoo[12];
      oo[(jj+3*nset)*m+i-3*jj+1]=myoo[13];
      oo[(jj+3*nset)*m+i-3*jj+2]=myoo[14];
      oo[(jj+3*nset)*m+i-3*jj+3]=myoo[15];
      
      memcpy(ii1,ii2,16*sizeof(float));
    }
  }
}
/*--------------------------------------------------------------------------------*/

void dedisperse_kernel(float **in, float **out, int n, int m)
{

  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<m;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=0;i<m-jj-1;i++) 
      out[jj+npair][i]=in[2*jj][i+jj]+in[2*jj+1][i+jj+1];
    
  }
}
/*--------------------------------------------------------------------------------*/

void dedisperse_kernel_v2(float **in, float **out, int n, int m)
{

  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<jj;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=jj;i<m-1;i++) {
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
      out[jj+npair][i-jj]=in[2*jj][i]+in[2*jj+1][i+1];
    }
    
  }
}

/*--------------------------------------------------------------------------------*/

void dedisperse(float **inin, float **outout, int nchan,int ndat)
{
  //return;
  int npass=get_npass(nchan);
  //printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  for (int i=0;i<npass;i++) {    
    //#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      dedisperse_kernel(in+j,out+j,bs,ndat);
    }
    bs/=2;
    float **tmp=in;
    in=out;
    out=tmp;
  }
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}

/*--------------------------------------------------------------------------------*/
void dedisperse_blocked_cached(float **dat, float **dat2, int nchan, int ndat)
{
  //int nchan1=128;
  //int chunk_size=768;

  int nchan1=128;
  //int chunk_size=1536;
  int chunk_size=1024;
  int nchunk=ndat/chunk_size;
  int npass1=get_npass(nchan1);
  int npass=get_npass(nchan);
  int npass2=npass-npass1;
  int nchan2=nchan/nchan1;

  int nblock=nchan/nchan1;
  int nblock2=nchan/nchan2;


  
#pragma omp parallel 
  {
    float **tmp1=matrix(nchan1,chunk_size+nchan1);
    float **tmp2=matrix(nchan1,chunk_size+nchan1); 
    
#pragma omp for collapse(2) schedule(dynamic,2)
    for (int i=0;i<nblock;i++) {      
      //printf("i is %d\n",i);
      for (int j=0;j<nchunk;j++) {
        int istart=j*chunk_size;
        int istop=(j+1)*chunk_size+nchan1;
        if (istop>ndat) {
          istop=ndat;
          for (int k=0;k<nchan1;k++)
            memset(tmp1[k]+chunk_size,0,sizeof(float)*nchan1);
        }
        for (int k=0;k<nchan1;k++)
          memcpy(tmp1[k],&(dat[i*nchan1+k][istart]),(istop-istart)*sizeof(float));
        
        dedisperse(tmp1,tmp2,nchan1,chunk_size+nchan1);
        
        for (int k=0;k<nchan1;k++)
          memcpy(&(dat2[i*nchan1+k][istart]),tmp1[k],chunk_size*sizeof(float));
      }
    }
#if 1
    free(tmp1[0]);   
    free(tmp1);
    free(tmp2[0]);
    free(tmp2);
#endif
  }
  
  

  float **dat_shift=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nblock;i++)
    for (int j=0;j<nchan1;j++)
      dat_shift[j*nblock+i]=dat2[i*nchan1+j]+i*j;  


  //recalculate block sizes to keep amount in cache about the same
  int nelem=nchan1*chunk_size;
  chunk_size=nelem/nchan2;
  nchunk=ndat/chunk_size;

#pragma omp parallel 
  {
    float **tmp1=matrix(nchan2,chunk_size+nchan2);
    float **tmp2=matrix(nchan2,chunk_size+nchan2); 

#pragma omp for  collapse(2) schedule(dynamic,4)
    for (int i=0;i<nblock2;i++) {      
      //printf("i is now %d\n",i);
      for (int j=0;j<nchunk;j++) {
        int istart=j*chunk_size;
        int istop=(j+1)*chunk_size+nchan2;
        if (istop>ndat) {
          istop=ndat;
          for (int k=0;k<nchan2;k++)
            memset(tmp1[k]+chunk_size,0,sizeof(float)*nchan2);
        }
        for (int k=0;k<nchan2;k++) {
          memcpy(tmp1[k],dat_shift[i*nchan2+k]+istart,(istop-istart)*sizeof(float));
        }
        dedisperse(tmp1,tmp2,nchan2,chunk_size+nchan2);
        for (int k=0;k<nchan2;k++)
          memcpy(dat[i*nchan2+k]+istart,tmp2[k],chunk_size*sizeof(float));
      }
    }
    free(tmp1[0]);
    free(tmp1);
    free(tmp2[0]);
    free(tmp2);
    
  }
  //printf("Finished dedispersion.\n");
}


/*--------------------------------------------------------------------------------*/

void dedisperse_inplace(float **inin, int nchan, int m)
{
  omp_set_dynamic(0);
  omp_set_num_threads(OMP_THREADS);

  int npass=get_npass(nchan);

  float **in=inin;

  int radix = 1;
  int pairs = nchan/2;
  int threads = 8;

  //initial channel map
  int *fmap = malloc(sizeof(int)*nchan);

  for (int i=0; i<nchan;i++){
    fmap[i] = i;
  }

  //float *vec = (float*)malloc(sizeof(float)*OMP_THREADS*m);
  //float **tmp = (float**)malloc(sizeof(float*)*OMP_THREADS);
  //for(int i = 0; i < OMP_THREADS; i++){

  //}
  float **tmp = matrix(OMP_THREADS,m);

  for (int i=0;i<npass;i++) {

    generate_shift_group(fmap,radix,nchan);

    #pragma omp parallel for
    for (int j=0;j<pairs;j++) {
      int zero = 2*j;
      int zero_ind = 0;
      int id = omp_get_thread_num();
      //Inefficient, but it scans over at most n/2
      while(fmap[zero_ind] != zero - (zero % (nchan/radix))){
        zero_ind++;
      }
      zero_ind += radix*(zero % (nchan/radix));

      int comp_ind = zero_ind + radix;

      int jeff = j % (nchan/(radix*2));

      for(int k = 0; k < m; k++){
        tmp[id][k] = in[zero_ind][k];
      }
      for(int k = 0; k < m; k++){
       in[zero_ind][k] = in[zero_ind][k] + in[comp_ind][k];
      }
      for(int k = 0; k < m - jeff - 1; k++){
       in[comp_ind][k] = tmp[id][k + jeff] + in[comp_ind][k + jeff + 1];
      }
    }

    radix *=2;
  }

  fast_unshuffle(in,fmap,nchan,m);
  //unshuffle(in,fmap,nchan,m);
  free(fmap);
  free(tmp);
  free(tmp[0]);
}

void fast_unshuffle(float **data, int* fmap, int nchan, int m){
  float* zero = data[0];
  float ** tmp = malloc(sizeof(float*)*nchan);
  for(int i = 0; i < nchan; i++){
    tmp[i] = &(data[i][0]);
  }

  for(int i = 0; i < nchan; i++){
    //for any alignment
    data[fmap[i]] = tmp[i];
  }
  free(tmp);
}

void unshuffle(float **data, int* fmap,int nchan, int m){

  float *tmp = (float*)malloc(sizeof(float*)*m);

  int send_ind = nchan/2;
  int this_ind = 1;
  int sorted = 0;

  while(sorted == 0){
    for(int j=0; j < m; j++){
      tmp[j] = data[send_ind][j];
    }

    for(int j=0; j < m; j++){    
      data[send_ind][j] = data[this_ind][j];
    }

    for(int j=0; j < m; j++){    
      data[this_ind][j] = tmp[j];
    }

    fmap[this_ind] = fmap[send_ind];
    fmap[send_ind] = send_ind;

    send_ind = fmap[this_ind];
    if(this_ind == send_ind){
      int count = 0;
      while(fmap[this_ind % nchan] == this_ind % nchan){
        this_ind++;
        count++;
        if(count == nchan){
          sorted = 1;
          break;
        }
      }
      this_ind = this_ind % nchan;
      send_ind = fmap[this_ind];
    }
  }
  free(tmp);
}

/*--------------------------------------------------------------------------------*/
void generate_shift_group(int *inin, int radix, int n)
{
  int *in = inin;

  int nrad = n/radix;
  int nrad2 = nrad/2;

  for (int i = 0; i < (radix); i++){
    int val = in[i];
    in[i] = (val % (nrad))/2 + (val % 2)*(nrad) + (val/nrad)*(nrad);
  }

  //Can easily improve by referencing only the base array
  //of size 2^j
  for (int i = radix; i < n; i++){
    in[i] = in[i % radix] + (i/radix);
  }

}

void dedisperse_single(float **inin, float **outout, int nchan,int ndat)
{
  //omp_set_num_threads(8);
  int npass=get_npass(nchan);
  //printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;

  float **out=outout;
  //FILE *fout;
  //fout = fopen('/var/log/burst_bench.log', 'w');
  
  //fclose(fout);
//  omp_set_dynamic(0);
//  omp_set_num_threads(8);

  for (int i=0;i<npass;i++) {    
#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      //printf("dedisperse using %i threads\n",omp_get_num_threads());
      dedisperse_kernel(in+j,out+j,bs,ndat);
    }
    bs/=2;
    float **tmp=in;
    in=out;
    out=tmp;
  }
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}
/*--------------------------------------------------------------------------------*/

void dedisperse_dual(float **inin, float **outout, int nchan,int ndat)
{
  int npass=get_npass(nchan);
  //printf("need %d passes from %d channels..\n",npass,nchan);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  //the npasss-1 is so that we stop in time to hand the final pass to 
  //the single-step kernel in the event of an odd depth.
  for (int i=0;i<npass-1;i+=2) {    
#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      //dedisperse_kernel_2pass_v2(in+j,out+j,bs,ndat);
      dedisperse_block_kernel_2pass((const float **)(in+j),out+j,bs,ndat);
    }
    bs/=4;
    float **tmp=in;
    in=out;
    out=tmp;
  }


  if (npass%2==1) {
    //do a single step if we come in with odd depth
    //printf("doing final step for odd depth with block size %d.\n",bs);
#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs)
      dedisperse_kernel(in+j,out+j,bs,ndat);
    float **tmp=in;
    in=out;
    out=tmp;
    
  }
  
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}
/*--------------------------------------------------------------------------------*/
void dedisperse_gbt(Data *dat, float *outdata)
{
  
  //remap_data(dat);  
  //float **tmp=matrix(dat->nchan, dat->ndata);


  //float **tmp=(float **)malloc(sizeof(float *)*dat->nchan);
  //for (int i=0;i<dat->nchan;i++)
    //tmp[i]=outdata+i*dat->ndata;

  //memset(tmp[0],0,sizeof(tmp[0][0])*dat->ndata*dat->nchan);

  double t1=omp_get_wtime();

  //dedisperse_dual(dat->data,tmp,dat->nchan,dat->ndata);  


  //float **ip_dat =(float **)malloc(sizeof(float *)*dat->nchan);

  //for (int i=0;i<dat->nchan;i++)
    //ip_dat[i]=dat->data[0]+i*dat->ndata;

  //memcpy(ip_dat[0],dat->data[0],dat->nchan*dat->ndata*sizeof(dat->data[0][0]));

  //dedisperse_blocked_cached_inplace(dat->data,dat->nchan,dat->ndata,64,1024);
  dedisperse_inplace(dat->data,dat->nchan,dat->ndata);
  
  for(int i = 0; i < dat->nchan; i++){
    memcpy(outdata + i*dat->ndata, dat->data[i],dat->ndata*sizeof(float));
  }

  if (0) {
    printf("element 500,300 is %12.5e\n",dat->data[500][300]);
    FILE *outfile=fopen("burst_dump.dat","w");
    fwrite(&(dat->nchan),sizeof(dat->nchan),1,outfile);
    fwrite(&(dat->ndata),sizeof(dat->ndata),1,outfile);
    fwrite(outdata,sizeof(outdata[0]),dat->nchan*dat->ndata,outfile);
    fclose(outfile);
  }
}
/*--------------------------------------------------------------------------------*/

void dedisperse_gbt_jon(Data *dat, float *outdata)
{
  
  //remap_data(dat);  
  //float **tmp=matrix(dat->nchan, dat->ndata);


  float **tmp=(float **)malloc(sizeof(float *)*dat->nchan);
  for (int i=0;i<dat->nchan;i++)
    tmp[i]=outdata+i*dat->ndata;

  memset(tmp[0],0,sizeof(tmp[0][0])*dat->ndata*dat->nchan);

  double t1=omp_get_wtime();

  //dedisperse_dual(dat->data,tmp,dat->nchan,dat->ndata);

  //memcpy(ip_dat[0],dat->data[0],dat->nchan*dat->ndata*sizeof(dat->data[0][0]));

  //dedisperse_inplace(ip_dat,dat->nchan,dat->ndata);  
  //dedisperse_single(dat->data,tmp,dat->nchan,dat->ndata);
  dedisperse_blocked_cached(dat->data,tmp,dat->nchan,dat->ndata);
  
  //printf("took %12.4f seconds to dedisperse.\n",omp_get_wtime()-t1);
  for(int i = 0; i < dat->nchan; i++){
    memcpy(outdata + i*dat->ndata,dat->data[i],dat->ndata*sizeof(outdata[0]));
  }
  free(tmp);

  if (0) {
    printf("element 500,300 is %12.5e\n",dat->data[500][300]);
    FILE *outfile=fopen("burst_dump.dat","w");
    fwrite(&(dat->nchan),sizeof(dat->nchan),1,outfile);
    fwrite(&(dat->ndata),sizeof(dat->ndata),1,outfile);
    fwrite(outdata,sizeof(outdata[0]),dat->nchan*dat->ndata,outfile);
    fclose(outfile);
  }
}


void clean_cols(Data *dat)
{
  for (int i=0;i<dat->raw_nchan;i++) {
    float tot=0;
    for (int j=0;j<dat->ndata;j++)
      tot+=dat->raw_data[i][j];
    tot/=dat->ndata;
    for (int j=0;j<dat->ndata;j++)
      dat->raw_data[i][j]-=tot;
  }
}
/*--------------------------------------------------------------------------------*/
void clean_rows_weighted(Data *dat,float *weights)
{
  float *tot=vector(dat->ndata);
  memset(tot,0,sizeof(tot[0])*dat->ndata);
#pragma omp parallel 
  {
    float *mytot=vector(dat->ndata);
    memset(mytot,0,sizeof(mytot[0])*dat->ndata);
#pragma omp for
    for (int i=0;i<dat->raw_nchan;i++) {
      for (int j=0;j<dat->ndata;j++) {
        mytot[j]+=dat->raw_data[i][j]*weights[i];
      }
    }
#pragma omp critical
    for (int i=0;i<dat->ndata;i++)
      tot[i]+=mytot[i];
  }
  float wt_sum=0;
  for (int i=0;i<dat->raw_nchan;i++)
    wt_sum+=weights[i];
  for (int i=0;i<dat->ndata;i++)
    tot[i]/=wt_sum;
  
  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<dat->ndata;j++)
      dat->raw_data[i][j]-=tot[i];
  free(tot);
    
}
/*--------------------------------------------------------------------------------*/
void clean_rows(Data *dat)
{
  float *tot=vector(dat->ndata);
  memset(tot,0,sizeof(tot[0])*dat->ndata);
  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<dat->ndata;j++)
      tot[j]+=dat->raw_data[i][j];

  for (int j=0;j<dat->ndata;j++)
    tot[j]/=dat->raw_nchan;

  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<dat->ndata;j++)
      dat->raw_data[i][j]-=tot[j];
  
  free(tot);
  
}
/*--------------------------------------------------------------------------------*/
void get_omp_iminmax(int n,  int *imin, int *imax)
{
  int nthreads=omp_get_num_threads();
  int myid=omp_get_thread_num();
  int bs=n/nthreads;
  *imin=myid*bs;
  *imax=(myid+1)*bs;
  if (*imax>n)
    *imax=n;

}
/*--------------------------------------------------------------------------------*/
void clean_rows_2pass(float *vec, size_t nchan, size_t ndata)
{
  float **dat=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nchan;i++)
    dat[i]=vec+i*ndata;
  
  float *tot=vector(ndata);
  memset(tot,0,sizeof(tot[0])*ndata);

  //find the common mode based on averaging over channels.
  //inner loop is the natural one to parallelize over, but for some
  //architectures/compilers doing so with a omp for is slow, hence
  //rolling my own.
#pragma omp parallel shared(ndata,nchan,dat,tot) default(none)
  {
    int imin,imax;
    get_omp_iminmax(ndata,&imin,&imax);
    
    for (int i=0;i<nchan;i++)
      for (int j=imin;j<imax;j++)
        tot[j]+=dat[i][j];
    
    for (int j=imin;j<imax;j++)
      tot[j]/=nchan;
  }
  //#define BURST_DUMP_DEBUG
#ifdef BURST_DUMP_DEBUG
  FILE *outfile=fopen("common_mode_1.dat","w");
  fwrite(tot,sizeof(float),ndata,outfile);
  fclose(outfile);
#endif

  float *amps=vector(nchan);
  memset(amps,0,sizeof(amps[0])*nchan);
  float totsqr=0;
  for (int i=0;i<ndata;i++)
    totsqr+=tot[i]*tot[i];

  //find the best-fit amplitude for each channel
#pragma omp parallel for shared(ndata,nchan,dat,tot,amps,totsqr) default(none)
  for (int i=0;i<nchan;i++) {
    float myamp=0;
    for (int j=0;j<ndata;j++)
      myamp+=dat[i][j]*tot[j];
    myamp/=totsqr;
    //for (int j=0;j<ndata;j++)
    //   dat[i][j]-=tot[j]*myamp;
    amps[i]=myamp;    
  }

#ifdef BURST_DUMP_DEBUG
  outfile=fopen("mean_responses1.txt","w");
  for (int i=0;i<nchan;i++)
    fprintf(outfile,"%12.4f\n",amps[i]);
  fclose(outfile);
#endif

  //decide that channels with amplitude between 0.5 and 1.5 are the good ones.
  //recalculate the common mode based on those guys, with appropriate calibration
  memset(tot,0,sizeof(tot[0])*ndata);
  float amp_min=0.5;
  float amp_max=1.5;
#pragma omp parallel shared(ndata,nchan,amps,dat,tot,amp_min,amp_max) default(none)
  {
    int imin,imax;
    get_omp_iminmax(ndata,&imin,&imax);

    for (int i=0;i<nchan;i++) {
      if ((amps[i]>amp_min)&&(amps[i]<amp_max))
        for (int j=imin;j<imax;j++)
          tot[j]+=dat[i][j]/amps[i];
    }
  }
  float tot_sum=0;
  for (int i=0;i<nchan;i++)
    if ((amps[i]>amp_min)&&(amps[i]<amp_max))
      tot_sum+=1./amps[i];
  totsqr=0;
  for (int i=0;i<ndata;i++) {
    tot[i]/=tot_sum;
    totsqr+=tot[i]*tot[i];
  }
  
#ifdef BURST_DUMP_DEBUG
  outfile=fopen("common_mode_2.dat","w");
  fwrite(tot,sizeof(float),ndata,outfile);
  fclose(outfile);

  {
    float *chansum=vector(nchan);
    float *chansumsqr=vector(nchan);
#pragma omp parallel for
    for (int i=0;i<nchan;i++)
      for (int j=0;j<ndata;j++) {
        chansum[i]+=dat[i][j];
        chansumsqr[i]+=dat[i][j]*dat[i][j];
      }
    outfile=fopen("chan_variances_pre.txt","w");
    for (int i=0;i<nchan;i++)
      fprintf(outfile,"%12.6e\n",sqrt(chansumsqr[i]-chansum[i]*chansum[i]/ndata));
    fclose(outfile);
    free(chansum);
    free(chansumsqr);
  }
#endif



  memset(amps,0,sizeof(amps[0])*nchan);
#pragma omp parallel for shared(ndata,nchan,amps,dat,tot,totsqr) default(none)
  for (int i=0;i<nchan;i++) {
    float myamp=0;
    for (int j=0;j<ndata;j++) 
      myamp+=dat[i][j]*tot[j];
    myamp/=totsqr;
    amps[i]=myamp;
    for (int j=0;j<ndata;j++)
      dat[i][j]-=tot[j]*myamp;
  }
  
#ifdef BURST_DUMP_DEBUG
  outfile=fopen("mean_responses2.txt","w");
  for (int i=0;i<nchan;i++)
    fprintf(outfile,"%12.4f\n",amps[i]);
  fclose(outfile);


  {
    float *chansum=vector(nchan);
    float *chansumsqr=vector(nchan);
#pragma omp parallel for
    for (int i=0;i<nchan;i++)
      for (int j=0;j<ndata;j++) {
        chansum[i]+=dat[i][j];
        chansumsqr[i]+=dat[i][j]*dat[i][j];
      }
    outfile=fopen("chan_variances_post.txt","w");
    for (int i=0;i<nchan;i++)
      fprintf(outfile,"%12.6e\n",sqrt(chansumsqr[i]-chansum[i]*chansum[i]/ndata));
    fclose(outfile);
    free(chansum);
    free(chansumsqr);
  }


  #endif
  
  


  free(amps);
  free(tot);
  
}
/*--------------------------------------------------------------------------------*/
int find_cal_phase(float **dat, int nchan, int period)
{
  //find the phase of the noise cal.  Do it by summing across 
  float *tot=vector(2*period);
  memset(tot,0,sizeof(tot[0])*2*period);
  for (int i=0;i<nchan;i++)
    for (int j=0;j<period;j++) {
      tot[j]+=dat[i][j];
    }
  for (int j=0;j<period;j++) {
    tot[j+period]=tot[j];
  }
  FILE *outfile=fopen("noisecal.txt","w");
  for (int i=0;i<2*period;i++) {
    fprintf(outfile,"%14.5e\n",tot[i]);
  }
  fclose(outfile);
  float max=-1e30;
  int imax=0;
  //find the peak by finding the maximum positive jump in the noise cal
  for (int i=0;i<period;i++) {
    float tmp=tot[i+1]-tot[i];
    if (tmp>max) {
      max=tmp;
      imax=i;
    }
  }
  //now check to see if the next sample is significantly higher.  If so, it's probably a better starting point.
  if (tot[imax+2]-tot[imax+1]>0.5*(tot[imax+1]-tot[imax]))
    imax++;
  free(tot);
  //printf("noise cal phase is %d\n",imax);
  return imax;
}
/*--------------------------------------------------------------------------------*/
void find_cals(float **tot, int nchan, int period, int phase, float *cal)
{
  for (int i=0;i<nchan;i++) {
    cal[i]=0;
    for (int j=0;j<period/2;j++) 
      cal[i]+=tot[i][ (j+phase)%period];
    for (int j=period/2;j<period;j++)
      cal[i]-=tot[i][(j+phase)%period];
  }
  float norm=0;
  for (int i=0;i<nchan;i++)
    norm+=cal[i];
  norm/=nchan;
  for (int i=0;i<nchan;i++)
    cal[i]/=norm;
}
/*--------------------------------------------------------------------------------*/
void remove_noisecal(Data *dat, int period, int apply_calib) //, float *cal_facs)
{
  double t1=omp_get_wtime();
  float **tot=matrix(dat->raw_nchan, period);
  float **nn=matrix(dat->raw_nchan, period);
  memset(nn[0],0,sizeof(nn[0][0])*dat->raw_nchan*period);
  memset(tot[0],0,sizeof(tot[0][0])*dat->raw_nchan*period);



#pragma omp parallel for shared(tot,nn,dat,period) default(none)
  for (int i=0;i<dat->raw_nchan;i++) {
    int jj=0;
    for (int j=0;j<dat->ndata;j++) {
      //int jj=j%period;
      tot[i][jj]+=dat->raw_data[i][j];
      nn[i][jj]++;
      jj++;
      if (jj==period)
        jj=0;
    }
  }
  

  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<period;j++)
      tot[i][j]/=nn[i][j];
  
  
  if (0)  {
    FILE *outfile=fopen("noise_cal_template.txt","w");
    for (int i=0;i<dat->raw_nchan;i++) {
      for (int j=0;j<period;j++)
        fprintf(outfile,"%14.6e ",tot[i][j]);
      fprintf(outfile,"\n");
    }
    fclose(outfile);
  }
  
#pragma omp parallel for shared(tot,dat,period) default(none)
  for (int i=0;i<dat->raw_nchan;i++) {
    int jj=0;
    for (int j=0;j<dat->ndata;j++) {
      //dat->raw_data[i][j]-=tot[i][j%period];
      dat->raw_data[i][j]-=tot[i][jj];
      jj++;
      if (jj==period)
        jj=0;
    }
  }
  
  //if (cal_facs) {
  if (apply_calib) {
    //printf("doing calibration in remove_noisecal.\n");
    int my_phase=find_cal_phase(tot,dat->raw_nchan,period);
    float *calib=vector(dat->raw_nchan);
    find_cals(tot,dat->raw_nchan,period,my_phase,calib);
    //FILE *outfile=fopen("my_cals.txt","w");
    //for (int i=0;i<dat->raw_nchan;i++)
    //  fprintf(outfile,"%12.4e\n",calib[i]);
    //fclose(outfile);
#pragma omp parallel for
    for (int i=0;i<dat->raw_nchan;i++) 
      if (calib[i]>0)
        for (int j=0;j<dat->ndata;j++)
          dat->raw_data[i][j]/=calib[i];
    free(calib);
    
    //memset(cal_facs,0,sizeof(cal_facs[0]*dat->raw_nchan));
    //for (int i=0;i<dat->raw_nchan;i++) {
    //  for (int j=0;j<period/2;j++
    //           }
  }

  free(nn[0]);
  free(nn);
  free(tot[0]);
  free(tot);

  //printf("removed noisecal in %12.4f seconds.\n",omp_get_wtime()-t1);
}
/*--------------------------------------------------------------------------------*/
float *find_sigmas(float **mat, int n, int m)
{
  float *totsqr=vector(n);
  memset(totsqr,0,sizeof(totsqr[0])*n);
  float *tot=vector(n);
  memset(tot,0,sizeof(tot[0])*n);
  
#pragma omp parallel for shared(mat,n,m,totsqr,tot) default(none)
  for (int i=0;i<n;i++) 
    for (int j=0;j<m;j++) {
      float val=mat[i][j];
      tot[i]+=val;
      totsqr[i]+=val*val;
    }
  
  float *sigs=vector(n);
  for (int i=0;i<n;i++) {
    tot[i]/=m;
    totsqr[i]/=m;
    sigs[i]=sqrt(totsqr[i]-tot[i]*tot[i]);
  }
  free(tot);
  free(totsqr);
  return sigs;
}
/*--------------------------------------------------------------------------------*/
void clean_outliers(Data *dat, float sig_thresh, float *sigs_out)
{
  //find all points greater than sig_thresh times the standard deviation and replace with zero
  //if noise cal has already been removed, mean should be zero coming into here.
  
  double t1=omp_get_wtime();


  float *sigs=find_sigmas(dat->raw_data,dat->raw_nchan,dat->ndata);
  if (sigs_out)
    memcpy(sigs_out,sigs,sizeof(sigs[0])*dat->raw_nchan);
  for (int i=0;i<dat->raw_nchan;i++)
    sigs[i]*=sig_thresh;
  
#pragma omp parallel for shared(dat,sig_thresh,sigs) default(none)
  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<dat->ndata;j++) {
      if (dat->raw_data[i][j]>sigs[i])
        dat->raw_data[i][j]=0;       
    }
  
  free(sigs);
  //printf("took %12.4f seconds to clean outliers.\n",omp_get_wtime()-t1);
}
/*--------------------------------------------------------------------------------*/
void zap_bad_channels(Data *dat, char *fname)
{
  FILE *infile=fopen(fname,"r");
  int nbad;
  size_t nread=fread(&nbad,1,sizeof(nbad),infile);
  int *bad=ivector(nbad);
  nread=fread(bad,nbad,sizeof(bad[0]),infile);
  fclose(infile);
  printf("zapping %d bad channels.\n",nbad);
  for (int i=0;i<nbad;i++) {
    printf("%4d ",bad[i]);    
  }
  printf("\n");
  for (int i=0;i<nbad;i++) 
    for (int j=0;j<dat->ndata;j++) {
      dat->raw_data[bad[i]][j]=0.0;
    }
  free(bad);
}
/*--------------------------------------------------------------------------------*/
void sigs2weights(float *sigs, int nchan, float thresh)
//convert channel sigmas into weights, optionally cutting low-weight channels
{
  float tot=0;
  for (int i=0;i<nchan;i++) {
    if (sigs[i]>0) {
      sigs[i]=1./sigs[i]/sigs[i];
      tot+=sigs[i];
    }
  }
  tot/=nchan;
  if (thresh>0)
    for (int i=0;i<nchan;i++)
      if (sigs[i]<tot*thresh)
        sigs[i]=0;
  
}
/*--------------------------------------------------------------------------------*/
void apply_weights(Data *dat, float *weights)
{
  #pragma omp parallel for
  for (int i=0;i<dat->raw_nchan;i++)
    for (int j=0;j<dat->ndata;j++)
      dat->raw_data[i][j]*=weights[i];
  
}
/*--------------------------------------------------------------------------------*/
void setup_data(Data *dat)
{
  //clean_cols(dat);

  
  remove_noisecal(dat,NOISE_PERIOD,1);
  
  if (1) {
    float *sigs=vector(dat->raw_nchan);
    clean_outliers(dat,SIG_THRESH,sigs);
    sigs2weights(sigs,dat->raw_nchan,0.3);
    clean_rows(dat);
    apply_weights(dat,sigs);
    
    free(sigs);
  }
  //remove_noisecal(dat,NOISE_PERIOD);


  //zap_bad_channels(dat,"bad_chans.dat");
  


  remap_data(dat);

  
}

/*--------------------------------------------------------------------------------*/
size_t get_burst_nextra(size_t ndata2, int depth)
{
  size_t nchan=get_nchan_from_depth(depth);
  size_t nextra=nchan;
  if (nextra>ndata2)
    nextra=ndata2;
  
  
  return nextra;
}
/*--------------------------------------------------------------------------------*/
void copy_in_data(Data *dat, float *indata1, int ndata1, float *indata2, int ndata2)
{
  int npad=dat->ndata-ndata1;
  memset(dat->raw_data[0],0,dat->raw_nchan*dat->ndata*sizeof(float));
  //printf("npad is %d\n",npad);
  assert(ndata2>=npad);
  
  for (int i=0;i<dat->raw_nchan;i++) {
    for (int j=0;j<ndata1;j++) {

      //this line changes depending on memory ordering of input data
      //dat->raw_data[i*dat->ndata+j]=indata1[i*ndata1+j];      
#ifdef BURST_DM_NOTRANSPOSE
      dat->raw_data[i][j]=indata1[i*ndata1+j];      
#else
      dat->raw_data[i][j]=indata1[j*dat->raw_nchan+i];
#endif


      //dat->raw_data[i][j]=0;
    }
    for (int j=0;j<npad;j++) {
      //this line also changes depending on memory ordering of input data
      //dat->raw_data[i*dat->ndata+ndata1+j]=indata2[i*ndata2+j];

#ifdef BURST_DM_NOTRANSPOSE
      dat->raw_data[i][ndata1+j]=indata2[i*ndata2+j];
#else
      dat->raw_data[i][ndata1+j]=indata2[i+j*dat->raw_nchan];
#endif
      
    }
  }
  //printf("data are put inside.\n");
}

/*--------------------------------------------------------------------------------*/
Data *put_data_into_burst_struct(float *indata1, float *indata2, size_t ntime1, size_t ntime2, size_t nfreq, size_t *chan_map, int depth)
{
  
  Data *dat=(Data *)calloc(1,sizeof(Data));
  dat->raw_nchan=nfreq;
  int nchan=get_nchan_from_depth(depth);
  //printf("expecting %d channels.\n",nchan);
  dat->nchan=nchan;
  int nextra=get_burst_nextra(ntime2,depth);
  dat->ndata=ntime1+nextra;
  dat->raw_data=matrix(dat->raw_nchan,dat->ndata);
  dat->chan_map=chan_map;
  dat->data=matrix(dat->nchan,dat->ndata);
  copy_in_data(dat,indata1,ntime1,indata2,ntime2);
  
  return dat;
}

/*--------------------------------------------------------------------------------*/

void find_4567_peaks_wnoise(float *vec, int nsamp, Peak *peak4, Peak *peak5, Peak *peak6, Peak *peak7)
{
  float s4=0,s5=0,s6=0,s7=0;
  float v4=0,v5=0,v6=0,v7=0;
  peak4->ind=0;
  peak5->ind=0;
  peak6->ind=0;
  peak7->ind=0;
  peak4->duration=4;
  peak5->duration=5;
  peak6->duration=6;
  peak7->duration=7;

  float cur4=vec[2]+vec[3]+vec[4]+vec[5];
  peak4->peak=cur4;
  peak5->peak=cur4+vec[6];
  float cur6=vec[0]+vec[1]+cur4;
  peak6->peak=cur6;
  peak7->peak=cur6+vec[6];
  for (int i=6;i<nsamp;i++) {
    cur4=cur4+vec[i];
    s5+=cur4;
    v5+=cur4*cur4;
    if (cur4>peak5->peak) {
      peak5->peak=cur4;
      peak5->ind=i;
    }

    cur4=cur4-vec[i-4];
    s4+=cur4;
    v4+=cur4*cur4;
    if (cur4>peak4->peak) {
      peak4->peak=cur4;
      peak4->ind=i;
    }

    cur6=cur6+vec[i];
    s7+=cur6;
    v7+=cur6*cur6;
    if (cur6>peak7->peak) {
      peak7->peak=cur6;
      peak7->ind=i;
    }

    cur6=cur6-vec[i-6];
    s6+=cur6;
    v6+=cur6*cur6;
    if (cur6>peak6->peak) {
      peak6->peak=cur6;
      peak6->ind=i;
    }

  }
  float n4,n5,n6,n7;
  s4/=(nsamp-7);
  s5/=(nsamp-7);
  s6/=(nsamp-7);
  s7/=(nsamp-7);
  v4/=(nsamp-7);
  v5/=(nsamp-7);
  v6/=(nsamp-7);
  v7/=(nsamp-7);

  n4=sqrt(v4-s4*s4);
  n5=sqrt(v5-s5*s5);
  n6=sqrt(v6-s6*s6);
  n7=sqrt(v7-s7*s7);

  peak4->snr=(peak4->peak-s4)/n4;
  peak5->snr=(peak5->peak-s5)/n5;
  peak6->snr=(peak6->peak-s6)/n6;
  peak7->snr=(peak7->peak-s7)/n7;

  peak4->noise=n4;
  peak5->noise=n5;
  peak6->noise=n6;
  peak7->noise=n7;
}

/*--------------------------------------------------------------------------------*/


Peak find_peaks_wnoise_onedm(float *vec, int nsamples, int max_depth, int cur_depth)
{

  int wt=1<<cur_depth;



  Peak best;  
  best.snr=0;
  best.peak=0;
  //do the 1/2/3 sample case on the first pass through
  if (cur_depth==0) {
    float best1=0;
    float best2=0;
    float best3=0;
    float s1=0,s2=0,s3=0;
    float v1=0,v2=0,v3=0;
    int i1=0,i2=0,i3=0;
    float tmp=vec[0]+vec[1];
    best1=vec[0];
    if (vec[1]>best1)
      best1=vec[1];
    for (int i=2;i<nsamples;i++) {
      if (vec[i]>best1) {
        best1=vec[i];
        i1=i;
      }
      s1+=vec[i];
      v1+=vec[i]*vec[i];
      tmp+=vec[i];
      if (tmp>best3) {
        best3=tmp;
        i3=i;
      }
      s3+=tmp;
      v3+=tmp*tmp;
      tmp-=vec[i-2];
      if (tmp>best2) {
        best2=tmp;
        i2=i;
      }
      s2+=tmp;
      v2+=tmp*tmp;
    }
    s1/=nsamples-2;
    s2/=nsamples-2;
    s3/=nsamples-2;
    v1/=nsamples-2;
    v2/=nsamples-2;
    v3/=nsamples-2;
    v1=sqrt(v1-s1*s1);
    v2=sqrt(v2-s2*s2);
    v3=sqrt(v3-s3*s3);
    float snr1=(best1-s1)/v1;
    float snr2=(best2-s2)/v2;
    float snr3=(best3-s3)/v3;
    if (snr1>best.snr) {
      best.snr=snr1;
      best.peak=best1;
      best.ind=i1;
      best.depth=0;
      best.noise=v1;
      best.duration=1;
    }
    if (snr2>best.snr) {
      best.snr=snr2;
      best.peak=best2;
      best.ind=i2;
      best.depth=0;
      best.noise=v2;
      best.duration=2;
    }
    if (snr3>best.snr) {
      best.snr=snr3;
      best.peak=best3;
      best.ind=i3;
      best.depth=0;
      best.noise=v3;
      best.duration=3;
    }
    
    
    
  }
  
  
  
  Peak peak4,peak5,peak6,peak7;
  find_4567_peaks_wnoise(vec,nsamples,&peak4,&peak5,&peak6,&peak7);
  
  //peak4=peak4/sqrt(4*wt);                                                                                                               
  //peak5=peak5/sqrt(5*wt);                                                                                                               
  //peak6=peak6/sqrt(6*wt);                                                                                                               
  //peak7=peak7/sqrt(7*wt);                                                                                                               





  if (peak4.snr>best.snr)
    best=peak4;
  if (peak5.snr>best.snr)
    best=peak5;
  if (peak6.snr>best.snr)
    best=peak6;
  if (peak7.snr>best.snr)
    best=peak7;
  best.depth=cur_depth;
  //printf("peaks are %12.5g %12.5g %12.5g %12.5g\n",peak4,peak5,peak6,peak7);                                                            

  if (cur_depth<max_depth) {
    int nn=nsamples/2;
    float *vv=(float *)malloc(sizeof(float)*nn);
    for (int i=0;i<nn;i++)
      vv[i]=vec[2*i]+vec[2*i+1];
    Peak new_best=find_peaks_wnoise_onedm(vv,nn,max_depth,cur_depth+1);
    free(vv);
    if (new_best.snr>best.snr)
      best=new_best;
  }
  
  return best;
}

/*--------------------------------------------------------------------------------*/
Peak find_peak(Data *dat, int len_limit)
{
  //find the longest segment to be searched for
  //can't have a 5-sigma event w/out at least 25 samples to search over
  int max_len=dat->ndata/20;
  if ((len_limit > 0) && (len_limit < max_len))
      max_len=len_limit;
  int max_seg=max_len/7;
  int max_depth=log2(max_seg);
  //printf("max_depth is %d from %d\n",max_depth,dat->ndata);
  Peak best;
  best.snr=0;
  if (max_depth<1)
    return;
#pragma omp parallel
  {
    Peak mybest;
    mybest.snr=0;
#pragma omp for
    for (int i=0;i<dat->nchan;i++) {
      Peak dm_best=find_peaks_wnoise_onedm(dat->data[i],dat->ndata-dat->nchan,max_depth,0);
      if (dm_best.snr>mybest.snr) {
        mybest=dm_best;
        mybest.dm_channel=i;
      }
    }
#pragma omp critical
    {
      if (mybest.snr>best.snr)
        best=mybest;
    }
  }
  return best;
}
/*--------------------------------------------------------------------------------*/
size_t find_peak_wrapper(float *data, int nchan, int ndata, int len_limit, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)
{
  Data dat;
  float **mat=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nchan;i++)
    mat[i]=data+i*ndata;
  dat.data=mat;
  dat.ndata=ndata;
  dat.nchan=nchan;
  Peak best=find_peak(&dat, len_limit);
  free(dat.data); //get rid of the pointer array
  *peak_snr=best.snr;
  *peak_channel=best.dm_channel;
  //starting sample of the burst
  *peak_sample=best.ind*(1<<best.depth);
  *peak_duration=best.duration*(1<<best.depth);
  return 0;
  

}
/*--------------------------------------------------------------------------------*/
size_t my_burst_dm_transform(float *indata1, float *indata2, float *outdata,
                             size_t ntime1, size_t ntime2, float delta_t,
                             size_t nfreq, size_t *chan_map, int depth,int jon) 
{

  double t1=omp_get_wtime();

  //clean_rows_2pass(indata1,nfreq,ntime1);
  //if (ntime2>0)
  //  clean_rows_2pass(indata2,nfreq,ntime2);
  //printf("did row cleaning in %12.5f seconds.\n",omp_get_wtime()-t1);

  //t1=omp_get_wtime();
  Data *dat=put_data_into_burst_struct(indata1,indata2,ntime1,ntime2,nfreq,chan_map,depth);
  

  //setup_data does a bunch of cleaning, like removal of noise-cal, glitch finding, 
  //calibration off the noise cal, channel weighting...  If that has been done, just
  //call remap_data which will copy the data to where it needs to go.

  //setup_data(dat);
  remap_data(dat);

  if(jon == 1){
    dedisperse_gbt_jon(dat,outdata);
  }
  else{
    dedisperse_gbt(dat,outdata);
  }
  double t2=omp_get_wtime();
  //Peak best=find_peak(dat);

#if 0
  Peak best;
  find_peak_wrapper(dat->data[0],dat->nchan,dat->ndata,&best.snr,&best.dm_channel,&best.ind,&best.duration);
  double t3=omp_get_wtime();
  printf("times are %12.5f %12.5f, peak is %12.3f with channel %d at sample %d and duration %d\n",t2-t1,t3-t2,best.snr,best.dm_channel,best.ind,best.duration);
  
  int nskip=100;
  find_peak_wrapper(dat->data[nskip],dat->nchan-nskip,dat->ndata,&best.snr,&best.dm_channel,&best.ind,&best.duration);
  printf("times are %12.5f %12.5f, peak is %12.3f with channel %d at sample %d and duration %d\n",t2-t1,t3-t2,best.snr,best.dm_channel+nskip,best.ind,best.duration);
#endif

  size_t ngood=dat->ndata-dat->nchan;
  
  free(dat->raw_data[0]);
  free(dat->raw_data);
  free(dat->data[0]);
  free(dat->data);
  free(dat);

  //return ngood;
  return dat->ndata;

}



/*================================================================================*/




#if 0

int main(int argc, char *argv[])
{
  //printf("file has %d bytes.\n",(int)get_file_size("GBT11B_wigglez1hr_01_0123_dump.dat"));
  //int nchan,nsamp;

  Data *dat=read_gbt("GBT11B_wigglez1hr_01_0123_dump.dat");
  printf("nchan and nsamp are %d %d\n",dat->raw_nchan,dat->ndata);
  printf("top and bottom channels are %12.5f %12.5f\n",dat->raw_chans[0],dat->raw_chans[dat->raw_nchan-1]);


  double t1=omp_get_wtime();


  //This sets up the mapping of GBT channels into the linear-in-lambda^2 channels, and cleans up the data
  map_chans(dat,12);
  clean_rows(dat);  
  setup_data(dat);


  float **tmp=matrix(dat->nchan, dat->ndata);
  memset(tmp[0],0,sizeof(tmp[0][0])*dat->ndata*dat->nchan);
  
  dedisperse_gbt(dat,tmp[0]);
  printf("total processing took %12.4f seconds.\n",omp_get_wtime()-t1);



  FILE *rawfile=fopen("gbt_i.dat","w");
  fwrite(dat->raw_data[0],sizeof(dat->raw_data[0][0]),dat->ndata*dat->nchan,rawfile);
  fclose(rawfile);


  FILE *outfile=fopen("gbt_dedispersed.dat","w");
  fwrite(dat->data[0],sizeof(dat->data[0][0]),dat->nchan*dat->ndata,outfile);
  fclose(outfile);


  FILE *fid=fopen("crap.dat","w");
  for (int i=0;i<dat->raw_nchan;i++) {
    fwrite(&(dat->raw_chans[i]),1,sizeof(float),fid);
    fwrite(&(dat->chans[dat->chan_map[i]]),1,sizeof(float),fid);
  }
  fclose(fid);
  for (int i=dat->ndata-6000;i<dat->ndata-5994;i++) {
    printf("dat[1][i]=%12.4f\n",dat->data[1][i]);
  }
}
#endif
