#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#define DM0 4000.0
#define NOISE_PERIOD 64
#define SIG_THRESH 30.0

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
  double t0=omp_get_wtime();
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

void dedisperse_kernel(float **in, float **out, int n, int m)
{
  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<m;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=0;i<m-jj-2;i++) 
      out[jj+npair][i]=in[2*jj][i+jj]+in[2*jj+1][i+jj+1];
    
  }
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

void dedisperse_single(float **inin, float **outout, int nchan,int ndat)
{
  int npass=get_npass(nchan);
  //printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  for (int i=0;i<npass;i++) {    
#pragma omp parallel for
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
  float **tmp=(float **)malloc(sizeof(float *)*dat->nchan);
  for (int i=0;i<dat->nchan;i++)
    tmp[i]=outdata+i*dat->ndata;
  
  memset(tmp[0],0,sizeof(tmp[0][0])*dat->ndata*dat->nchan);
  double t1=omp_get_wtime();
  //dedisperse_dual(dat->data,tmp,dat->nchan,dat->ndata);  
  dedisperse_single(dat->data,tmp,dat->nchan,dat->ndata);  
  
  //printf("took %12.4f seconds to dedisperse.\n",omp_get_wtime()-t1);
  memcpy(outdata,dat->data[0],dat->nchan*dat->ndata*sizeof(outdata[0]));


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
/*--------------------------------------------------------------------------------*/
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
    //	   }
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


      //dat->raw_data[i][j]=indata1[i*ndata1+j];      
      dat->raw_data[i][j]=indata1[j*dat->raw_nchan+i];


      //dat->raw_data[i][j]=0;
    }
    for (int j=0;j<npad;j++) {
      //this line also changes depending on memory ordering of input data
      //dat->raw_data[i*dat->ndata+ndata1+j]=indata2[i*ndata2+j];


      //dat->raw_data[i][ndata1+j]=indata2[i*ndata2+j];
      dat->raw_data[i][ndata1+j]=indata2[i+j*dat->raw_nchan];
      
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
size_t my_burst_dm_transform(float *indata1, float *indata2, float *outdata,
			     size_t ntime1, size_t ntime2, float delta_t,
			     size_t nfreq, size_t *chan_map, int depth) 
{

  double t1=omp_get_wtime();
  Data *dat=put_data_into_burst_struct(indata1,indata2,ntime1,ntime2,nfreq,chan_map,depth);
  

  //setup_data does a bunch of cleaning, like removal of noise-cal, glitch finding, 
  //calibration off the noise cal, channel weighting...  If that has been done, just
  //call remap_data which will copy the data to where it needs to go.

  //setup_data(dat);
  remap_data(dat);

  dedisperse_gbt(dat,outdata);

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
