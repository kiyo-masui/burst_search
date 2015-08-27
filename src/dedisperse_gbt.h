
typedef struct {
  //float dm_max;
  //float dm_offset;
  int nchan;
  int raw_nchan;
  int ndata;
  float *chans;  //delay in pixels is chan_map[i]*dm
  float *raw_chans; 
  float dt;

  float **raw_data; //remap to float so we can do things like clean up data without worrying about overflow
  float **data;
  size_t *chan_map;

  //int icur;  //useful if we want to collapse the data after dedispersing
} Data;

/*--------------------------------------------------------------------------------*/

typedef struct {
  float snr;
  float peak;
  int ind;
  int depth;
  float noise;
  int dm_channel;
  int duration;
} Peak;

/*--------------------------------------------------------------------------------*/



int get_nchan_from_depth(int depth);
float get_diagonal_dm_simple(float nu1, float nu2, float dt, int depth);
int *ivector(int n);
size_t get_burst_nextra(size_t ndata2, int depth);
Data *put_data_into_burst_struct(float *indata1, float *indata2, size_t ntime1, size_t ntime2, size_t nfreq, size_t *chan_map, int depth);
size_t my_burst_dm_transform(float *indata1, float *indata2, float *outdata,size_t ntime1, size_t ntime2, float delta_t,size_t nfreq, size_t *chan_map, int depth, int jon);
size_t find_peak_wrapper(float *data, int nchan, int ndata, int len_limit, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration);
void clean_rows(Data *dat);
void clean_rows_2pass(float *vec, size_t nchan, size_t ndata);
void setup_data(Data *dat);
void remove_noisecal(Data *dat, int period,int apply_calib);
Data *read_gbt(const char *fname);
