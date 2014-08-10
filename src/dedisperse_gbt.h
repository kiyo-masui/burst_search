
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
  int *chan_map;

  //int icur;  //useful if we want to collapse the data after dedispersing
} Data;
