
#define CM_DTYPE size_t
//#define CM_DTYPE int64_t
#define DTYPE float

size_t burst_get_num_dispersions(size_t nfreq, float freq0,float delta_f, int depth);
int burst_depth_for_max_dm(float max_dm, float delta_t, size_t nfreq, float freq0,float delta_f);
void burst_setup_channel_mapping(CM_DTYPE *chan_map, size_t nfreq, float freq0, float delta_f, int depth);
size_t burst_dm_transform(DTYPE *indata1, DTYPE *indata2, CM_DTYPE *chan_map, DTYPE *outdata,size_t ntime1, size_t ntime2, float delta_t, size_t nfreq, float freq0, float delta_f, int depth, int jon);



