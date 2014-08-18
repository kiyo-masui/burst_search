size_t burst_get_num_dispersions(size_t nfreq, float freq0,float delta_f, int depth);
int burst_depth_for_max_dm(float max_dm, size_t nfreq, float freq0,float delta_f, float delta_t);
void burst_setup_channel_mapping(int *chan_map, size_t nfreq, float freq0, float delta_f,int depth);
size_t burst_dm_transform(float *indata1, float *indata2, float *outdata,size_t nfreq, float freq0, float delta_f,size_t ntime1, size_t ntime2, float delta_t, int depth);



