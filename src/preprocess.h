#define DTYPE float

void remove_outliers_c(DTYPE * data, size_t nchan, size_t ntime, DTYPE sigma_cut);
void remove_outliers_single(DTYPE * data, size_t ntime, DTYPE sigma_cut);
void full_algorithm(DTYPE* data, size_t nfreq, size_t ntime, int block, DTYPE sigma_cut);
void onepass_stat(DTYPE* data, size_t len, DTYPE* mean, DTYPE* stdev);