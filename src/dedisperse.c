/*
 * As far as I can tell there is only one parameter in Jon's code, a *depth*
 * parameter in `map_chans`.  *delta_t* and *max_dm* are only required to
 * interpret `depth`. I may be completely wrong on this. -km
 *
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>


// For allocating output buffer.
size_t burst_get_num_dispersions(size_t nfreq, float freq0,
                           float delta_f, int depth) {
    return 10;
}

// Return minimum *depth* parameter required to achieve given maximum DM.
int burst_depth_for_max_dm(float max_dm, float delta_t, size_t nfreq, 
                     float freq0, float delta_f) {
    return 9;
}

// *ntime2* is allowed to be 0.  Return number of valid dedispersed time samples,
// (always less than, and usually equal to, *ntime1*).  *delta_f* may be negative.
// Frequencies are in Hz.
size_t burst_dm_transform(float *indata1, float *indata2, float *outdata,
                  size_t ntime1, size_t ntime2, float delta_t,
                  size_t nfreq, float freq0, float delta_f, int depth) {
    return 8;
}
