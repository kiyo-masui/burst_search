#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "preprocess.h"

void onepass_stat(DTYPE* data, size_t len, DTYPE* mean, DTYPE* stdev){
	double s2 = 0.0;
	double val;

	double mean_est = 0.0;
	for(int i = 0; i < len; i++){
		mean_est += data[i];
	}
	mean_est = mean_est/((double) len);

	for(int i = 0; i < len; i++){
		val = data[i] - mean_est;
		s2 += val*val;
	}
	*mean = (DTYPE) mean_est;
	*stdev = (DTYPE) sqrt(s2/(((double) len) - 1.0));
}

void full_algorithm(DTYPE* data, size_t nfreq, size_t ntime, int block, DTYPE sigma_cut){
	if(block == -1){
		block = ntime;
	}

	for(int ii = 0; ii < ntime; ii += block){
		#pragma omp parallel for
		for(int jj = 0; jj < nfreq; jj++){
			DTYPE mean, std;
			onepass_stat(data + ii + jj*ntime,block,&mean,&std);
			for(int kk = 0; kk < block; kk++){
				if(fabs(*(data + jj*ntime + ii + kk)) > std*sigma_cut){
					*(data + jj*ntime + ii + kk) = mean;
				}
			}
		}
	}
}

void remove_outliers_single(DTYPE* data, size_t ntime, DTYPE sigma_cut){
	DTYPE val;
	DTYPE mean, stdev;
	onepass_stat(data,ntime,&mean,&stdev);
	for(int j = 0; j < ntime; j++){
		val = data[j];
		if (abs(val - mean) > stdev*sigma_cut){
			data[j] = mean;
		}
	}
}

void remove_outliers_c(DTYPE* data, size_t nchan, size_t ntime, DTYPE sigma_cut){
	#pragma omp parallel for
	for(int i = 0; i < nchan; i++){
		DTYPE val;
		DTYPE mean, stdev;
		DTYPE* dat;

		dat = data + i*ntime;
		onepass_stat(dat,ntime,&mean,&stdev);
		for(int j = 0; j < ntime; j++){
			val = dat[j];
			if (abs(val - mean) > stdev*sigma_cut){
				dat[j] = mean;
			}
		}
	}
}