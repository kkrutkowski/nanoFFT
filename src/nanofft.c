#define FLOAT float

#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#include "./cobra.h"

// Helper function to check if N is a power of two
static inline bool is_power_of_two(int N) {
    return (N > 0) && ((N & (N - 1)) == 0);
}


static inline const complex FLOAT* generate_buffer(int N){
    complex FLOAT* buffer = (complex FLOAT*) aligned_alloc(64 ,N * sizeof(complex FLOAT));;
    int shift = 0;
    for (int step = N; step > 1; step >>= 1) { // Right bit shift for division by 2
        int half_step = step >> 1; // Right bit shift for division by 2
            for (int j = 0; j < half_step; j++) {
                buffer[shift + j] = cexp(-2.0 * I * M_PI * j / step);
                //printf("%i\t%i\n", N, shift+j);
            }
        shift += step >> 1;
    }
return buffer;}


static inline void sande_tukey_in_place(complex FLOAT *signal, const complex FLOAT *buffer, int N) {
    int shift = 0;
    for (int step = N; step > 1; step >>= 1) {
        int half_step = step >> 1;
        for (int i = 0; i < N; i += step) {
            for (int j = 0; j < half_step; j++) {
                complex FLOAT even = signal[i + j];
                complex FLOAT odd = signal[i + j + half_step];
                signal[i + j] = even + odd;
                signal[i + j + half_step] = (even - odd) * buffer[shift + j]; //* cexp(-2.0 * I * M_PI * j / step);
            }
        }
        shift += half_step;
    }
}

void sande_tukey_fft(complex FLOAT *signal, const complex FLOAT* buffer, int N) {
    if ((N & (N - 1)) != 0) {
        fprintf(stderr, "Signal length must be a power of 2\n");
        exit(EXIT_FAILURE);
    }
    sande_tukey_in_place(signal, buffer, N);
    bit_reverse_permutation(signal, N);
}
