#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "../nanofft.c"

#include <immintrin.h>

void sande_tukey_scalar(FLOAT *real_signal, FLOAT *imag_signal, const FLOAT *real_buffer, const FLOAT *imag_buffer, uint32_t N) {
    uint32_t shift = 0;
    for (uint32_t step = intmin(VEC_LEN, N); step > 1; step >>= 1) { // Required addition of SIMD secondary loop to reach reasonable performance levels
    uint32_t half_step = step >> 1;
        for (uint32_t i = 0; i < N; i += step) {
            for (uint32_t j = 0; j < half_step; j++) {
                FLOAT real_even = real_signal[i + j];
                FLOAT imag_even = imag_signal[i + j];
                FLOAT real_odd = real_signal[i + j + half_step];
                FLOAT imag_odd = imag_signal[i + j + half_step];

                // Butterfly operation
                real_signal[i + j] = real_even + real_odd;
                imag_signal[i + j] = imag_even + imag_odd;

                //if(step == 2){printf("%i ", i + j + half_step);}

                // Calculate (even - odd) * buffer
                FLOAT real_temp = real_even - real_odd;
                FLOAT imag_temp = imag_even - imag_odd;
                //printf("%.4ff, ",real_buffer[shift + j]);
                //printf("%.4ff, ",imag_buffer[shift + j]);
                //real_signal[i + j + half_step] = real_temp * real_buffer[shift + j] - imag_temp * imag_buffer[shift + j];
                real_signal[i + j + half_step] = real_temp * real_buffer[shift + j] - imag_temp * imag_buffer[shift + j];
                imag_signal[i + j + half_step] = real_temp * imag_buffer[shift + j] + imag_temp * real_buffer[shift + j];
            }
        }
        shift += half_step;
        //for (uint32_t j = 0; j < N; j+= 1) {printf("%.1f %.1f \t", real_signal[j], imag_signal[j]);} printf("\n"); //debug printf
    }
}



void sande_tukey_vector(FLOAT *real_signal, FLOAT *imag_signal, uint32_t N) {
    //uint32_t iteration_counter = 0;
    for (uint32_t i = 0; i < 3; i+= 1) { // Required addition of SIMD secondary loop to reach reasonable performance levels
        //printf("%i", i);
            for (uint32_t j = 0; j < N; j+= VEC_LEN * 2) {
                // Load data into VEC variables
                VEC real_even = LOAD_VEC(&real_signal[j]);
                VEC real_odd = LOAD_VEC(&real_signal[j + VEC_LEN]);
                PERM_VEC(&real_even, &real_odd, i);
                SHUFFLE_VEC(&real_even, &real_odd);
                VEC imag_even = LOAD_VEC(&imag_signal[j]);
                VEC imag_odd = LOAD_VEC(&imag_signal[j + VEC_LEN]);
                PERM_VEC(&imag_even, &imag_odd, i);
                SHUFFLE_VEC(&imag_even, &imag_odd);

                // Butterfly operation
                VEC real_output_even = ADD_VEC(real_even, real_odd);
                VEC imag_output_even = ADD_VEC(imag_even, imag_odd);

                // Calculate (even - odd) * buffer
                VEC real_temp = SUB_VEC(real_even, real_odd);
                VEC imag_temp = SUB_VEC(imag_even, imag_odd);

                VEC real_output_odd = SUB_VEC(MUL_VEC(real_temp, RTWIDDLES[i].vec), MUL_VEC(imag_temp, ITWIDDLES[i].vec));
                VEC imag_output_odd = ADD_VEC(MUL_VEC(real_temp, ITWIDDLES[i].vec), MUL_VEC(imag_temp, RTWIDDLES[i].vec));

                //resture vectors to original permutation for next iteration

                SHUFFLE_VEC(&real_output_even, &real_output_odd);
                INVPERM_VEC(&real_output_even, &real_output_odd, i);
                SHUFFLE_VEC(&imag_output_even, &imag_output_odd);
                INVPERM_VEC(&imag_output_even, &imag_output_odd, i); //fails at last iteration for some reason

                STORE_VEC(&real_signal[j], real_output_even);
                STORE_VEC(&imag_signal[j], imag_output_even);
                STORE_VEC(&real_signal[j + VEC_LEN], real_output_odd);
                STORE_VEC(&imag_signal[j + VEC_LEN], imag_output_odd);
                }
                //for (uint32_t j = 0; j < N; j+= 1) {printf("%.1f %.1f \t", real_signal[j], imag_signal[j]);} printf("\n"); //debug printf
    }
}


int main() {
    //int N = 64;
    //printf("ąąą");
    int N = 8;
    FLOAT t[N];
    FLOAT *real_signal = (FLOAT *) aligned_alloc(64, intmax(N * sizeof(FLOAT), sizeof(VEC) << 1));
    FLOAT *imag_signal = (FLOAT *) aligned_alloc(64, intmax(N * sizeof(FLOAT), sizeof(VEC) << 1));
    FLOAT *real_buffer = (FLOAT *) aligned_alloc(64, intmax(N * sizeof(FLOAT), sizeof(VEC) << 1));
    FLOAT *imag_buffer = (FLOAT *) aligned_alloc(64, intmax(N * sizeof(FLOAT), sizeof(VEC) << 1));

    for (int i = 0; i < N; i++) {
        t[i] = (FLOAT)i / N;
        real_signal[i] = sin(2 * M_PI * 3 * t[i]) + 0.5 * sin(2 * M_PI * 8 * t[i]) + 0.75 * sin(2 * M_PI * 20 * t[i]);
        imag_signal[i] = 0.0; // Initialize imaginary part to zero
    }

    generate_buffer(N, real_buffer , imag_buffer);
    //nanofft_execute(real_signal, imag_signal, real_buffer, imag_buffer, N);
    //sande_tukey_scalar(real_signal, imag_signal, real_buffer, imag_buffer, N);
    sande_tukey_vector(real_signal, imag_signal, N);
    bit_reverse_permutation(real_signal, imag_signal, N);

    printf("\n\nSande-Tukey FFT output:\n");
    for (int i = 0; i < N; i++) {printf("signal[%d] = %.5f + %.5fi\n", i, real_signal[i], imag_signal[i]);}

    free(real_signal); free(imag_signal);
    free(real_buffer); free(imag_buffer);

    return 0;
}
