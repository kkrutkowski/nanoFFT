#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "./nanofft.c"

int main() {
    int N = 64;
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
    nanofft_execute(real_signal, imag_signal, real_buffer, imag_buffer, N);

    printf("Sande-Tukey FFT output:\n");
    for (int i = 0; i < N; i++) {printf("signal[%d] = %.5f + %.5fi\n", i, real_signal[i], imag_signal[i]);}

    free(real_signal); free(imag_signal);
    free(real_buffer); free(imag_buffer);

    return 0;
}
