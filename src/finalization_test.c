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

    //#undef VEC_LEN
    //#define VEC_LEN 128

static inline uint32_t intmax(uint32_t a, uint32_t b) {return (a > b) ? a : b;}

typedef union {__m256i m256i; uint32_t i[8];} m256i_union;
// Permutation keys
                    // Permutation keys
                    static const m256i_union permutations[3] __attribute__((aligned(64))) = {
                    {.i = {0, 1, 2, 3, 4, 5, 6, 7}},
                    {.i = {0, 1, 4, 5, 2, 3, 6, 7}},
                    {.i = {0, 2, 4, 6, 1, 3, 5, 7}}}; //doesn't invert itself
                    static const m256i_union inv_permutations[3] __attribute__((aligned(64))) = {
                    {.i = {0, 1, 2, 3, 4, 5, 6, 7}},
                    {.i = {0, 1, 4, 5, 2, 3, 6, 7}},
                    {.i = {0, 4, 1, 5, 2, 6, 3, 7}}};
                static inline void nanofft_mm256_shuffle(__m256 *a, __m256 *b){
                    __m256 tmp = _mm256_set_m128(_mm256_extractf128_ps(*b, 0), _mm256_extractf128_ps(*a, 0));
                    *b = _mm256_set_m128(_mm256_extractf128_ps(*b, 1), _mm256_extractf128_ps(*a, 1));
                    *a = tmp;}
                static inline void nanofft_mm256_perm(__m256 *a, __m256 *b, uint32_t idx){
                    *a = _mm256_permutevar8x32_ps(*a, permutations[idx].m256i);
                    *b = _mm256_permutevar8x32_ps(*b, permutations[idx].m256i);}
                static inline void nanofft_mm256_inv_perm(__m256 *a, __m256 *b, uint32_t idx){
                    *a = _mm256_permutevar8x32_ps(*a, inv_permutations[idx].m256i);
                    *b = _mm256_permutevar8x32_ps(*b, inv_permutations[idx].m256i);}

typedef union {__m256 vec; float f[8];} m256_union;
static const m256_union real_twiddles[3] __attribute__((aligned(64))) = {
                    {.f = {1.0f, M_SQRT1_2, 0.0f, -M_SQRT1_2, 1.0f, M_SQRT1_2, 0.0f, -M_SQRT1_2}},
                    {.f = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}},
                    {.f = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}}};
static const m256_union imag_twiddles[3] __attribute__((aligned(64))) = {
                    {.f = {0.0f, M_SQRT1_2, 1.0f, M_SQRT1_2, 0.0f, M_SQRT1_2, 1.0f, M_SQRT1_2}},
                    {.f = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}},
                    {.f = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}};

#define RTWIDDLES real_twiddles
#define ITWIDDLES imag_twiddles


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

                // Calculate (even - odd) * buffer
                FLOAT real_temp = real_even - real_odd;
                FLOAT imag_temp = imag_even - imag_odd;

                real_signal[i + j + half_step] = real_temp * real_buffer[shift + j] - imag_temp * imag_buffer[shift + j];
                imag_signal[i + j + half_step] = real_temp * imag_buffer[shift + j] + imag_temp * real_buffer[shift + j];
            }
        }
        shift += half_step;
        //for (uint32_t j = 0; j < N; j+= 1) {printf("%.1f %.1f \t", real_signal[j], imag_signal[j]);} printf("\n"); //debug printf
    }
}



void sande_tukey_vector(FLOAT *real_signal, FLOAT *imag_signal, uint32_t N) {
    for (uint32_t i = 0; i < 3; i+= 1) { // Required addition of SIMD secondary loop to reach reasonable performance levels
            for (uint32_t j = 0; j < N; j+= VEC_LEN * 2) {
                // Load data into VEC variables
                VEC real_even = LOAD_VEC(&real_signal[j]);
                VEC real_odd = LOAD_VEC(&real_signal[j + VEC_LEN]);
                nanofft_mm256_perm(&real_even, &real_odd, i);
                nanofft_mm256_shuffle(&real_even, &real_odd);
                VEC imag_even = LOAD_VEC(&imag_signal[j]);
                VEC imag_odd = LOAD_VEC(&imag_signal[j + VEC_LEN]);
                nanofft_mm256_perm(&imag_even, &imag_odd, i);
                nanofft_mm256_shuffle(&imag_even, &imag_odd);

                // Butterfly operation
                real_even = ADD_VEC(real_even, real_odd);
                imag_even = ADD_VEC(imag_even, imag_odd);

                // Calculate (even - odd) * buffer
                real_odd = SUB_VEC(real_even, real_odd);
                imag_odd = SUB_VEC(imag_even, imag_odd);

                real_odd = SUB_VEC(MUL_VEC(real_odd, RTWIDDLES[i].vec), MUL_VEC(imag_odd, ITWIDDLES[i].vec));
                imag_odd = ADD_VEC(MUL_VEC(real_odd, ITWIDDLES[i].vec), MUL_VEC(imag_odd, RTWIDDLES[i].vec));

                //resture vectors to original permutation for next iteration
                nanofft_mm256_shuffle(&real_even, &real_odd);
                nanofft_mm256_inv_perm(&real_even, &real_odd, i);
                STORE_VEC(&real_signal[j], real_even);
                STORE_VEC(&imag_signal[j], imag_even);

                nanofft_mm256_shuffle(&imag_even, &imag_odd);
                nanofft_mm256_inv_perm(&imag_even, &imag_odd, i);
                STORE_VEC(&real_signal[j + VEC_LEN], real_odd);
                STORE_VEC(&imag_signal[j + VEC_LEN], imag_odd);
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
    sande_tukey_scalar(real_signal, imag_signal, real_buffer, imag_buffer, N);
    //sande_tukey_vector(real_signal, imag_signal, N);
    bit_reverse_permutation(real_signal, imag_signal, N);

    printf("\n\nSande-Tukey FFT output:\n");
    for (int i = 0; i < N; i++) {printf("signal[%d] = %.5f + %.5fi\n", i, real_signal[i], imag_signal[i]);}

    free(real_signal); free(imag_signal);
    free(real_buffer); free(imag_buffer);

    return 0;
}
