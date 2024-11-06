# nanoFFT
An attempt to write a minimalistic FFT library using Sande-Tukey and COBRA algorithms in pure C. Only the AVX2 + FMA single precision variant supports full vectorization. Partial vectorization is supported for AVX and FMA instruction sets in single and double precision.

The provided benchmarks compare performance with some of the popular FFT libraries for AMD Ryzen 4600h CPU with AVX2 and FMA sets enabled. Generally, the library tends to approximately match FFTW3f in measure mode for the scalar variant, however, its relative performance rapidly diminishes with increasing vector length.

![Bench_1](https://github.com/user-attachments/assets/bd50d30c-2846-46d9-9cbe-05c3429e5143)
![Bench_2](https://github.com/user-attachments/assets/db298a76-44bd-465b-a7de-1effb5366638)
