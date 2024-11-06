all:
	gcc -Ofast -mavx2 -mfma -c ./nanofft.c -o nanofft
	gcc -Ofast -mavx2 -mfma -D DOUBLE -c ./nanofft.c -o nanofftd
	gcc -Ofast -mavx2 -mfma -D DOUBLE -c ./demo.c -o nanofft_demo
