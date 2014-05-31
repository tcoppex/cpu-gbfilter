all: 
	g++ -O3 -fopenmp -msse4.1 -march=core2 -DGBF_OMP_STATS -Wno-unused-result -o gbfilter gbfilter.cc
