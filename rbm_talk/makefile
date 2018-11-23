all: metro
metro: metropolis.o
	g++ metropolis.o -o metro
metropolis.o: metropolis.C
	g++ -c metropolis.C -o metropolis.o
clean:
	rm -f *.o metro
