LIB=libpgm.so

${LIB}: main.cpp
	$(CXX) -std=c++11 -fPIC -c main.cpp -o main.o
	$(CXX) -shared  -Wl,-soname,${LIB} -o ${LIB} main.o

clean:
	rm -f *.bc *.o ${LIB}
