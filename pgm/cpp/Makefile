OS = $(shell uname -s)

ifeq (${OS}, Darwin)
  # OS X
	CLANG ?= clang++
	LIB = libpgm.dylib
	CFLAGS = -O3 -std=c++11 -fPIC -I /usr/local/Cellar/eigen/3.3.7/include/eigen3/
	SHARED_FLAG = -dynamiclib
else ifeq (${OS}, Linux)
  # Linux
 	CLANG ?= clang++
	LIB = libpgm.so
	CFLAGS = -O3 -std=c++11 -fPIC -I /home/ubuntu/libs/
	SHARED_FLAG = -shared
else
  $(error Unsupported platform: ${OS})
endif

${LIB}: main.cc
	$(CLANG) $(CFLAGS) -c main.cc -o main.o
	$(CLANG) $(SHARED_FLAG) -o ${LIB} main.o

clean:
	rm -f *.bc *.o *.out ${LIB}
