OS = $(shell uname -s)
#LIB=libpgm.so

#CFLAGS = -O3 -std=c++11 -Wall -fno-use-cxa-atexit -fPIC
ifeq (${OS}, Darwin)
  # OS X
	CLANG ?= clang++
	LIB = libpgm.dylib
else ifeq (${OS}, Linux)
  # Linux
 	CLANG ?= clang++
	LIB = libpgm.so
else
  $(error Unsupported platform: ${OS})
endif

${LIB}: main.cc
	$(CLANG) -std=c++11 -fPIC -c main.cc -o main.o
	$(CLANG) -shared  -Wl,-soname,${LIB} -o ${LIB} main.o

clean:
	rm -f *.bc *.o ${LIB}
