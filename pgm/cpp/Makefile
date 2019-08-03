OS = $(shell uname -s)

CFLAGS = -O3 -std=c++11 -fPIC -I /usr/local/Cellar/eigen/3.3.7/include/eigen3/
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
	$(CLANG) $(CFLAGS) -c main.cc -o main.o
	# FIXME: need to test on linux, will just sharing dynamiclib to shared work?
	# original command was:
	#$(CLANG) -shared  -Wl,-soname,${LIB} -o ${LIB} main.o
	$(CLANG) -dynamiclib -o ${LIB} main.o

clean:
	rm -f *.bc *.o *.out ${LIB}
