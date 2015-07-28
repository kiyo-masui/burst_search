


OBJECTS = dedisperse_gbt.o dedisperse.o
CC=gcc

SHLIB_LD =      gcc -shared
SHLIB_SUFFIX =  .so

DEFS = 
CFLAGS = -O3 -std=c99 -fPIC -fopenmp
LDFLAGS =       $(CFLAGS)
#CC =            mpicc

.c.o:
	$(CC) -c $(CFLAGS) $(DEFS) $<


libdedisperse${SHLIB_SUFFIX}: ${OBJECTS}
	${SHLIB_LD} ${LDFLAGS} -o $@ ${OBJECTS}
