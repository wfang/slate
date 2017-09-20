
ifeq ($(MAKECMDGOALS),mac)
	include mac.mk
else ifeq ($(MAKECMDGOALS),lin)
	include lin.mk
else ifeq ($(MAKECMDGOALS),power8)
	include power8.mk
else ifeq ($(MAKECMDGOALS),phi)
	include phi.mk
endif

mac lin power8 phi:
	$(CC) $(CFLAGS) -I$(MPI)/include -DMPI -c trace/trace.c -o trace/trace.o
	$(CPP) $(CCFLAGS) $(INC) app.cc trace/trace.o $(LIB) -o app

clean:
	rm -rf app trace_*.svg
