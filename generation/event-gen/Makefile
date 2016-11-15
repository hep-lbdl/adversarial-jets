# --------------------------------------------- #
# Makefile for Jet Images Framework             #
# Luke de Oliveira, April 3, 2015               # 
# lukedeo@stanford.edu                          # 
# --------------------------------------------- #

# set directories
BIN          := bin
SRC          := src
INC          := include
LIB          := $(CURDIR)/lib

NSUBDIR      := Nsubjettiness

# set search path
vpath %.o    $(BIN)
vpath %.cc   $(SRC) 
vpath %.h    $(INC) 

# set compiler and flags

CXX          ?= g++
CXXFLAGS     := -Wall -fPIC -I$(INC) -I$(NSUBDIR) -g -std=c++11 

ifeq ($(CXX),clang++)
CXXFLAGS += -stdlib=libc++
endif

# HEP flags
FASTJETFLAGS   := $(shell fastjet-config --cxxflags --plugins)
FASTJETLDFLAGS  =
FASTJETLIBS     = $(shell fastjet-config --libs --plugins)

ROOTFLAGS      := $(shell root-config --cflags)
ROOTLDFLAGS    := $(shell root-config --ldflags)
ROOTLIBS       := $(shell root-config --glibs)

PYTHIAFLAGS    := $(shell pythia8-config --cxxflags)
PYTHIALDFLAGS  := $(shell pythia8-config --ldflags)
PYTHIALIBS      = $(shell pythia8-config --libs)


HEPLIBS += $(FASTJETLIBS)
HEPLIBS += $(ROOTLIBS)
HEPLIBS += $(PYTHIALIBS)

# Add HEP flags to common stuff
CXXFLAGS += $(ROOTFLAGS)
CXXFLAGS += $(FASTJETFLAGS)
CXXFLAGS += $(PYTHIAFLAGS)

LIBS     += $(HEPLIBS)

LDFLAGS   = $(ROOTLDFLAGS) $(PYTHIALDFLAGS) $(FASTJETLDFLAGS)

# building excecutable
OBJ := Mediator.o JetImageBuffer.o JetImageProperties.o

EXECUTABLE := event-gen

EXTERNALS  := njettiness


all: $(EXTERNALS) $(EXECUTABLE)
	@echo "jet-images build sucessful."

njettiness:
	@echo "Building $@"
	@$(MAKE) -C $(NSUBDIR)

$(EXECUTABLE): $(OBJ:%=$(BIN)/%)
	@echo "linking $^ --> $@"
	@$(CXX) -o $@ $^ $(shell find ./Nsubjettiness/ | grep "\.o")  $(LDFLAGS) $(LIBS)


# auto dependency generation for build

# compile rule
$(BIN)/%.o: %.cc
	@echo compiling $<
	@mkdir -p $(BIN)
	@$(CXX) -c $(CXXFLAGS) $< -o $@

# use auto dependency generation
DEP = $(BIN)

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),rmdep)
ifneq ($(MAKECMDGOALS),purge)
include  $(OBJ:%.o=$(DEP)/%.d)
endif
endif
endif

DEPTARGSTR = -MT $(BIN)/$*.o -MT $(DEP)/$*.d
$(DEP)/%.d: %.cc
	@echo making dependencies for $<
	@mkdir -p $(DEP)
	@$(CXX) -MM -MP $(DEPTARGSTR) $(CXXFLAGS) $< -o $@ 

# clean
.PHONY : clean rmdep
CLEANLIST     = *~ *.o *.o~ *.d core 

clean:
	rm -fr $(CLEANLIST) $(CLEANLIST:%=$(BIN)/%) $(CLEANLIST:%=$(DEP)/%)
	rm -fr $(BIN) 
	@$(MAKE) $@ -C $(NSUBDIR)

purge:
	rm -fr $(CLEANLIST) $(CLEANLIST:%=$(BIN)/%) $(CLEANLIST:%=$(DEP)/%)
	rm -fr $(BIN) 
	rm -fr $(EXECUTABLE)
	@$(MAKE) $@ -C $(NSUBDIR)

rmdep: 
	rm -f $(DEP)/*.d
	@$(MAKE) $@ -C $(NSUBDIR)