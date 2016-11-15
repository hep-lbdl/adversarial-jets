# --------------------------------------------- #
# Makefile for Adversarial Jets Framework       #
# Luke de Oliveira, 11/11/16                    # 
# lukedeo@vaitech.io                            # 
# --------------------------------------------- #


# -- set dirs
BUILD        := build
SRC          := src
INC          := include
OUTPUT       := bin
NSUBDIR      := Nsubjettiness
EXECUTABLE   := event-gen
EXTERNALS    := njettiness

# -- set search paths
vpath %.cc   $(SRC)
vpath %.hh   $(INC)
vpath %.h    $(INC)

# -- set compiler and flags
CXX          ?= g++
CXXFLAGS     := -O2 -Wall -fPIC -I$(INC) -I$(NSUBDIR) -std=c++11 -pedantic
CXXFLAGS     += -Wsign-compare
LIBS         :=
LDFLAGS      :=

# -- define objects from files in the SRC directory
GEN_OBJ_SRC   := $(wildcard $(SRC)/*.cc)
GEN_OBJ       := $(notdir $(GEN_OBJ_SRC:%.cc=%.o))

GEN_OBJ_PATHS := $(GEN_OBJ:%=$(BUILD)/%)

# -- all top level 
ALL_TOP_LEVEL := 


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

LIBS     += $(HEPLIBS) -lNsubjettiness # from local NSJ
LDFLAGS  += $(ROOTLDFLAGS) $(PYTHIALDFLAGS) $(FASTJETLDFLAGS)


# everything with this prefix will be built as an executable
EXE_PREFIX   := event-

ALL_EXE_SRC   := $(wildcard $(SRC)/$(EXE_PREFIX)*.cc)
ALL_EXE       := $(notdir $(ALL_EXE_SRC:%.cc=%))
ALL_EXE_PATHS := $(ALL_EXE:%=$(OUTPUT)/%)

# filter out the general objects
GEN_OBJ_PATHS := $(filter-out $(BUILD)/$(EXE_PREFIX)%.o,$(GEN_OBJ_PATHS))

# add to all top level
ALL_TOP_LEVEL += $(ALL_EXE_PATHS)

# -- first call here
all: $(EXTERNALS) $(ALL_TOP_LEVEL)

# build exe
$(OUTPUT)/$(EXE_PREFIX)%: $(GEN_OBJ_PATHS) $(BUILD)/$(EXE_PREFIX)%.o
	@mkdir -p $(OUTPUT)
	@echo "linking $^ --> $@"
	@$(CXX) -o $@ $^ -L$(NSUBDIR) $(LIBS) $(LDFLAGS)

njettiness:
	@echo "Building $@"
	@$(MAKE) -C $(NSUBDIR)

# compile rule
$(BUILD)/%.o: %.cc
	@echo compiling $<
	@mkdir -p $(BUILD)
	@$(CXX) -c $(CXXFLAGS) $< -o $@

# use auto dependency generation
ALLOBJ       := $(GEN_OBJ)
DEP          := $(BUILD)


DEPTARGSTR = -MT $(BUILD)/$*.o -MT $(DEP)/$*.d
$(DEP)/%.d: %.cc
	@echo making dependencies for $<
	@mkdir -p $(DEP)
	@$(CXX) -MM -MP $(DEPTARGSTR) $(CXXFLAGS) $(PY_FLAGS) $< -o $@

# clean
.PHONY : clean rmdep all
CLEANLIST     = *~ *.o *.o~ *.d core
clean:
	@$(MAKE) $@ -C $(NSUBDIR)
	rm -fr $(CLEANLIST) $(CLEANLIST:%=$(BUILD)/%) $(CLEANLIST:%=$(DEP)/%)
	rm -fr $(BUILD) $(DICT) $(OUTPUT) $(LIBDIR)

rmdep:
	@$(MAKE) $@ -C $(NSUBDIR)
	rm -f $(DEP)/*.d


