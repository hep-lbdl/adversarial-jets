FROM phusion/baseimage:latest
MAINTAINER Luke de Oliveira <lukedeo@vaitech.io>

USER root

# update and install prerequisites for ROOT
RUN apt-get update && \
 	apt-get -y --force-yes install   \
	    bc                           \
        curl                         \
        git                          \
        wget                         \
		python-dev                   \
		python-pip                   \
		python-numpy                 \
		python-scipy                 \
		libx11-dev                   \
		libxpm-dev                   \
		libxft-dev                   \
		libxext-dev                  \
		libpng3                      \
		libjpeg8                     \
		gfortran                     \
		libssl-dev                   \
		libpcre3-dev                 \
		libgl1-mesa-dev              \
		libglew1.5-dev               \
		libftgl-dev                  \
		libmysqlclient-dev           \
		libfftw3-dev                 \
		libcfitsio3-dev              \
		graphviz-dev                 \
		libavahi-compat-libdnssd-dev \
		libldap2-dev                 \
		libxml2-dev  &&              \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
	cython \
	scikit-image


ENV MPLBACKEND=PDF

# all SW is installed under /opt
WORKDIR /opt

# unpack ROOT to /opt/troot
RUN wget -O root.tgz \
	https://root.cern.ch/download/root_v5.34.36.Linux-ubuntu14-x86_64-gcc4.8.tar.gz && \
	tar -xzf root.tgz 

ENV ROOTSYS=/opt/root
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTSYS/lib
ENV DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$ROOTSYS/lib
ENV PYTHONPATH=$PYTHONPATH:$ROOTSYS/lib
ENV DISPLAY=""

# ROOT specific python packages
RUN pip install --no-cache-dir \
	rootpy \
	root_numpy

# Install Pythia8
RUN wget http://home.thep.lu.se/~torbjorn/pythia8/pythia8219.tgz && \
	tar -xzf pythia8219.tgz
RUN cd pythia8219 && \
	./configure   && \
	make          && \
	make install  && \
	cd /opt
ENV PYTHIA_ROOT=/opt/pythia8219

# Install FastJet
RUN wget http://fastjet.fr/repo/fastjet-3.2.1.tar.gz && \
	tar -xzf fastjet-3.2.1.tar.gz
RUN cd fastjet-3.2.1 && \
	./configure      && \
	make             && \
	make install     && \
	cd /opt
ENV FASTJET_ROOT=/opt/fastjet-3.2.1

# Setup all system level env vars
ENV C_INCLUDE_PATH=FASTJET_ROOT/include:$PYTHIA_ROOT/include        
ENV CPLUS_INCLUDE_PATH=$FASTJET_ROOT/include:$PYTHIA_ROOT/include
ENV PATH=$PATH:$ROOTSYS/bin:$PYTHIA_ROOT/bin:$FASTJET_ROOT/bin

ADD generation $HOME/generation

WORKDIR $HOME









