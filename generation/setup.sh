#!/bin/bash

setup_GCC()
{
    source /cvmfs/atlas.cern.ch/repo/sw/atlas-gcc/491/x86_64/setup.sh
}


setup_PYTHIA() {
    #export PYTHIA8LOCATION=/u/at/pnef/Work/Code/pythia8183/
    export PYTHIA8LOCATION=/u/at/lukedeo/software/pythia8210
    export PYTHIA8DATA=${PYTHIA8LOCATION}/share/Pythia8/xmldoc/
    export LD_LIBRARY_PATH=${PYTHIA8LOCATION}/lib/:$LD_LIBRARY_PATH
}

setup_ROOT() {
    source /u/at/pnef/Work/Code/root_v5.34.17/bin/thisroot.sh
}

setup_fastjet() {
    export FASTJETLOCATION=/u/at/pnef/Work/Code/TrackBasedGrooming/stable/fastjet-3.0.3/fastjet-install/
    #export FASTJETLOCATION=/u/at/pnef/Work/Code/TrackBasedGrooming/fastjet-3.0.3/fastjet-install/
    #export FASTJETLOCATION=/u/at/pnef/Work/Code/fastjet-install/
    export LD_LIBRARY_PATH=${FASTJETPATH}lib/:$LD_LIBRARY_PATH
}

setup_atlint() {
    export PATH+=:$PYTHIA8LOCATION/bin
    export PATH+=:$FASTJETLOCATION/bin
}

setup_ROOT
setup_PYTHIA
setup_fastjet
setup_atlint