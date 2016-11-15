# `jet-simulations`

## Dependencies

* `numpy`, `matplotlib`, `rootpy`, `PyROOT`, and `scikit-image` for python.
* `fastjet` version >= 3.1.0
* `Pythia` version >= 8.1
* `ROOT`

## Building the framework

Typing `make -j` should do the trick on most systems if `fastjet-config`, `pythia8-config`, and `root-config` are all in your `$PATH`. If you're on the SLAC computers, run `./setup.sh` first. This generates the low level event generation script (contained in the `./event-gen` folder), which can be invoked as `./event-gen/event-gen`, if you really need to.

## Event generation.

To speed things up, there is a multicore wrapper around this generation process in `generateEvents.py`. Calling `python generateEvents.py --help` yields:

```
usage: generateEvents.py [-h] [--outfile OUTFILE] [--nevents NEVENTS]
                         [--ncpu NCPU] [--process PROCESS] [--pixels PIXELS]
                         [--range RANGE] [--pileup PILEUP]
                         [--pt_hat_min PT_HAT_MIN] [--pt_hat_max PT_HAT_MAX]
                         [--bosonmass BOSONMASS]

optional arguments:
  -h, --help            show this help message and exit
  --outfile OUTFILE
  --nevents NEVENTS
  --ncpu NCPU
  --process PROCESS     Can be one of ZprimeTottbar, WprimeToWZ_lept,
                        WprimeToWZ_had, or QCD
  --pixels PIXELS
  --range RANGE
  --pileup PILEUP
  --pt_hat_min PT_HAT_MIN
  --pt_hat_max PT_HAT_MAX
  --bosonmass BOSONMASS
```

So, let's say you wanted to generate 1000 jet images `(25, 25)`, `R = 1.0` from a QCD process over all CPUs. You would invoke

```bash
python generateEvents.py --outfile=events.root --nevents=1000 --process=QCD --pixels=25 --range=1.0
```

This will generate files `events_cpuN.root` for `N = 1, ..., N_cpus`.



## Image processing

Image processing is handled in the parent directory. 


```
me@mycomputer$ python jetconverter.py --help
usage: jetconverter.py [-h] [--verbose] [--signal SIGNAL] [--save SAVE]
                       [--plot PLOT]
                       [files [files ...]]

positional arguments:
  files            Files to pass in

optional arguments:
  -h, --help       show this help message and exit
  --verbose        Verbose output
  --signal SIGNAL  String to search for in filenames to indicate a signal file
  --save SAVE      Filename to write out the data.
  --plot PLOT      File prefix that will be part of plotting filenames.
```

An example invocation could look like `./jetconverter.py --signal=Wprime --save=data.npy ./data/*.root`.


