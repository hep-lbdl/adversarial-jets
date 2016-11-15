#!/usr/bin/env python
from multiprocessing import Pool, cpu_count
from numpy import floor
import os
import sys
import argparse
from subprocess import Popen, PIPE, STDOUT

DEVNULL = open(os.devnull, 'wb')

def parallel_run(f, parms):
    pool = Pool()
    ret = pool.map(f, parms)
    pool.close(); pool.join()
    return ret

def determine_allocation(n_samples, n_cpus = -1):
    if n_cpus < 0:
        n_cpus = cpu_count() - 1
    if n_cpus > (cpu_count() - 1):
        n_cpus = cpu_count() - 1

    allocation = n_cpus * [0]

    if n_samples <= n_cpus:
        for i in xrange(0, n_samples):
            allocation[i] += 1
        return allocation

    per_cpu = int(floor(float(n_samples) / n_cpus))

    added = 0
    to_add = n_samples

    for i in xrange(n_cpus):
        if i == (n_cpus - 1):
            if added + to_add < n_samples:
                to_add = n_samples - added
                leftover_allocation = determine_allocation(to_add - per_cpu, n_cpus)
                for i in xrange(n_cpus):
                    allocation[i] += leftover_allocation[i]
                to_add = per_cpu
        else:
            to_add = min(per_cpu, n_samples - added)
        allocation[i] += to_add
        added += to_add
    return allocation


def generate_calls(n_events, n_cpus=-1, outfile='gen.root', process='WprimeToWZ_lept', pixels=25, imrange=1, pileup=0, pt_hat_min=100, pt_hat_max = 500, bosonmass=800):
    if n_cpus < 0:
        n_cpus = cpu_count() - 1
    if n_cpus > (cpu_count() - 1):
        n_cpus = cpu_count() - 1
    print 'Splitting event generation over {} CPUs'.format(n_cpus)
    events_per_core = zip(determine_allocation(n_events, n_cpus), range(n_cpus))
    
    def _filename_prepare(f):
        if f.find('.root') < 0:
            return f + '.root'
        return f

    lookup = {'ZprimeTottbar' : "1",
              'WprimeToWZ_lept'  : "2",
              'WprimeToWZ_had' : "3",
              'QCD' : "4"}

    if n_cpus == 1:
        _call = ['./event-gen/event-gen', 
                 '--OutFile', _filename_prepare(outfile), 
                 '--NEvents', str(n_events), 
                 '--Proc', lookup[process], 
                 '--Pixels', str(pixels), 
                 '--Range', str(imrange), 
                 '--Pileup', str(pileup), 
                 '--pThatMin', str(pt_hat_min), 
                 '--pThatMax', str(pt_hat_max),
                 '--BosonMass', str(bosonmass)]
        return [_call]


    def _generate_call(par):

        try:
            _ = lookup[process]
        except:
            raise ValueError('process can be one of ' + ', '.join(a for a in lookup.keys()))

        _call = ['./event-gen/event-gen', 
                 '--OutFile', _filename_prepare(outfile).replace('.root', '_cpu{}.root'.format(par[1])), 
                 '--NEvents', str(par[0]), 
                 '--Proc', lookup[process], 
                 '--Pixels', str(pixels), 
                 '--Range', str(imrange), 
                 '--Pileup', str(pileup), 
                 '--pThatMin', str(pt_hat_min), 
                 '--pThatMax', str(pt_hat_max),
                 '--BosonMass', str(bosonmass)]
        return _call

    return [_generate_call(c) for c in events_per_core]




def excecute_call(f):
    return Popen(f, stdin=PIPE, stderr=STDOUT)



# 'Can be one of ZprimeTottbar, WprimeToWZ_lept, WprimeToWZ_had, or QCD'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default='events.root')
    parser.add_argument('--nevents', type=int, default=1000)
    parser.add_argument('--ncpu', type=int, default=-1)
    parser.add_argument('--process', type=str, default='WprimeToWZ_lept', help = 'Can be one of ZprimeTottbar, WprimeToWZ_lept, WprimeToWZ_had, or QCD')
    parser.add_argument('--pixels', type=int, default=25)
    parser.add_argument('--range', type=float, default=1)
    parser.add_argument('--pileup', type=int, default=0)
    parser.add_argument('--pt_hat_min', type=float, default=100)
    parser.add_argument('--pt_hat_max', type=float, default=500)
    parser.add_argument('--bosonmass', type=float, default=800)
    # parser.add_argument('--verbose', type=float, default=800)
    args = parser.parse_args()

    calls = generate_calls(args.nevents, 
                           args.ncpu, 
                           args.outfile, 
                           args.process, 
                           args.pixels, 
                           args.range, 
                           args.pileup, 
                           args.pt_hat_min, 
                           args.pt_hat_max, 
                           args.bosonmass)

    _ = parallel_run(excecute_call, calls)



