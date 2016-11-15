//  Nsubjettiness Package
//  Questions/Comments?  jthaler@jthaler.net
//
//  Copyright (c) 2011-13
//  Jesse Thaler, Ken Van Tilburg, and Christopher K. Vermilion
//
//  Run this example with:
//     ./example_v1p0p3 < ../data/single-event.dat
//
//  $Id: example_v1p0p3.cc 704 2014-07-07 14:30:43Z jthaler $
//----------------------------------------------------------------------
// This file is part of FastJet contrib.
//
// It is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
//
// It is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this code. If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------

#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "Njettiness.hh"
#include "NjettinessPlugin.hh"
#include "Nsubjettiness.hh" // In external code, this should be fastjet/contrib/Nsubjettiness.hh
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/PseudoJet.hh"
#include <sstream>

using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

// forward declaration to make things clearer
void read_event(vector<PseudoJet> &event);
void PrintJets(const vector<PseudoJet> &jets);
void analyze(const vector<PseudoJet> &input_particles);

//----------------------------------------------------------------------
//
//  Note: This example file is here to test the previous v1.0.3 interface
//  to make sure it is backwards compatable
//
//----------------------------------------------------------------------

int main() {

    //----------------------------------------------------------
    // read in input particles
    vector<PseudoJet> event;
    read_event(event);
    cout << "# read an event with " << event.size() << " particles" << endl;

    //----------------------------------------------------------
    // illustrate how Nsubjettiness contrib works

    analyze(event);

    return 0;
}

// read in input particles
void read_event(vector<PseudoJet> &event) {
    string line;
    while (getline(cin, line)) {
        istringstream linestream(line);
        // take substrings to avoid problems when there are extra "pollution"
        // characters (e.g. line-feed).
        if (line.substr(0, 4) == "#END") {
            return;
        }
        if (line.substr(0, 1) == "#") {
            continue;
        }
        double px, py, pz, E;
        linestream >> px >> py >> pz >> E;
        PseudoJet particle(px, py, pz, E);

        // push event onto back of full_event vector
        event.push_back(particle);
    }
}

/// Helper function for output
void PrintJets(const vector<PseudoJet> &jets) {

    if (jets.size() == 0)
        return;
    const NjettinessExtras *extras = njettiness_extras(jets[0]);

    if (jets[0].has_area()) {
        if (extras == NULL) {
            printf("%5s %10s %10s %10s %10s %10s %10s\n", "jet #", "rapidity",
                   "phi", "pt", "m", "e", "area"); // label the columns
            for (unsigned int i = 0; i < jets.size(); i++) {
                printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", i,
                       jets[i].rap(), jets[i].phi(), jets[i].perp(),
                       jets[i].m(), jets[i].e(),
                       (jets[i].has_area() ? jets[i].area() : 0.0));
            }
        } else {
            fastjet::PseudoJet total(0, 0, 0, 0);
            printf("%5s %10s %10s %10s %10s %10s %10s %10s\n", "jet #",
                   "rapidity", "phi", "pt", "m", "e", "subTau",
                   "area"); // label the columns
            for (unsigned int i = 0; i < jets.size(); i++) {
                printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f %10.6f %10.3f\n",
                       i, jets[i].rap(), jets[i].phi(), jets[i].perp(),
                       jets[i].m(), jets[i].e(), extras->subTau(jets[i]),
                       (jets[i].has_area() ? jets[i].area() : 0.0));
                total += jets[i];
            }
            printf("%5s %10.3f %10.3f %10.3f %10.3f %10.3f %10.6f %10.3f\n",
                   "total", total.rap(), total.phi(), total.perp(), total.m(),
                   total.e(), extras->totalTau(),
                   (total.has_area() ? total.area() : 0.0));
        }
    } else {
        if (extras == NULL) {
            printf("%5s %10s %10s %10s %10s %10s\n", "jet #", "rapidity", "phi",
                   "pt", "m", "e"); // label the columns
            for (unsigned int i = 0; i < jets.size(); i++) {
                printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f\n", i,
                       jets[i].rap(), jets[i].phi(), jets[i].perp(),
                       jets[i].m(), jets[i].e());
            }
        } else {
            fastjet::PseudoJet total(0, 0, 0, 0);
            printf("%5s %10s %10s %10s %10s %10s %10s\n", "jet #", "rapidity",
                   "phi", "pt", "m", "e", "subTau"); // label the columns
            for (unsigned int i = 0; i < jets.size(); i++) {
                printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f %10.6f\n", i,
                       jets[i].rap(), jets[i].phi(), jets[i].perp(),
                       jets[i].m(), jets[i].e(), extras->subTau(jets[i]));
                total += jets[i];
            }
            printf("%5s %10.3f %10.3f %10.3f %10.3f %10.3f %10.6f\n", "total",
                   total.rap(), total.phi(), total.perp(), total.m(), total.e(),
                   extras->totalTau());
        }
    }
}

////////
//
//  Main Routine for Analysis
//
///////

void analyze(const vector<PseudoJet> &input_particles) {

    /////// N-subjettiness /////////////////////////////

    // Initial clustering with anti-kt algorithm
    JetAlgorithm algorithm = antikt_algorithm;
    double jet_rad = 1.00; // jet radius for anti-kt algorithm
    JetDefinition jetDef = JetDefinition(algorithm, jet_rad, E_scheme, Best);
    ClusterSequence clust_seq(input_particles, jetDef);
    vector<PseudoJet> antikt_jets = sorted_by_pt(clust_seq.inclusive_jets());

    // Defining Nsubjettiness parameters
    double beta = 1.0; // power for angular dependence, e.g. beta = 1 --> linear
                       // k-means, beta = 2 --> quadratic/classic k-means
    double R0 = 1.0;   // Characteristic jet radius for normalization
    double Rcut =
        1.0; // maximum R particles can be from axis to be included in jet

    for (int j = 0; j < 2; j++) { // Two hardest jets per event
        if (antikt_jets[j].perp() > 200) {
            vector<PseudoJet> jet_constituents =
                clust_seq.constituents(antikt_jets[j]);

            //
            // If you don't want subjets, you can use the simple functor
            // Nsubjettiness:
            // Recommended usage is Njettiness::onepass_kt_axes mode.
            //

            //
            // Note:  all instances of axes mode Njettiness::min_axes have been
            // commented out.  This method is not guarenteed to give a global
            // minimum, only a local minimum, and different choices of the
            // random
            // number seed can give different results.  For that reason,
            // Njettiness::onepass_kt_axes is the recommended usage.
            //

            // 1-subjettiness
            Nsubjettiness nSub1KT(1, Njettiness::kt_axes, beta, R0, Rcut);
            double tau1 = nSub1KT(antikt_jets[j]);
            // Nsubjettiness nSub1Min(1, Njettiness::min_axes, beta, R0, Rcut);
            // double tau1min = nSub1Min(antikt_jets[j]);
            Nsubjettiness nSub1OnePass(1, Njettiness::onepass_kt_axes, beta, R0,
                                       Rcut);
            double tau1onepass = nSub1OnePass(antikt_jets[j]);

            // 2-subjettiness
            Nsubjettiness nSub2KT(2, Njettiness::kt_axes, beta, R0, Rcut);
            double tau2 = nSub2KT(antikt_jets[j]);
            // Nsubjettiness nSub2Min(2, Njettiness::min_axes, beta, R0, Rcut);
            // double tau2min = nSub2Min(antikt_jets[j]);
            Nsubjettiness nSub2OnePass(2, Njettiness::onepass_kt_axes, beta, R0,
                                       Rcut);
            double tau2onepass = nSub2OnePass(antikt_jets[j]);

            // 3-subjettiness
            Nsubjettiness nSub3KT(3, Njettiness::kt_axes, beta, R0, Rcut);
            double tau3 = nSub3KT(antikt_jets[j]);
            // Nsubjettiness nSub3Min(3, Njettiness::min_axes, beta, R0, Rcut);
            // double tau3min = nSub3Min(antikt_jets[j]);
            Nsubjettiness nSub3OnePass(3, Njettiness::onepass_kt_axes, beta, R0,
                                       Rcut);
            double tau3onepass = nSub3OnePass(antikt_jets[j]);

            //
            // Or, if you want subjets, use the FastJet plugin on a jet's
            // constituents
            // Recommended usage is Njettiness::onepass_kt_axes mode.
            //

            NjettinessPlugin nsub_plugin1(1, Njettiness::kt_axes, 1.0, 1.0,
                                          1.0);
            JetDefinition nsub_jetDef1(&nsub_plugin1);
            ClusterSequence nsub_seq1(antikt_jets[j].constituents(),
                                      nsub_jetDef1);
            vector<PseudoJet> kt1jets = nsub_seq1.inclusive_jets();

            NjettinessPlugin nsub_plugin2(2, Njettiness::kt_axes, 1.0, 1.0,
                                          1.0);
            JetDefinition nsub_jetDef2(&nsub_plugin2);
            ClusterSequence nsub_seq2(antikt_jets[j].constituents(),
                                      nsub_jetDef2);
            vector<PseudoJet> kt2jets = nsub_seq2.inclusive_jets();

            NjettinessPlugin nsub_plugin3(3, Njettiness::kt_axes, 1.0, 1.0,
                                          1.0);
            JetDefinition nsub_jetDef3(&nsub_plugin3);
            ClusterSequence nsub_seq3(antikt_jets[j].constituents(),
                                      nsub_jetDef3);
            vector<PseudoJet> kt3jets = nsub_seq3.inclusive_jets();

            // NjettinessPlugin nsubMin_plugin1(1, Njettiness::min_axes, 1.0,
            // 1.0, 1.0);
            // JetDefinition nsubMin_jetDef1(&nsubMin_plugin1);
            // ClusterSequence nsubMin_seq1(antikt_jets[j].constituents(),
            // nsubMin_jetDef1);
            // vector<PseudoJet> min1jets = nsubMin_seq1.inclusive_jets();

            // NjettinessPlugin nsubMin_plugin2(2, Njettiness::min_axes, 1.0,
            // 1.0, 1.0);
            // JetDefinition nsubMin_jetDef2(&nsubMin_plugin2);
            // ClusterSequence nsubMin_seq2(antikt_jets[j].constituents(),
            // nsubMin_jetDef2);
            // vector<PseudoJet> min2jets = nsubMin_seq2.inclusive_jets();

            // NjettinessPlugin nsubMin_plugin3(3, Njettiness::min_axes, 1.0,
            // 1.0, 1.0);
            // JetDefinition nsubMin_jetDef3(&nsubMin_plugin3);
            // ClusterSequence nsubMin_seq3(antikt_jets[j].constituents(),
            // nsubMin_jetDef3);
            // vector<PseudoJet> min3jets = nsubMin_seq3.inclusive_jets();

            NjettinessPlugin nsubOnePass_plugin1(1, Njettiness::onepass_kt_axes,
                                                 1.0, 1.0, 1.0);
            JetDefinition nsubOnePass_jetDef1(&nsubOnePass_plugin1);
            ClusterSequence nsubOnePass_seq1(antikt_jets[j].constituents(),
                                             nsubOnePass_jetDef1);
            vector<PseudoJet> onepass1jets = nsubOnePass_seq1.inclusive_jets();

            NjettinessPlugin nsubOnePass_plugin2(2, Njettiness::onepass_kt_axes,
                                                 1.0, 1.0, 1.0);
            JetDefinition nsubOnePass_jetDef2(&nsubOnePass_plugin2);
            ClusterSequence nsubOnePass_seq2(antikt_jets[j].constituents(),
                                             nsubOnePass_jetDef2);
            vector<PseudoJet> onepass2jets = nsubOnePass_seq2.inclusive_jets();

            NjettinessPlugin nsubOnePass_plugin3(3, Njettiness::onepass_kt_axes,
                                                 1.0, 1.0, 1.0);
            JetDefinition nsubOnePass_jetDef3(&nsubOnePass_plugin3);
            ClusterSequence nsubOnePass_seq3(antikt_jets[j].constituents(),
                                             nsubOnePass_jetDef3);
            vector<PseudoJet> onepass3jets = nsubOnePass_seq3.inclusive_jets();

            printf("-----------------------------------------------------------"
                   "--------------------------");
            printf("\n");
            printf("-----------------------------------------------------------"
                   "--------------------------");
            printf("\n");
            cout << "Beta = " << beta << endl;
            cout << "kT Axes:" << endl;
            PrintJets(kt1jets);
            PrintJets(kt2jets);
            PrintJets(kt3jets);
            // cout << "Multi-Pass Minimization Axes:" << endl;
            // PrintJets(min1jets);
            // PrintJets(min2jets);
            // PrintJets(min3jets);
            cout << "One Pass Minimization Axes from kT" << endl;
            PrintJets(onepass1jets);
            PrintJets(onepass2jets);
            PrintJets(onepass3jets);
            printf("-----------------------------------------------------------"
                   "--------------------------");
            printf("\n");
            cout << "Beta = " << beta << setprecision(6) << endl;
            cout << "     kT: "
                 << "tau1: " << tau1 << "  tau2: " << tau2 << "  tau3: " << tau3
                 << "  tau2/tau1: " << tau2 / tau1
                 << "  tau3/tau2: " << tau3 / tau2 << endl;
            // cout << "    Min: " << "tau1: " << tau1min << "  tau2: " <<
            // tau2min << "  tau3: " << tau3min << "  tau2/tau1: " <<
            // tau2min/tau1min << "  tau3/tau2: " << tau3min/tau2min << endl;
            cout << "OnePass: "
                 << "tau1: " << tau1onepass << "  tau2: " << tau2onepass
                 << "  tau3: " << tau3onepass
                 << "  tau2/tau1: " << tau2onepass / tau1onepass
                 << "  tau3/tau2: " << tau3onepass / tau2onepass << endl;
            cout << endl;
            printf("-----------------------------------------------------------"
                   "--------------------------");
            printf("\n");
            printf("-----------------------------------------------------------"
                   "--------------------------");
            printf("\n");
        }
    }

    // Note:  Removed all from example file since none of the NjettinessPlugin
    // is strictly backwards compatible.

    //   ////////// N-jettiness as a jet algorithm ///////////////////////////
    //
    //   // WARNING:  This is extremely preliminary.  You should not use for
    //   //   physics studies without contacting the authors.
    //   // You can also find jets with Njettiness:
    //
    //   NjettinessPlugin njet_plugin(3, Njettiness::onepass_kt_axes, 1.0, 1.0,
    //   1.0);
    //   JetDefinition njet_jetDef(&njet_plugin);
    //   ClusterSequence njet_seq(input_particles, njet_jetDef);
    //   vector<PseudoJet> njet_jets = njet_seq.inclusive_jets();
    //
    //
    //   NjettinessPlugin geo_plugin(3, NsubGeometricParameters(1.0));
    //   JetDefinition geo_jetDef(&geo_plugin);
    //   ClusterSequence geo_seq(input_particles, geo_jetDef);
    //   vector<PseudoJet> geo_jets = geo_seq.inclusive_jets();
    //
    //   // The axes might point in a different direction than the jets
    //   // Using the NjettinessExtras pointer (ClusterSequence::Extras) to
    //   access that information
    //   vector<PseudoJet> njet_axes;
    //   const NjettinessExtras * extras = njettiness_extras(njet_seq);
    //   if (extras != NULL) {
    //      njet_axes = extras->axes();
    //   }
    //
    //   printf("-------------------------------------------------------------------------------------");
    //   printf("\n");
    //   cout << "Event-wide Jets from One-Pass Minimization (beta = 1.0)" <<
    //   endl;
    //   PrintJets(njet_jets);
    //   cout << "Event-wide Axis Location for Above Jets" << endl;
    //   PrintJets(njet_axes);
    //
    //   cout << "Event-wide Jets from Geometric Measure" << endl;
    //   PrintJets(geo_jets);
    //   printf("-------------------------------------------------------------------------------------");
    //   printf("\n");
    //
    //   // You can also find jet areas using this method (quite slow, though)
    //
    //   double ghost_maxrap = 5.0; // e.g. if particles go up to y=5
    //   AreaDefinition area_def(active_area_explicit_ghosts,
    //   GhostedAreaSpec(ghost_maxrap));
    //
    //   ClusterSequenceArea njet_seq_area(input_particles,
    //   njet_jetDef,area_def);
    //   vector<PseudoJet> njet_jets_area = njet_seq_area.inclusive_jets();
    //
    //   ClusterSequenceArea geo_seq_area(input_particles, geo_jetDef,area_def);
    //   vector<PseudoJet> geo_jets_area = geo_seq_area.inclusive_jets();
    //
    //   // The axes might point in a different direction than the jets
    //   // Using the NjettinessExtras pointer (ClusterSequence::Extras) to
    //   access that information
    //   vector<PseudoJet> njet_axes_area;
    //   const NjettinessExtras * extras_area =
    //   njettiness_extras(njet_seq_area);
    //   if (extras_area != NULL) {
    //      njet_axes_area = extras_area->axes();
    //   }
    //
    //   printf("-------------------------------------------------------------------------------------");
    //   printf("\n");
    //   cout << "Event-wide Jets from One-Pass Minimization (beta = 1.0) (with
    //   area information)" << endl;
    //   PrintJets(njet_jets_area);
    //   cout << "Event-wide Axis Location for Above Jets (with area
    //   information)" << endl;
    //   PrintJets(njet_axes_area);
    //   cout << "Event-wide Jets from Geometric Measure (with area
    //   information)" << endl;
    //   PrintJets(geo_jets_area);
    //   printf("-------------------------------------------------------------------------------------");
    //   printf("\n");
}
