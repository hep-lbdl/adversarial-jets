//  Nsubjettiness Package
//  Questions/Comments?  jthaler@jthaler.net
//
//  Copyright (c) 2011-13
//  Jesse Thaler, Ken Van Tilburg, Christopher K. Vermilion, and TJ Wilkason
//
//  Run this example with:
//     ./example_basic_usage < ../data/single-event.dat
//
//  $Id: example_basic_usage.cc 704 2014-07-07 14:30:43Z jthaler $
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
void analyze(const vector<PseudoJet> &input_particles);

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

// Helper Function for Printing out Jet Information
void PrintJets(const vector<PseudoJet> &jets, TauComponents components,
               bool showTotal = true);

////////
//
//  Main Routine for Analysis
//
///////

void analyze(const vector<PseudoJet> &input_particles) {

    ////////
    //
    //  Start of analysis.  First find anti-kT jets, then find N-subjettiness
    //  values of those jets
    //
    ///////

    // Initial clustering with anti-kt algorithm
    JetAlgorithm algorithm = antikt_algorithm;
    double jet_rad = 1.00; // jet radius for anti-kt algorithm
    JetDefinition jetDef = JetDefinition(algorithm, jet_rad, E_scheme, Best);
    ClusterSequence clust_seq(input_particles, jetDef);
    vector<PseudoJet> antikt_jets = sorted_by_pt(clust_seq.inclusive_jets());

    for (int j = 0; j < 2; j++) { // Two hardest jets per event
        if (antikt_jets[j].perp() < 200)
            continue;

        vector<PseudoJet> jet_constituents =
            clust_seq.constituents(antikt_jets[j]);

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << "Analyzing Jet " << j + 1 << ":" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        ////////
        //
        //  Basic checks of tau values first
        //
        //  If you don't want to know the directions of the subjets,
        //  then you can use the simple function Nsubjettiness.
        //
        //  Recommended usage for Nsubjettiness:
        //  AxesMode:  kt_axes, wta_kt_axes, onepass_kt_axes, or
        //  onepass_wta_kt_axes
        //  MeasureMode:  unnormalized_measure
        //  beta with kt_axes: 2.0
        //  beta with wta_kt_axes: anything greater than 0.0 (particularly good
        //  for 1.0)
        //  beta with onepass_kt_axes or onepass_wta_kt_axes:  between 1.0 and
        //  3.0
        //
        ///////

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << "N-subjettiness with Unnormalized Measure (in GeV)" << endl;
        cout << "beta = 1.0:  One-pass Winner-Take-All kT Axes" << endl;
        cout << "beta = 2.0:  One-pass E-Scheme kT Axes" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        // Now loop through all options
        cout << setprecision(6) << right << fixed;

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << setw(15) << "beta" << setw(14) << "tau1" << setw(14) << "tau2"
             << setw(14) << "tau3" << setw(14) << "tau2/tau1" << setw(14)
             << "tau3/tau2" << endl;

        // Axes modes to try
        OnePass_WTA_KT_Axes axisMode1;
        OnePass_KT_Axes axisMode2;

        // Measure modes to try
        double beta1 = 1.0;
        double beta2 = 2.0;
        UnnormalizedMeasure measureSpec1(beta1);
        UnnormalizedMeasure measureSpec2(beta2);

        // define Nsubjettiness functions (beta = 1.0)
        Nsubjettiness nSub1_beta1(1, axisMode1, measureSpec1);
        // an alternative (but more prone to error) constructor is
        // nSub1_beta1(1,  axisMode1,Njettiness::unnormalized_measure,beta1);
        Nsubjettiness nSub2_beta1(2, axisMode1, measureSpec1);
        Nsubjettiness nSub3_beta1(3, axisMode1, measureSpec1);
        NsubjettinessRatio nSub21_beta1(2, 1, axisMode1, measureSpec1);
        NsubjettinessRatio nSub32_beta1(3, 2, axisMode1, measureSpec1);

        // define Nsubjettiness functions (beta = 2.0)
        Nsubjettiness nSub1_beta2(1, axisMode2, measureSpec2);
        Nsubjettiness nSub2_beta2(2, axisMode2, measureSpec2);
        Nsubjettiness nSub3_beta2(3, axisMode2, measureSpec2);
        NsubjettinessRatio nSub21_beta2(2, 1, axisMode2, measureSpec2);
        NsubjettinessRatio nSub32_beta2(3, 2, axisMode2, measureSpec2);

        // calculate Nsubjettiness values (beta = 1.0)
        double tau1_beta1 = nSub1_beta1(antikt_jets[j]);
        double tau2_beta1 = nSub2_beta1(antikt_jets[j]);
        double tau3_beta1 = nSub3_beta1(antikt_jets[j]);
        double tau21_beta1 = nSub21_beta1(antikt_jets[j]);
        double tau32_beta1 = nSub32_beta1(antikt_jets[j]);

        // calculate Nsubjettiness values (beta = 2.0)
        double tau1_beta2 = nSub1_beta2(antikt_jets[j]);
        double tau2_beta2 = nSub2_beta2(antikt_jets[j]);
        double tau3_beta2 = nSub3_beta2(antikt_jets[j]);
        double tau21_beta2 = nSub21_beta2(antikt_jets[j]);
        double tau32_beta2 = nSub32_beta2(antikt_jets[j]);

        // Output results (beta = 1.0)
        cout << setw(15) << beta1 << setw(14) << tau1_beta1 << setw(14)
             << tau2_beta1 << setw(14) << tau3_beta1 << setw(14) << tau21_beta1
             << setw(14) << tau32_beta1 << endl;

        // Output results (beta = 2.0)
        cout << setw(15) << beta2 << setw(14) << tau1_beta2 << setw(14)
             << tau2_beta2 << setw(14) << tau3_beta2 << setw(14) << tau21_beta2
             << setw(14) << tau32_beta2 << endl;

        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                "^^^^^^^^^^^^^^^^^^^^^^^"
             << endl;
        cout << "Subjets found using beta = 1.0 tau values" << endl;
        PrintJets(nSub1_beta1.currentSubjets(),
                  nSub1_beta1.currentTauComponents()); // these subjets have
                                                       // valid constituents
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                "- - - - - - - - - - - -"
             << endl;
        PrintJets(nSub2_beta1.currentSubjets(),
                  nSub2_beta1.currentTauComponents()); // these subjets have
                                                       // valid constituents
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                "- - - - - - - - - - - -"
             << endl;
        PrintJets(nSub3_beta1.currentSubjets(),
                  nSub3_beta1.currentTauComponents()); // these subjets have
                                                       // valid constituents
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                "^^^^^^^^^^^^^^^^^^^^^^^"
             << endl;
        cout << "Axes used for above beta = 1.0 tau values" << endl;

        PrintJets(nSub1_beta1.currentAxes(), nSub1_beta1.currentTauComponents(),
                  false);
        // PrintJets(nSub1_beta1.seedAxes());  // For one-pass minimization,
        // this would show starting seeds
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                "- - - - - - - - - - - -"
             << endl;
        PrintJets(nSub2_beta1.currentAxes(), nSub2_beta1.currentTauComponents(),
                  false);
        // PrintJets(nSub2_beta1.seedAxes());  // For one-pass minimization,
        // this would show starting seeds
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                "- - - - - - - - - - - -"
             << endl;
        PrintJets(nSub3_beta1.currentAxes(), nSub3_beta1.currentTauComponents(),
                  false);
        // PrintJets(nSub3_beta1.seedAxes());  // For one-pass minimization,
        // this would show starting seeds

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
    }
}

void PrintJets(const vector<PseudoJet> &jets, TauComponents components,
               bool showTotal) {

    string commentStr = "";

    // gets extras information
    if (jets.size() == 0)
        return;

    // For printing out component tau information
    vector<double> subTaus = components.jet_pieces();
    double totalTau = components.tau();

    bool useArea = jets[0].has_area();

    // define nice tauN header
    int N = jets.size();
    stringstream ss("");
    ss << "tau" << N;
    string tauName = ss.str();

    cout << fixed << right;

    cout << commentStr << setw(5) << "jet #"
         << "   " << setw(10) << "rap" << setw(10) << "phi" << setw(11) << "pt"
         << setw(11) << "m" << setw(11) << "e";
    if (jets[0].has_constituents())
        cout << setw(11) << "constit";
    cout << setw(13) << tauName;
    if (useArea)
        cout << setw(10) << "area";
    cout << endl;

    // print out individual jet information
    for (unsigned i = 0; i < jets.size(); i++) {
        cout << commentStr << setw(5) << i + 1 << "   " << setprecision(4)
             << setw(10) << jets[i].rap() << setprecision(4) << setw(10)
             << jets[i].phi() << setprecision(4) << setw(11) << jets[i].perp()
             << setprecision(4) << setw(11)
             << max(jets[i].m(),
                    0.0) // needed to fix -0.0 issue on some compilers.
             << setprecision(4) << setw(11) << jets[i].e();
        if (jets[i].has_constituents())
            cout << setprecision(4) << setw(11)
                 << jets[i].constituents().size();
        cout << setprecision(6) << setw(13) << max(subTaus[i], 0.0);
        if (useArea)
            cout << setprecision(4) << setw(10)
                 << (jets[i].has_area() ? jets[i].area() : 0.0);
        cout << endl;
    }

    // print out total jet
    if (showTotal) {
        fastjet::PseudoJet total = join(jets);

        cout << commentStr << setw(5) << "total"
             << "   " << setprecision(4) << setw(10) << total.rap()
             << setprecision(4) << setw(10) << total.phi() << setprecision(4)
             << setw(11) << total.perp() << setprecision(4) << setw(11)
             << max(total.m(),
                    0.0) // needed to fix -0.0 issue on some compilers.
             << setprecision(4) << setw(11) << total.e();
        if (jets[0].has_constituents())
            cout << setprecision(4) << setw(11) << total.constituents().size();
        cout << setprecision(6) << setw(13) << totalTau;
        if (useArea)
            cout << setprecision(4) << setw(10)
                 << (total.has_area() ? total.area() : 0.0);
        cout << endl;
    }
}
