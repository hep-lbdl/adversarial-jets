//  Nsubjettiness Package
//  Questions/Comments?  jthaler@jthaler.net
//
//  Copyright (c) 2011-13
//  Jesse Thaler, Ken Van Tilburg, Christopher K. Vermilion, and TJ Wilkason
//
// Run this example with
//     ./example_advanced_usage < ../data/single-event.dat
//
//  $Id: example_advanced_usage.cc 704 2014-07-07 14:30:43Z jthaler $
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

// Simple class to store Axes along with a name for display
class AxesStruct {

  private:
    // Shared Ptr so it handles memory management
    SharedPtr<AxesDefinition> _axes_def;

  public:
    AxesStruct(const AxesDefinition &axes_def) : _axes_def(axes_def.create()) {
    }

    // Need special copy constructor to make it possible to put in a std::vector
    AxesStruct(const AxesStruct &myStruct)
        : _axes_def(myStruct._axes_def->create()) {
    }

    const AxesDefinition &def() const {
        return *_axes_def;
    }
    string description() const {
        return _axes_def->description();
    }
    string short_description() const {
        return _axes_def->short_description();
    }
};

// Simple class to store Measures to make it easier to put in std::vector
class MeasureStruct {

  private:
    // Shared Ptr so it handles memory management
    SharedPtr<MeasureDefinition> _measure_def;

  public:
    MeasureStruct(const MeasureDefinition &measure_def)
        : _measure_def(measure_def.create()) {
    }

    // Need special copy constructor to make it possible to put in a std::vector
    MeasureStruct(const MeasureStruct &myStruct)
        : _measure_def(myStruct._measure_def->create()) {
    }

    const MeasureDefinition &def() const {
        return *_measure_def;
    }
    string description() const {
        return _measure_def->description();
    }
};

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
void PrintJets(const vector<PseudoJet> &jets, bool commentOut = false);

////////
//
//  Main Routine for Analysis
//
///////

void analyze(const vector<PseudoJet> &input_particles) {

    ////////
    //
    //  This code will check multiple axes/measure modes
    //  First thing we do is establish the various modes we will check
    //
    ///////

    // A list of all of the available axes modes
    vector<AxesStruct> _testAxes;
    _testAxes.push_back(KT_Axes());
    _testAxes.push_back(CA_Axes());
    _testAxes.push_back(AntiKT_Axes(0.2));
    _testAxes.push_back(WTA_KT_Axes());
    _testAxes.push_back(WTA_CA_Axes());
    _testAxes.push_back(OnePass_KT_Axes());
    _testAxes.push_back(OnePass_CA_Axes());
    _testAxes.push_back(OnePass_AntiKT_Axes(0.2));
    _testAxes.push_back(OnePass_WTA_KT_Axes());
    _testAxes.push_back(OnePass_WTA_CA_Axes());
    _testAxes.push_back(MultiPass_Axes(100));

    //
    // Note:  Njettiness::min_axes is not guarenteed to give a global
    // minimum, only a local minimum, and different choices of the random
    // number seed can give different results.  For that reason,
    // the one-pass minimization are recommended over min_axes.
    //

    // Getting a smaller list of recommended axes modes
    // These are the ones that are more likely to give sensible results (and are
    // all IRC safe)
    vector<AxesStruct> _testRecommendedAxes;
    _testRecommendedAxes.push_back(KT_Axes());
    _testRecommendedAxes.push_back(WTA_KT_Axes());
    _testRecommendedAxes.push_back(OnePass_KT_Axes());
    _testRecommendedAxes.push_back(OnePass_WTA_KT_Axes());

    // Getting some of the measure modes to test
    // (When applied to a single jet we won't test the cutoff measures,
    // since cutoffs aren't typically helpful when applied to single jets)
    // Note that we are calling measures by their MeasureDefinition
    vector<MeasureStruct> _testMeasures;
    _testMeasures.push_back(NormalizedMeasure(1.0, 1.0));
    _testMeasures.push_back(UnnormalizedMeasure(1.0));
    _testMeasures.push_back(NormalizedMeasure(2.0, 1.0));
    _testMeasures.push_back(UnnormalizedMeasure(2.0));
    _testMeasures.push_back(GeometricMeasure(2.0));

    // When doing Njettiness as a jet algorithm, want to test the cutoff
    // measures.
    // (Since they are not senisible without a cutoff)
    vector<MeasureStruct> _testCutoffMeasures;
    _testCutoffMeasures.push_back(UnnormalizedCutoffMeasure(1.0, 0.8));
    _testCutoffMeasures.push_back(UnnormalizedCutoffMeasure(2.0, 0.8));
    _testCutoffMeasures.push_back(GeometricCutoffMeasure(2.0, 0.8));

    /////// N-subjettiness /////////////////////////////

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

    // small number to show equivalence of doubles
    double epsilon = 0.0001;

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
        cout << "Outputting N-subjettiness Values" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        // Now loop through all options
        cout << setprecision(6) << right << fixed;
        for (unsigned iM = 0; iM < _testMeasures.size(); iM++) {

            cout << "----------------------------------------------------------"
                    "---------------------------"
                 << endl;
            cout << _testMeasures[iM].description() << ":" << endl;
            cout << "       AxisMode" << setw(14) << "tau1" << setw(14)
                 << "tau2" << setw(14) << "tau3" << setw(14) << "tau2/tau1"
                 << setw(14) << "tau3/tau2" << endl;

            for (unsigned iA = 0; iA < _testAxes.size(); iA++) {

                // This case doesn't work, so skip it.
                if (_testAxes[iA].def().givesRandomizedResults() &&
                    !_testMeasures[iM].def().supportsMultiPassMinimization())
                    continue;

                // define Nsubjettiness functions
                Nsubjettiness nSub1(1, _testAxes[iA].def(),
                                    _testMeasures[iM].def());
                Nsubjettiness nSub2(2, _testAxes[iA].def(),
                                    _testMeasures[iM].def());
                Nsubjettiness nSub3(3, _testAxes[iA].def(),
                                    _testMeasures[iM].def());
                NsubjettinessRatio nSub21(2, 1, _testAxes[iA].def(),
                                          _testMeasures[iM].def());
                NsubjettinessRatio nSub32(3, 2, _testAxes[iA].def(),
                                          _testMeasures[iM].def());
                // calculate Nsubjettiness values
                double tau1 = nSub1(antikt_jets[j]);
                double tau2 = nSub2(antikt_jets[j]);
                double tau3 = nSub3(antikt_jets[j]);
                double tau21 = nSub21(antikt_jets[j]);
                double tau32 = nSub32(antikt_jets[j]);

                // Make sure calculations are consistent
                if (!_testAxes[iA].def().givesRandomizedResults()) {
                    assert(abs(tau21 - tau2 / tau1) < epsilon);
                    assert(abs(tau32 - tau3 / tau2) < epsilon);
                }

                string axesName = _testAxes[iA].short_description();
                // comment out with # because MultiPass uses random number seed
                if (_testAxes[iA].def().givesRandomizedResults())
                    axesName = "#    " + axesName;

                // Output results:
                cout << std::right << setw(14) << axesName << ":" << setw(14)
                     << tau1 << setw(14) << tau2 << setw(14) << tau3 << setw(14)
                     << tau21 << setw(14) << tau32 << endl;
            }
        }

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << "Done Outputting N-subjettiness Values" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        ////////
        //
        //  Finding axes/jets found by N-subjettiness partitioning
        //
        //  This uses the NjettinessPlugin as a jet finder (in this case, acting
        //  on an individual jet)
        //  Recommended usage is same as above
        //
        ///////

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << "Outputting N-subjettiness Subjets" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;

        // Loop through all options, this time setting up jet finding
        cout << setprecision(6) << left << fixed;
        for (unsigned iM = 0; iM < _testMeasures.size(); iM++) {

            for (unsigned iA = 0; iA < _testRecommendedAxes.size(); iA++) {

                // This case doesn't work, so skip it.
                if (_testAxes[iA].def().givesRandomizedResults() &&
                    !_testMeasures[iM].def().supportsMultiPassMinimization())
                    continue;

                // define the plugins
                NjettinessPlugin nsub_plugin1(1, _testRecommendedAxes[iA].def(),
                                              _testMeasures[iM].def());
                NjettinessPlugin nsub_plugin2(2, _testRecommendedAxes[iA].def(),
                                              _testMeasures[iM].def());
                NjettinessPlugin nsub_plugin3(3, _testRecommendedAxes[iA].def(),
                                              _testMeasures[iM].def());

                // define the corresponding jet definitions
                JetDefinition nsub_jetDef1(&nsub_plugin1);
                JetDefinition nsub_jetDef2(&nsub_plugin2);
                JetDefinition nsub_jetDef3(&nsub_plugin3);

                // and the corresponding cluster sequences
                ClusterSequence nsub_seq1(antikt_jets[j].constituents(),
                                          nsub_jetDef1);
                ClusterSequence nsub_seq2(antikt_jets[j].constituents(),
                                          nsub_jetDef2);
                ClusterSequence nsub_seq3(antikt_jets[j].constituents(),
                                          nsub_jetDef3);

                vector<PseudoJet> jets1 = nsub_seq1.inclusive_jets();
                vector<PseudoJet> jets2 = nsub_seq2.inclusive_jets();
                vector<PseudoJet> jets3 = nsub_seq3.inclusive_jets();

                cout << "------------------------------------------------------"
                        "-------------------------------"
                     << endl;
                cout << _testMeasures[iM].description() << ":" << endl;
                cout << _testRecommendedAxes[iA].description() << ":" << endl;

                bool commentOut = false;
                if (_testAxes[iA].def().givesRandomizedResults())
                    commentOut = true; // have to comment out min_axes, because
                                       // it has random values

                // This helper function tries to find out if the jets have tau
                // information for printing
                PrintJets(jets1, commentOut);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(jets2, commentOut);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(jets3, commentOut);

                // Also try to find axes location (if njettiness_extras works)
                vector<PseudoJet> axes1;
                vector<PseudoJet> axes2;
                vector<PseudoJet> axes3;
                const NjettinessExtras *extras1 = njettiness_extras(nsub_seq1);
                const NjettinessExtras *extras2 = njettiness_extras(nsub_seq2);
                const NjettinessExtras *extras3 = njettiness_extras(nsub_seq3);

                axes1 = extras1->axes();
                axes2 = extras2->axes();
                axes3 = extras3->axes();

                cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                     << endl;
                cout << "Axes Used for Above Subjets" << endl;

                PrintJets(axes1, commentOut);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(axes2, commentOut);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(axes3, commentOut);
            }
        }

        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
        cout << "Done Outputting N-subjettiness Subjets" << endl;
        cout << "--------------------------------------------------------------"
                "-----------------------"
             << endl;
    }

    ////////// N-jettiness as a jet algorithm ///////////////////////////

    ////////
    //
    //  You can also find jets event-wide with Njettiness.
    //  In this case, Winner-Take-All axes are a must, since the other axes get
    //  trapped in local minima
    //
    //  Recommended usage of NjettinessPlugin (event-wide)
    //  AxesMode:  wta_kt_axes or onepass_wta_kt_axes
    //  MeasureMode:  unnormalized_measure
    //  beta with wta_kt_axes: anything greater than 0.0 (particularly good for
    //  1.0)
    //  beta with onepass_wta_kt_axes:  between 1.0 and 3.0
    //
    ///////

    cout << "------------------------------------------------------------------"
            "-------------------"
         << endl;
    cout << "Using N-jettiness as a Jet Algorithm" << endl;
    cout << "------------------------------------------------------------------"
            "-------------------"
         << endl;

    for (unsigned iM = 0; iM < _testCutoffMeasures.size(); iM++) {

        for (unsigned iA = 0; iA < _testRecommendedAxes.size(); iA++) {

            // define the plugins
            NjettinessPlugin njet_plugin2(2, _testRecommendedAxes[iA].def(),
                                          _testCutoffMeasures[iM].def());
            NjettinessPlugin njet_plugin3(3, _testRecommendedAxes[iA].def(),
                                          _testCutoffMeasures[iM].def());
            NjettinessPlugin njet_plugin4(4, _testRecommendedAxes[iA].def(),
                                          _testCutoffMeasures[iM].def());

            // and the jet definitions
            JetDefinition njet_jetDef2(&njet_plugin2);
            JetDefinition njet_jetDef3(&njet_plugin3);
            JetDefinition njet_jetDef4(&njet_plugin4);

            // and the cluster sequences
            ClusterSequence njet_seq2(input_particles, njet_jetDef2);
            ClusterSequence njet_seq3(input_particles, njet_jetDef3);
            ClusterSequence njet_seq4(input_particles, njet_jetDef4);

            // and associated extras for more information
            const NjettinessExtras *extras2 = njettiness_extras(njet_seq2);
            const NjettinessExtras *extras3 = njettiness_extras(njet_seq3);
            const NjettinessExtras *extras4 = njettiness_extras(njet_seq4);

            // and find the jets
            vector<PseudoJet> njet_jets2 = njet_seq2.inclusive_jets();
            vector<PseudoJet> njet_jets3 = njet_seq3.inclusive_jets();
            vector<PseudoJet> njet_jets4 = njet_seq4.inclusive_jets();

            // (alternative way to find the jets)
            // vector<PseudoJet> njet_jets2 = extras2->jets();
            // vector<PseudoJet> njet_jets3 = extras3->jets();
            // vector<PseudoJet> njet_jets4 = extras4->jets();

            cout << "----------------------------------------------------------"
                    "---------------------------"
                 << endl;
            cout << _testCutoffMeasures[iM].description() << ":" << endl;
            cout << _testRecommendedAxes[iA].description() << ":" << endl;

            PrintJets(njet_jets2);
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                    "- - - - - - - - - - - - - -"
                 << endl;
            PrintJets(njet_jets3);
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                    "- - - - - - - - - - - - - -"
                 << endl;
            PrintJets(njet_jets4);

            // The axes might point in a different direction than the jets
            // Using the NjettinessExtras pointer (ClusterSequence::Extras) to
            // access that information
            vector<PseudoJet> njet_axes2 = extras2->axes();
            vector<PseudoJet> njet_axes3 = extras3->axes();
            vector<PseudoJet> njet_axes4 = extras4->axes();

            cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                    "^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                 << endl;
            cout << "Axes Used for Above Jets" << endl;

            PrintJets(njet_axes2);
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                    "- - - - - - - - - - - - - -"
                 << endl;
            PrintJets(njet_axes3);
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
                    "- - - - - - - - - - - - - -"
                 << endl;
            PrintJets(njet_axes4);

            bool calculateArea = false;
            if (calculateArea) {
                cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                     << endl;
                cout << "Adding Area Information (quite slow)" << endl;

                double ghost_maxrap = 5.0; // e.g. if particles go up to y=5
                AreaDefinition area_def(active_area_explicit_ghosts,
                                        GhostedAreaSpec(ghost_maxrap));

                // Defining cluster sequences with area
                ClusterSequenceArea njet_seq_area2(input_particles,
                                                   njet_jetDef2, area_def);
                ClusterSequenceArea njet_seq_area3(input_particles,
                                                   njet_jetDef3, area_def);
                ClusterSequenceArea njet_seq_area4(input_particles,
                                                   njet_jetDef4, area_def);

                vector<PseudoJet> njet_jets_area2 =
                    njet_seq_area2.inclusive_jets();
                vector<PseudoJet> njet_jets_area3 =
                    njet_seq_area3.inclusive_jets();
                vector<PseudoJet> njet_jets_area4 =
                    njet_seq_area4.inclusive_jets();

                PrintJets(njet_jets_area2);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(njet_jets_area3);
                cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
                        "- - - - - - - - - - - - - - - -"
                     << endl;
                PrintJets(njet_jets_area4);
            }
        }
    }

    cout << "------------------------------------------------------------------"
            "-------------------"
         << endl;
    cout << "Done Using N-jettiness as a Jet Algorithm" << endl;
    cout << "------------------------------------------------------------------"
            "-------------------"
         << endl;
}

void PrintJets(const vector<PseudoJet> &jets, bool commentOut) {

    string commentStr = "";
    if (commentOut)
        commentStr = "#";

    // gets extras information
    if (jets.size() == 0)
        return;
    const NjettinessExtras *extras = njettiness_extras(jets[0]);

    bool useExtras = (extras != NULL);
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
    if (useExtras)
        cout << setw(14) << tauName;
    if (useArea)
        cout << setw(10) << "area";
    cout << endl;

    fastjet::PseudoJet total(0, 0, 0, 0);

    // print out individual jet information
    for (unsigned i = 0; i < jets.size(); i++) {
        cout << commentStr << setw(5) << i + 1 << "   " << setprecision(4)
             << setw(10) << jets[i].rap() << setprecision(4) << setw(10)
             << jets[i].phi() << setprecision(4) << setw(11) << jets[i].perp()
             << setprecision(4) << setw(11)
             << max(jets[i].m(),
                    0.0) // needed to fix -0.0 issue on some compilers.
             << setprecision(4) << setw(11) << jets[i].e();
        if (useExtras)
            cout << setprecision(6) << setw(14)
                 << max(extras->subTau(jets[i]), 0.0);
        if (useArea)
            cout << setprecision(4) << setw(10)
                 << (jets[i].has_area() ? jets[i].area() : 0.0);
        cout << endl;
        total += jets[i];
    }

    // print out total jet
    if (useExtras) {
        double beamTau = extras->beamTau();

        if (beamTau > 0.0) {
            cout << commentStr << setw(5) << " beam"
                 << "   " << setw(10) << "" << setw(10) << "" << setw(11) << ""
                 << setw(11) << "" << setw(11) << "" << setw(14)
                 << setprecision(6) << beamTau << endl;
        }

        cout << commentStr << setw(5) << "total"
             << "   " << setprecision(4) << setw(10) << total.rap()
             << setprecision(4) << setw(10) << total.phi() << setprecision(4)
             << setw(11) << total.perp() << setprecision(4) << setw(11)
             << max(total.m(),
                    0.0) // needed to fix -0.0 issue on some compilers.
             << setprecision(4) << setw(11) << total.e() << setprecision(6)
             << setw(14) << extras->totalTau();
        if (useArea)
            cout << setprecision(4) << setw(10)
                 << (total.has_area() ? total.area() : 0.0);
        cout << endl;
    }
}
