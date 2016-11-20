#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "TClonesArray.h"
#include "TDatabasePDG.h"
#include "TError.h"
#include "TParticle.h"
#include "TString.h"
#include "TSystem.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"

#include "Pythia8/Pythia.h"

#include "JetImageBuffer.h"
#include "JetImageProperties.h"

#include "parser.hh"

int getSeed(int seed) {
    if (seed > -1) {
        return seed;
    }
    int timeSeed = time(NULL);
    return abs(((timeSeed * 181) * ((getpid() - 83) * 359)) % 104729);
}

int main(int argc, const char *argv[]) {
    // argument parsing  ------------------------
    std::cout << "Called as: ";

    for (int ii = 0; ii < argc; ++ii) {
        std::cout << argv[ii] << " ";
    }
    std::cout << std::endl;

    // agruments
    std::string outName = "Mediator.root";
    int pileup = 0;
    int nEvents = 0;
    int pixels = 25;
    int fDebug = 0;
    float pThatmin = 100;
    float pThatmax = 500;
    float boson_mass = 1500;
    float image_range = 1.0;
    int proc = 1;
    int seed = -1;

    optionparser::parser parser("Allowed options");

    parser.add_option("--NEvents")
        .mode(optionparser::store_value)
        .default_value(10)
        .help("Number of Events");
    parser.add_option("--Pixels")
        .mode(optionparser::store_value)
        .default_value(25)
        .help("Number of pixels per dimension");
    parser.add_option("--Range")
        .mode(optionparser::store_value)
        .default_value(1)
        .help("Image captures [-w, w] x [-w, w], where w is the value passed.");
    parser.add_option("--Debug")
        .mode(optionparser::store_value)
        .default_value(0)
        .help("Debug flag");
    parser.add_option("--Pileup")
        .mode(optionparser::store_value)
        .default_value(0)
        .help("Number of Additional Interactions");
    parser.add_option("--OutFile")
        .mode(optionparser::store_value)
        .default_value("test.root")
        .help("output file name");
    parser.add_option("--Proc")
        .mode(optionparser::store_value)
        .default_value(2)
        .help("Process: 1=ZprimeTottbar, 2=WprimeToWZ_lept, 3=WprimeToWZ_had, "
              "4=QCD");
    parser.add_option("--Seed")
        .mode(optionparser::store_value)
        .default_value(-1)
        .help("seed. -1 means random seed");
    parser.add_option("--pThatMin")
        .mode(optionparser::store_value)
        .default_value(100)
        .help("pThatMin for QCD");
    parser.add_option("--pThatMax")
        .mode(optionparser::store_value)
        .default_value(500)
        .help("pThatMax for QCD");
    parser.add_option("--BosonMass")
        .mode(optionparser::store_value)
        .default_value(800)
        .help("Z' or W' mass in GeV");

    parser.eat_arguments(argc, argv);

    nEvents = parser.get_value<int>("NEvents");
    pixels = parser.get_value<int>("Pixels");
    image_range = parser.get_value<float>("Range");
    fDebug = parser.get_value<int>("Debug");
    pileup = parser.get_value<int>("Pileup");
    outName = parser.get_value<std::string>("OutFile");
    proc = parser.get_value<int>("Proc");
    seed = parser.get_value<int>("Seed");
    pThatmin = parser.get_value<float>("pThatMin");
    pThatmax = parser.get_value<float>("pThatMax");
    boson_mass = parser.get_value<float>("BosonMass");

    // seed
    seed = getSeed(seed);

    // Configure and initialize pythia
    Pythia8::Pythia *pythia8 = new Pythia8::Pythia();

    pythia8->readString("Random:setSeed = on");
    std::stringstream ss;
    ss << "Random:seed = " << seed;
    std::cout << ss.str() << std::endl;
    pythia8->readString(ss.str());

    pythia8->readString("Next:numberShowInfo = 0");
    pythia8->readString("Next:numberShowEvent = 0");
    pythia8->readString("Next:numberShowLHA = 0");
    pythia8->readString("Next:numberShowProcess = 0");

    if (proc == 1) {
        std::stringstream bosonmass_str;
        bosonmass_str << "32:m0=" << boson_mass;
        pythia8->readString(bosonmass_str.str());
        pythia8->readString("NewGaugeBoson:ffbar2gmZZprime= on");
        pythia8->readString("Zprime:gmZmode=3");
        pythia8->readString("32:onMode = off");
        pythia8->readString("32:onIfAny = 6");
        pythia8->readString("24:onMode = off");
        pythia8->readString("24:onIfAny = 1 2 3 4");
        pythia8->init();
    } else if (proc == 2) {
        std::stringstream bosonmass_str;
        bosonmass_str << "34:m0=" << boson_mass;
        pythia8->readString(bosonmass_str.str());
        pythia8->readString("NewGaugeBoson:ffbar2Wprime = on");
        pythia8->readString("Wprime:coup2WZ=1");
        pythia8->readString("34:onMode = off");
        pythia8->readString("34:onIfAny = 23 24");
        pythia8->readString("24:onMode = off");
        pythia8->readString("24:onIfAny = 1 2 3 4");
        pythia8->readString("23:onMode = off");
        pythia8->readString("23:onIfAny = 12");
        pythia8->init();
    } else if (proc == 3) {
        std::stringstream bosonmass_str;
        bosonmass_str << "34:m0=" << boson_mass;
        pythia8->readString(bosonmass_str.str());
        pythia8->readString("NewGaugeBoson:ffbar2Wprime = on");
        pythia8->readString("Wprime:coup2WZ=1");
        pythia8->readString("34:onMode = off");
        pythia8->readString("34:onIfAny = 23 24");
        pythia8->readString("24:onMode = off");
        pythia8->readString("24:onIfAny = 11 12");
        pythia8->readString("23:onMode = off");
        pythia8->readString("23:onIfAny = 1 2 3 4 5");
        pythia8->init();
    } else if (proc == 4) {
        pythia8->readString("HardQCD:all = on");
        std::stringstream ptHatMin;
        std::stringstream ptHatMax;
        ptHatMin << "PhaseSpace:pTHatMin  =" << pThatmin;
        ptHatMax << "PhaseSpace:pTHatMax  =" << pThatmax;
        pythia8->readString(ptHatMin.str());
        pythia8->readString(ptHatMax.str());
        pythia8->init();
    } else {
        throw std::invalid_argument("received invalid 'process'");
    }

    // Setup the pileup
    Pythia8::Pythia *pythia_MB = new Pythia8::Pythia();
    pythia_MB->readString("Random:setSeed = on");
    ss.clear();
    ss.str("");
    ss << "Random:seed = " << seed + 1;
    std::cout << ss.str() << std::endl;
    pythia_MB->readString(ss.str());
    pythia_MB->readString("SoftQCD:nonDiffractive = on");
    pythia_MB->readString("HardQCD:all = off");
    pythia_MB->readString("PhaseSpace:pTHatMin  = .1");
    pythia_MB->readString("PhaseSpace:pTHatMax  = 20000");
    pythia_MB->init();

    JetImageBuffer *analysis = new JetImageBuffer(pixels);
    analysis->SetOutName(outName);
    analysis->Begin();
    analysis->Debug(fDebug);

    std::cout << pileup << " is the number of pileu pevents " << std::endl;

    // Event loop
    for (int iev = 0; iev < nEvents; iev++) {
        if (iev % 1000 == 0) {
            std::cout << "Generating event number " << iev << std::endl;
        }
        analysis->AnalyzeEvent(iev, pythia8, pythia_MB, pileup, pixels,
                               image_range);
    }

    analysis->End();

    // that was it
    delete pythia8;
    delete analysis;

    return 0;
}
