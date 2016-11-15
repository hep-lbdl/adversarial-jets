#ifndef JETIMAGEBUFFER_H
#define JETIMAGEBUFFER_H

#include <math.h>
#include <string>
#include <vector>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Filter.hh"

#include "TClonesArray.h"
#include "TFile.h"
#include "TParticle.h"
#include "TTree.h"

#include "FastJetConf.h"
#include "JetImageProperties.h"
#include "Pythia8/Pythia.h"

#include "TH2F.h"

using namespace std;
using namespace fastjet;

class JetImageBuffer {
  public:
    JetImageBuffer(int imagesize = 25);
    ~JetImageBuffer();

    void Begin();
    void AnalyzeEvent(int iEvt, Pythia8::Pythia *pythia8,
                      Pythia8::Pythia *pythia_MB, int NPV, int pixels,
                      float range);

    void End();
    void DeclareBranches();
    void ResetBranches();

    void Debug(int debug) {
        fDebug = debug;
    }

    void SetOutName(const string &outname) {
        fOutName = outname;
    }

  private:
    int ftest;
    int fDebug;
    string fOutName;

    TFile *tF;
    TTree *tT;
    JetImageProperties *tool;

    // Tree Vars ---------------------------------------
    int evt_number;
    int m_NPV;

    void SetupInt(int &val, TString name);
    void SetupFloat(float &val, TString name);

    vector<TString> names;
    vector<float> pts;
    vector<float> ms;
    vector<float> etas;
    vector<float> nsub21s;
    vector<float> nsub32s;
    vector<int> nsubs;

    TH2F *detector;

    int MaxN;

    int m_NFilled;

    float m_LeadingEta;
    float m_LeadingPhi;
    float m_LeadingPt;
    float m_LeadingM;

    float m_LeadingEta_nopix;
    float m_LeadingPhi_nopix;
    float m_LeadingPt_nopix;
    float m_LeadingM_nopix;

    float m_SubLeadingEta;
    float m_SubLeadingPhi;

    float m_PCEta;
    float m_PCPhi;
    
    float m_Tau1;
    float m_Tau2;
    float m_Tau3;

    float m_Tau21;
    float m_Tau32;

    float m_Tau1_nopix;
    float m_Tau2_nopix;
    float m_Tau3_nopix;

    float m_Tau21_nopix;
    float m_Tau32_nopix;

    float m_deltaR;
    float *m_Intensity;
};

#endif
