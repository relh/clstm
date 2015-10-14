#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
using namespace std;
using Eigen::Ref;

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
  int n = ROWS(lmatch), m = COLS(lmatch);
  lr.resize(n, m);
  Vec v(m), w(m);
  for (int j = 0; j < m; j++) v(j) = skip * j;
  for (int i = 0; i < n; i++) {
    // w.segment(1, m - 1) = v.segment(0, m - 1);
    array<int,1> bl{1}, sl{m-1}, br{0}, sr{m-1};
    w.slice(bl, sl) = v.slice(br, sr);
    w(0) = skip * i;
    for (int j = 0; j < m; j++) {
      Float same = log_mul(v(j), lmatch(i, j));
      Float next = log_mul(w(j), lmatch(i, j));
      v(j) = log_add(same, next);
    }
    lr.chip(i, 0) = v;
  }
}

void reverse2(Mat &out, Mat &in) {
  int n = in.dimension(0);
  int m = in.dimension(1);
  out.resize(n,m);
  for(int i=0;i<n;i++) {
    for(int j=0;j<m;j++) {
      out(i,j) = in(n-i-1,m-j-1);
    }
  }
}

void forwardbackward(Mat &both, Mat &lmatch) {
  Mat lr;
  forward_algorithm(lr, lmatch);
  Mat rlmatch;
  reverse2(rlmatch, lmatch);
  Mat temp;
  forward_algorithm(temp, rlmatch);
  Mat rl;
  reverse2(rl, temp);
  both = lr + rl;
}

void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets) {
  double lo = 1e-5;
  int n1 = ROWS(outputs);
  int n2 = ROWS(targets);
  int nc = COLS(targets);

  // compute log probability of state matches
  Mat lmatch;
  lmatch.resize(n1, n2);
  for (int t1 = 0; t1 < n1; t1++) {
    Vec out = outputs.chip(t1, 0);
    out = out.cwiseMax(Scalar(lo));
    Vec total = out.sum();
    out = out / total(0);
    for (int t2 = 0; t2 < n2; t2++) {
      Vec target = targets.chip(t2,0);
      lmatch(t1, t2) = log(dot(out, targets));
    }
  }
  // compute unnormalized forward backward algorithm
  Mat both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  Float (*f)(Float) = limexp;
  Mat epath = (both - both.maximum()).unaryExpr(f);
  for (int j = 0; j < n2; j++) {
    Vec l = epath.chip(j,1).sum();
    epath.chip(j,1) /= l(0) == 0 ? 1e-9 : l(0);
  }

  // compute posterior probabilities for each class and normalize
  Mat aligned;
  aligned.resize(n1, nc);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) {
      double total = 0.0;
      for (int k = 0; k < n2; k++) {
        double value = epath(i, k) * targets(k, j);
        total += value;
      }
      aligned(i, j) = total;
    }
  }
  for (int i = 0; i < n1; i++) {
    Vec l = aligned.chip(i,0).sum();
    aligned.chip(i,0) /= fmax(1e-9, l(0));
  }

  posteriors = aligned;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets) {
  assert(COLS(outputs[0]) == 1);
  assert(COLS(targets[0]) == 1);
  int n1 = outputs.size();
  int n2 = targets.size();
  int nc = targets[0].size();
  Mat moutputs(n1, nc);
  Mat mtargets(n2, nc);
  for (int i = 0; i < n1; i++) moutputs.chip(i,0) = outputs[i].chip(0,1);
  for (int i = 0; i < n2; i++) mtargets.chip(i,0) = targets[i].chip(0,1);
  Mat aligned;
  ctc_align_targets(aligned, moutputs, mtargets);
  posteriors.resize(n1);
  for (int i = 0; i < n1; i++) {
    posteriors[i].resize(aligned.dimension(0), 1);
    posteriors[i].chip(0,1) = aligned.chip(i,0);
  }
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets) {
  int nclasses = outputs[0].size();
  Sequence stargets;
  stargets.resize(targets.size());
  for (int t = 0; t < stargets.size(); t++) {
    stargets[t].resize(nclasses, 1);
    stargets[t].setConstant(0);
    stargets[t](targets[t], 0) = 1.0;
  }
  ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
  seq.resize(2 * transcript.size() + 1);
  for (int t = 0; t < seq.size(); t++) {
    seq[t].setZero(ndim, 1);
    if (t % 2 == 1)
      seq[t](transcript[(t - 1) / 2]) = 1;
    else
      seq[t](0) = 1;
  }
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch,
                    vector<int> *locs) {
  cs.clear();
  if (locs) locs->clear();
  int N = outputs.size();
  int t = 0;
  float mv = 0;
  int mc = -1;
  int mt = -1;
  while (t < N) {
    Vec vec = outputs[t].chip(batch,1);
    int index = argmax(vec);
    float v = vec[index];
    if (index == 0) {
      // NB: there should be a 0 at the end anyway
      if (mc != -1 && mc != 0) {
        cs.push_back(mc);
        if (locs) locs->push_back(mt);
      }
      mv = 0;
      mc = -1;
      mt = -1;
      t++;
      continue;
    }
    if (v > mv) {
      mv = v;
      mc = index;
      mt = t;
    }
    t++;
  }
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch) {
  trivial_decode(cs, outputs, batch, nullptr);
}
}  // ocropus
