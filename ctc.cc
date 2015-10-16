#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>
#include <limits>
#include "extras.h"

#ifndef MAXEXP
#define MAXEXP 30
#endif

#define SIGNAN numeric_limits<float>::signaling_NaN()

namespace ocropus {
using namespace std;
using Eigen::Ref;

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
  int n = ROWS(lmatch), m = COLS(lmatch);
  lr.resize(n, m);
  Vec v(m), w(m);
  assert(v.dimension(0)==m);
  assert(w.dimension(0)==m);
  v.setConstant(SIGNAN);
  w.setConstant(SIGNAN);
  for (int j = 0; j < m; j++) v(j) = skip * j;
  for (int i = 0; i < n; i++) {
    w(0) = skip * i;
    for(int k=1; k<m; k++) w(k) = v(k-1);
    for (int j = 0; j < m; j++) {
      Float cost = lmatch(i, j);
      Float same = log_mul(v(j), cost);
      Float next = log_mul(w(j), cost);
      v(j) = log_add(same, next);
    }
    for(int k=0; k<m; k++) lr(i, k) = v(k);
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
  int n = ROWS(lmatch), m = COLS(lmatch);
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
  //print("outputs", outputs.dimension(0), outputs.dimension(1));
  //print("targets", targets.dimension(0), targets.dimension(1));
  assert(COLS(outputs)==COLS(targets));
  double lo = 1e-5;
  int n1 = ROWS(outputs);
  int n2 = ROWS(targets);
  int nc = COLS(targets);

  // compute log probability of state matches
  Mat lmatch;
  lmatch.resize(n1, n2);
  lmatch.setConstant(NAN);
  for (int t1 = 0; t1 < n1; t1++) {
    Vec out = outputs.chip(t1, 0);
    out = out.cwiseMax(Scalar(lo));
    Vec total = out.sum();
    out = out / total(0);
    for (int t2 = 0; t2 < n2; t2++) {
      double total = 0.0;
      for(int k=0; k<nc; k++) total += outputs(t1,k) * targets(t2,k);
      lmatch(t1, t2) = Scalar(total);
    }
  }
  // compute unnormalized forward backward algorithm
  Mat both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  Vec m = both.maximum();
  Mat epath = (both - m(0)).unaryExpr([](Scalar x) {
    return Scalar(x<-30?exp(-30):x>30?exp(30):exp(x));
  });
  for (int j = 0; j < n2; j++) {
    Vec l = epath.chip(j,1).sum();
    Float scale = fmax(l(0), 1e-9);
    epath.chip(j,1) = epath.chip(j,1) / scale;
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
    aligned.chip(i,0) = aligned.chip(i,0) / Scalar(fmax(1e-9, l(0)));
  }

  posteriors = aligned;
}

void mat_of_sequence(Mat &a, Sequence &s) {
  int n = s.size();
  int m = s.rows();
  assert(s.cols()==1);
  a.resize(n, m);
  for(int t=0;t<n;t++)
    for(int i=0;i<m;i++)
      a(t,i) = s[t].v(i,0);
}
void sequence_of_mat(Sequence &s, Mat &a) {
  int n = a.dimension(0);
  int m = a.dimension(1);
  s.resize(n,m,1);
  for(int t=0;t<n;t++)
    for(int i=0;i<m;i++)
      s[t].v(i,0) = a(t,i);
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Sequence &targets) {
  //print("outputs", outputs.size(), outputs.rows(), outputs.cols());
  //print("targets", targets.size(), targets.rows(), targets.cols());
  Mat moutputs, mtargets;
  mat_of_sequence(moutputs, outputs);
  mat_of_sequence(mtargets, targets);
  Mat aligned;
  ctc_align_targets(aligned, moutputs, mtargets);
  sequence_of_mat(posteriors,aligned);
}

void mktargets(Mat &seq, Classes &transcript, int ndim) {
  int n = transcript.size();
  int n2 = 2 * n + 1;
  seq.resize(n2, ndim);
  for (int t = 0; t < n2; t++) {
    for (int i=0; i<ndim; i++) seq(t,i) = 0;
    if (t % 2 == 1)
      seq(t, transcript[(t - 1) / 2]) = 1;
    else
      seq(t,0) = 1;
  }
}
void mktargets(Sequence &seq, Classes &transcript, int ndim) {
  Mat targets;
  mktargets(targets, transcript, ndim);
  sequence_of_mat(seq, targets);
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs,
                       Classes &targets) {
  Mat moutputs, mtargets;
  mat_of_sequence(moutputs, outputs);
  mktargets(mtargets, targets, outputs.rows());
  Mat aligned;
  ctc_align_targets(aligned, moutputs, mtargets);
  sequence_of_mat(posteriors,aligned);
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
    Vec vec = outputs[t].v.chip(batch,1);
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
