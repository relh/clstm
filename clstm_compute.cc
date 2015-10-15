#include "clstm_compute.h"
#include <iostream>

namespace {
using namespace std;
inline void debug() { cerr << endl; }

template <class T>
inline void debug(const T &arg) {
  using namespace std;
  cerr << arg << endl;
}

template <class T, typename... Args>
inline void debug(T arg, Args... args) {
  cerr << arg << " ";
  debug(args...);
}
}

namespace ocropus {

#define MAPFUNC(X,F) (X).unaryExpr(F)

inline void ADDCOLS(Mat &m, Vec &v) {
//   for (int i = 0; i < COLS(m); i++)
//     for (int j = 0; j < ROWS(m); j++) m(j, i) += v(j);
}

void gradient_clip(Sequence &s, Float m) {
//   if (m < 0) return;
//   for (int t = 0; t < s.size(); t++) {
//     s[t].d =
//         MAPFUNC(s[t].d, [m](Float x) { return x > m ? m : x < -m ? -m : x; });
//   }
}

void gradient_clip(Mat &d, Float m) {
//   if (m < 0) return;
//   d = MAPFUNC(d, [m](Float x) { return x > m ? m : x < -m ? -m : x; });
}

void gradient_clip(Batch &b, Float m) { gradient_clip(b.d, m); }

inline array<IndexPair<int>,1> axes(int i,int j) {
  array<IndexPair<int>,1> result = {IndexPair<int>(i,j)};
  return result;
}
inline array<ptrdiff_t,1> A(int i) {
  return array<ptrdiff_t,1>({i});
}
inline array<ptrdiff_t,2> A(int i,int j) {
  return array<ptrdiff_t,2>({i,j});
}
inline Eigen::Sizes<1> S(int i) {
  return Eigen::Sizes<1>({i});
}
inline Eigen::Sizes<2> S(int i,int j) {
  return Eigen::Sizes<2>({i,j});
}

template <class F>
void forward_full(Batch &y, Params &W, Batch &x) {
   Float (*f)(Float) = F::nonlin;
   y.v = W.v.contract(x.v, axes(1,0)).unaryExpr(f);
}
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc) {
   Float (*g)(Float) = F::yderiv;
   Mat temp = y.v.unaryExpr(g) * y.d;
   x.d += W.v.contract(temp, axes(0,0));
   W.d += temp.contract(x.v, axes(1,1));
}

template void forward_full<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full<SigmoidNonlin>(Batch &y, Params &W, Batch &x,
                                           Float gc);
template void backward_full<TanhNonlin>(Batch &y, Params &W, Batch &x,
                                        Float gc);
template void backward_full<ReluNonlin>(Batch &y, Params &W, Batch &x,
                                        Float gc);

template <class F>
void forward_full1(Batch &y, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  Float (*f)(Float) = F::nonlin;
  y.v = (W1.v.slice(A(0,1),A(n,m-1)).contract(x.v,axes(1,0)) +
         W1.v.chip(0,1).reshape(A(n,1)).broadcast(A(1,bs)))
           .unaryExpr(f);
}

template <class F>
void backward_full1(Batch &y, Params &W1, Batch &x, Float gc) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  Mat W;
  W = W1.v.slice(A(0,1),A(n,m-1));
  auto d_W = W1.d.slice(A(0,1),A(n,m-1));
  auto d_w = W1.d.chip(0,1);
  Float (*g)(Float) = F::yderiv;
  Mat temp = y.d * y.v.unaryExpr(g);
  x.d = W.contract(temp,axes(0,0));
  d_W += temp.contract(x.v, axes(1,1));
  d_w += temp.sum(A(1));
}
template void forward_full1<NoNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x);
template void forward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x);
template void backward_full1<NoNonlin>(Batch &y, Params &W, Batch &x, Float gc);
template void backward_full1<SigmoidNonlin>(Batch &y, Params &W, Batch &x,
                                            Float gc);
template void backward_full1<TanhNonlin>(Batch &y, Params &W, Batch &x,
                                         Float gc);
template void backward_full1<ReluNonlin>(Batch &y, Params &W, Batch &x,
                                         Float gc);


void forward_softmax(Batch &z, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  int bs = x.v.dimension(1);
  Float (*f)(Float) = limexp;
  z.v = (W1.v.slice(A(0,1),A(n,m-1)).contract(x.v,axes(1,0)) +
         W1.v.chip(0,1).reshape(A(n,1)).broadcast(A(1,bs))).unaryExpr(f);
  for (int b = 0; b < COLS(z.v); b++) {
    auto v = z.v.chip(b, 1);
    Vec total = v.sum();
    Vec v1 = v / Scalar(fmax(total(0), 1e-9));
    v = v1;
  }
}

void backward_softmax(Batch &z, Params &W1, Batch &x) {
  int n = W1.v.dimension(0), m = W1.v.dimension(1);
  auto W = W1.v.slice(A(0,1),A(n,m-1));
  auto w = W1.v.chip(0,1);
  auto d_W = W1.d.slice(A(0,1),A(n,m-1));
  auto d_w = W1.d.chip(0,1);
  x.d = W.contract(z.d,axes(0,0));
  d_W += z.d.contract(x.v, axes(1,1));
  d_w += z.d.sum(A(1));
}

void forward_stack(Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  z.v.slice(A(0,0),A(nx,bs)) = x.v;
  z.v.slice(A(nx,0),A(ny,bs)) = y.v;
}
void backward_stack(Batch &z, Batch &x, Batch &y) {
  int nx = x.v.dimension(0), ny = y.v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d.slice(A(0,0),A(nx,bs));
  y.d += z.d.slice(A(nx,0),A(ny,bs));
}

void forward_stack(Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  z.v.slice(A(0,0),A(nx,bs)) = x.v;
  if(last>=0) z.v.slice(A(nx,0),A(ny,bs)) = y[last].v;
  else z.v.slice(A(nx,0),A(ny,bs)).setZero();
}
void backward_stack(Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d.slice(A(0,0),A(nx,bs));
  if(last>=0) y[last].d += z.d.slice(A(nx,0),A(ny,bs));
}

void forward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  //y.resize(N, x.rows(), x.cols());
  for (int i = 0; i < N; i++) y[N - i - 1] = x[i];
}
void backward_reverse(Sequence &y, Sequence &x) {
  int N = x.size();
  for (int i = 0; i < N; i++) x[N - i - 1].d += y[i].d;
}

void forward_stack1(Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  z.v.slice(A(0,0),A(1,bs)).setConstant(Scalar(1));
  z.v.slice(A(1,0),A(nx,bs)) = x.v;
  if(last>=0) z.v.slice(A(1+nx,0),A(ny,bs)) = y[last].v;
  else z.v.slice(A(nx,0),A(ny,bs)).setZero();
}
void backward_stack1(Batch &z, Batch &x, Sequence &y, int last) {
  int nx = x.v.dimension(0), ny = y[0].v.dimension(0);
  int bs = x.v.dimension(1);
  x.d += z.d.slice(A(1,0),A(nx,bs));
  if(last>=0) y[last].d += z.d.slice(A(1+nx,0),A(ny,bs));
}

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states, int last, Batch &gf) {
  state.v = ci.v * gi.v;
  if (last>=0) state.v += gf.v * states[last].v;
}
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states, int last, Batch &gf) {
  if (last>=0) states[last].d += state.d * gf.v;
  if (last>=0) gf.d += state.d * states[last].v;
  gi.d += state.d * ci.v;
  ci.d += state.d * gi.v;
}

template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  out.v = state.v.unaryExpr(f) * go.v;
}

template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go) {
  Float (*f)(Float) = H::nonlin;
  auto g = [](Float x) { return H::yderiv(H::nonlin(x)); };
  go.d += state.v.unaryExpr(f) * out.d;
  state.d += state.v.unaryExpr(g) * go.v * out.d;
}

template void forward_nonlingate<TanhNonlin>(Batch &out, Batch &state,
                                             Batch &go);
template void forward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state,
                                                Batch &go);
template void forward_nonlingate<NoNonlin>(Batch &out, Batch &state, Batch &go);
template void forward_nonlingate<ReluNonlin>(Batch &out, Batch &state,
                                             Batch &go);
template void backward_nonlingate<TanhNonlin>(Batch &out, Batch &state,
                                              Batch &go);
template void backward_nonlingate<SigmoidNonlin>(Batch &out, Batch &state,
                                                 Batch &go);
template void backward_nonlingate<NoNonlin>(Batch &out, Batch &state,
                                            Batch &go);
template void backward_nonlingate<ReluNonlin>(Batch &out, Batch &state,
                                              Batch &go);

void randunif(Mat &m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> randn;
  for (int i = 0; i < ROWS(m); i++)
    for (int j = 0; j < COLS(m); j++) m(i, j) = randn(gen);
}
void randunif(Vec &v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> randn;
  for (int i = 0; i < ROWS(v); i++) v(i) = randn(gen);
}
void randgauss(Mat &m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn;
  for (int i = 0; i < ROWS(m); i++)
    for (int j = 0; j < COLS(m); j++) m(i, j) = randn(gen);
}
void randgauss(Vec &v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn;
  for (int i = 0; i < ROWS(v); i++) v(i) = randn(gen);
}
void randinit(Mat &m, float s, const string mode) {
  if (mode == "unif") {
    m.setRandom();
    m = m * Scalar(2*s) - Scalar(s);
  } else if (mode == "pos") {
    m.setRandom();
    m = m * Scalar(s);
  } else if (mode == "normal") {
    randgauss(m);
    m = m * Scalar(s);
  }
}
void randinit(Vec &m, float s, const string mode) {
  if (mode == "unif") {
    m.setRandom();
    m = m * Scalar(2 * s) - Scalar(s);
  } else if (mode == "pos") {
    m.setRandom();
    m = m * Scalar(s);
  } else if (mode == "normal") {
    randgauss(m);
    m = m * Scalar(s);
  }
}
void randinit(Mat &m, int no, int ni, float s, const string mode) {
  m.resize(no, ni);
  randinit(m, s, mode);
}
void randinit(Vec &m, int no, float s, const string mode) {
  m.resize(no);
  randinit(m, s, mode);
}
void zeroinit(Mat &m, int no, int ni) {
  m.resize(no, ni);
  m.setZero();
}
void zeroinit(Vec &m, int no) {
  m.resize(no);
  m.setZero();
}

void resize(Sequence &seq, int nsteps, int dims, int bs) {
  seq.resize(nsteps);
  for (int i = 0; i < nsteps; i++) seq[i].resize(dims, bs);
}
int size(Sequence &seq, int dim) {
  if (dim == 0) return seq.size();
  if (dim == 1) return seq[0].rows();
  if (dim == 2) return seq[0].cols();
  THROW("bad dim ins size");
  return -1;
}

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

Vec timeslice(const Sequence &s, int i, int b) {
  Vec result(s.size());
  for (int t = 0; t < s.size(); t++) result[t] = s[t].v(i, b);
  return result;
}

bool anynan(Mat &a) {
  for (int j = 0; j < ROWS(a); j++) {
    for (int k = 0; k < COLS(a); k++) {
      float x = a(j, k);
      if (isnan(x)) return true;
    }
  }
}
bool anynan(Batch &b) {
  return anynan(b.v) || anynan(b.d);
}
bool anynan(Sequence &a) {
  for (int i = 0; i < a.size(); i++)
    if (anynan(a[i])) return true;
  return false;
}
}
