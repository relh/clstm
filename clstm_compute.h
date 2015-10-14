#ifndef clstm_compute__
#define clstm_compute__

#include <vector>
// #include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ocropus {
using namespace std;

#define ROWS(A) (A).dimension(0)
#define COLS(A) (A).dimension(1)
#define MAPFUN(M, F) ((M).unaryExpr(ptr_fun(F)))

#if 0
#ifdef LSTM_DOUBLE
typedef double Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
#else
typedef float Float;
typedef Eigen::VectorXi iVec;
typedef Eigen::VectorXf Vec;
typedef Eigen::MatrixXf Mat;
#endif
#else
#ifdef LSTM_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif
typedef Eigen::Tensor<Float, 2, Eigen::ColMajor> Mat;
typedef Eigen::Tensor<Float, 1, Eigen::ColMajor> Vec;
using Eigen::array;
typedef Mat::Scalar Scalar;
using Eigen::IndexPair;
inline int rows(const Mat &a) { return a.dimension(0); }
inline int cols(const Mat &a) { return a.dimension(1); }
#endif

inline Float dot(const Vec &u, const Vec &v) {
    assert(u.dimension(0)==v.dimension(0));
    double total = 0.0;
    for (int i=0; i<u.dimension(0); i++)
      total += u(i) * v(i);
    return total;
}

inline int argmax(const Vec &v) {
  int mi = 0;
  Float mv = v(0);
  for(int i=1;i<v.dimension(0);i++) {
    if(v(i)<=v(0)) continue;
    mi = i;
    mv = v(i);
  }
  return mi;
}

template <typename F, typename T>
void each(F f, T &a) {
  f(a);
}
template <typename F, typename T, typename... Args>
void each(F f, T &a, Args &&... args) {
  f(a);
  each(f, args...);
}

#ifndef MAXEXP
#define MAXEXP 30
#endif

inline Float tanh_(Float x) { return tanh(x); }
inline Float relu_(Float x) { return x <= 0 ? 0 : x; }
inline Float heavi_(Float x) { return x <= 0 ? 0 : 1; }

inline Float limexp(Float x) {
#if 1
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
#else
  return exp(x);
#endif
}

inline Float sigmoid(Float x) {
#if 1
  return 1.0 / (1.0 + limexp(-x));
#else
  return 1.0 / (1.0 + exp(-x));
#endif
}

inline Float log_add(Float x, Float y) {
  if (abs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline Float log_mul(Float x, Float y) { return x + y; }

template <class NONLIN, class T>
inline Mat nonlin(T &a) {
  Mat result = a;
  NONLIN::f(result);
  return result;
}
template <class NONLIN, class T>
inline Mat yprime(T &a) {
  Mat result;
  result.resize(ROWS(a), COLS(a));
  result.setConstant(1);
  NONLIN::df(result, a);
  return result;
}
template <class NONLIN, class T>
inline Mat xprime(T &a) {
  Mat result;
  result.resize(ROWS(a), COLS(a));
  result.setConstant(1);
  Mat temp = a;
  NONLIN::f(temp);
  NONLIN::df(result, temp);
  return result;
}

struct Batch {
  Mat v;
  Mat d;
  int rows() const { return v.dimension(0); }
  int cols() const { return v.dimension(1); }
  Mat &operator+() { return v; }
  void setZero() {
    v.setConstant(0);
    d.setConstant(0);
  }
  void setZero(int n, int m) {
    v.resize(n, m);
    d.resize(n, m);
    setZero();
  }
  void resize(int n, int m) {
    setZero(n, m);
  }
  void zeroGrad() { 
    d.resize(rows(), cols());
    d.setConstant(0);
  }
};
struct Params : Batch {
  void update(Float lr, Float mom) {
    v += d * Scalar(lr);
    d = d * Scalar(mom);
  }
};

// typedef vector<Mat> Sequence;
struct Sequence {
  vector<Batch> steps;
  Sequence() {}
  Sequence(int n) : steps(n) {}
  void clear() { steps.clear(); }
  int rows() const { return steps[0].rows(); }
  int cols() const { return steps[0].cols(); }
  void check() const {
    int N = steps.size();
    if (N == 0) return;
    assert(steps[0].rows() > 0);
    assert(steps[0].cols() > 0);
    for (int t = 0; t < N; t++) {
      assert(steps[t].rows() == steps[0].rows());
      assert(steps[t].cols() == steps[0].cols());
    }
  }
  int size() const { return steps.size(); }
  void resize(int n) { resize(n, 1, 1); }
  void resize(int n, int rows, int cols) {
    steps.resize(n);
    for (int t = 0; t < n; t++) {
      steps[t].setZero(rows, cols);
    }
  }
  void like(const Sequence &other) {
    resize(other.size(), other.rows(), other.cols());
  }
  void copy(const Sequence &other) {
    resize(other.size());
    for (int t = 0; t < other.size(); t++) steps[t] = other[t];
  }
  Batch &operator[](int i) { return steps[i]; }
  const Batch &operator[](int i) const { return steps[i]; }
  void zero() {
    for (int t = 0; t < steps.size(); t++) steps[t].setZero();
  }
  void zeroGrad() {
    for (int t = 0; t < steps.size(); t++) steps[t].zeroGrad();
  }
};

typedef vector<int> Classes;
typedef vector<Classes> BatchClasses;

void gradient_clip(Sequence &s, Float m = 100.0);
void gradient_clip(Batch &b, Float m = 100.0);
void gradient_clip(Mat &d, Float m = 100.0);

// FIXME: refactor into forward_/backward_
struct NoNonlin {
  static constexpr const char *kind = "Linear";
  static inline Float nonlin(Float x) { return x; }
  static inline Float yderiv(Float y) { return 1; }
  template <class T>
  static void f(T &x) {}
  template <class T, class U>
  static void df(T &dx, U &y) {}
};

struct SigmoidNonlin {
  static constexpr const char *kind = "Sigmoid";
  static inline Float nonlin(Float x) { return sigmoid(x); }
  static inline Float yderiv(Float y) { return y*(1-y); }
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, sigmoid);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= y.array() * (1 - y.array());
  }
};
struct TanhNonlin {
  static constexpr const char *kind = "Tanh";
  static inline Float nonlin(Float x) { return tanh(x); }
  static inline Float yderiv(Float y) { return 1-y*y; }
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, tanh_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= (1 - y.array().square());
  }
};
struct ReluNonlin {
  static constexpr const char *kind = "Relu";
  static inline Float nonlin(Float x) { return relu_(x); }
  static inline Float yderiv(Float y) { return heavi_(y); }
  template <class T>
  static void f(T &x) {
    x = MAPFUN(x, relu_);
  }
  template <class T, class U>
  static void df(T &dx, U &y) {
    dx.array() *= MAPFUN(y, heavi_).array();
  }
};

void forward_stack(Batch &z, Batch &x, Batch &y);
void backward_stack(Batch &z, Batch &x, Batch &y);
void forward_stack(Batch &z, Batch &x, Sequence &y, int last);
void backward_stack(Batch &z, Batch &x, Sequence &y, int last);

void forward_reverse(Sequence &y, Sequence &x);
void backward_reverse(Sequence &y, Sequence &x);

template <class F>
void forward_full1(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full1(Batch &y, Params &W, Batch &x, Float gc);

void forward_softmax(Batch &z, Params &W1, Batch &x);
void backward_softmax(Batch &z, Params &W1, Batch &x);
void forward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);
void backward_softmax(Sequence &outputs, Params &W1, Sequence &inputs);

void forward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                      int last, Batch &gf);
void backward_statemem(Batch &state, Batch &ci, Batch &gi, Sequence &states,
                       int last, Batch &gf);
template <class H>
void forward_nonlingate(Batch &out, Batch &state, Batch &go);
template <class H>
void backward_nonlingate(Batch &out, Batch &state, Batch &go);

// FIXME: replace these in LSTM; eliminate gradient_clip here
void forward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
void backward_stack1(Batch &all, Batch &inp, Sequence &out, int last);
template <class F>
void forward_full(Batch &y, Params &W, Batch &x);
template <class F>
void backward_full(Batch &y, Params &W, Batch &x, Float gc);

void randgauss(Mat &m);
void randgauss(Vec &v);
void randinit(Mat &m, float s, const string mode = "unif");
void randinit(Vec &m, float s, const string mode = "unif");
void randinit(Mat &m, int no, int ni, float s, const string mode = "unif");
void randinit(Vec &m, int no, float s, const string mode = "unif");
void zeroinit(Mat &m, int no, int ni);
void zeroinit(Vec &m, int no);
void resize(Sequence &seq, int nsteps, int dims, int bs);
int size(Sequence &seq, int dim);
Vec timeslice(const Sequence &s, int i, int b = 0);

bool anynan(Batch &a);
bool anynan(Sequence &a);
}

#endif
