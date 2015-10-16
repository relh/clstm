#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include "extras.h"

using std_string = std::string;
#define string std_string
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::make_pair;
using std::cout;
using std::stoi;
using namespace Eigen;
using namespace ocropus;

double randu() {
  static int count = 177;
  for (;;) {
    double x = cos(count * 3.7);
    count++;
    if (fabs(x) > 0.1) return x;
  }
}

void randseq(Sequence &a, int N, int n, int m) {
  a.resize(N, n, m);
  for (int t = 0; t < N; t++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a[t].v(i, j) = randu();
}
void constseq(Sequence &a, Scalar value, int N, int n, int m) {
  a.resize(N, n, m);
  for (int t = 0; t < N; t++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) a[t].v(i, j) = value;
}

int main(int argc, char **argv) {
  Sequence outputs, targets, aligned;
  randseq(targets, 7, 11, 1);
  randseq(outputs, 29, 11, 1);
  constseq(aligned, 0, 29, 11, 1);
  ctc_align_targets(aligned, outputs, targets);
}
