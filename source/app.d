import std.array : array;
import std.range : iota, repeat, enumerate;
import std.conv : to;
import std.numeric : dotProduct;
import std.algorithm;
import std.stdio;

import mir.ndslice;

class SVM {
public:
  alias Real = double;

  static auto zeros(T=Real)(size_t n) pure nothrow {
    return repeat(0.to!T, n).array;
  }

  void fit(Real[] inputs, Real[] outputs, size_t nsamples, size_t ndim,
           Real penalty=1.0, Real tolerance=1e-3, int maxIter=1000, bool isLinear=true) {
    this.torlerance = torlerance;
    this.penalty = penalty;
    this.isLinear = isLinear;
    this.ndim = ndim;
    this.trainInputs = inputs;

    auto xs = inputs.sliced(nsamples, ndim);
    auto ys = outputs.sliced(nsamples);
    this.multipliers = zeros(nsamples);
    this.weights = zeros(nsamples);
    this.bias = 0.0;
    this.errors = zeros(nsamples);

    bool isAll = true;
    foreach (size_t n; iota(maxIter)) {
      size_t nchanged = 0;
      foreach (i, a; multipliers) {
        if (isAll || (torlerance < a && a < penalty - torlerance)) {
          nchanged += countUpdate(i);
        }
      }
      if (isAll) {
        isAll = false;
        if (nchanged == 0) break;
      } else if (nchanged == 0) {
        isAll = true;
      }
    }

    // get support vectors
    svIndices = [];
    foreach (i, a; multipliers) {
      if (a != 0.0) svIndices ~= [i];
    }
    // get a weight vector
    foreach (i; svIndices) {
      weights[i] = ys[i] * multipliers[i];
    }
  }

  Real decision_function(Real[] input) {
    assert(input.length == ndim);
    auto xs = trainInputs.sliced(weights.length, ndim);
    return svIndices.map!(i => weights[i] * dotProduct(xs[i], input)).sum - bias;
  }


private:
  Real penalty = 1.0;
  Real torlerance = 1e-3;
  bool isLinear = true;
  Real[] weights, multipliers, errors;
  Real bias;
  size_t[] svIndices;
  size_t ndim;
  Real[] trainInputs;

  size_t countUpdate(size_t i) {
    // TODO step SMO
    return 0;
  }
}

void main()
{
  auto svm = new SVM;
	writeln("Edit source/app.d to start your project.");
}
