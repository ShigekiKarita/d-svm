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
    this.multTolerance = multTolerance;
    this.penalty = penalty;
    this.isLinear = isLinear;
    this.nsamples = nsamples;
    this.ndim = ndim;
    this.trainInputs = inputs;
    this.trainOutputs = outputs;

    this.multipliers = zeros(nsamples);
    this.weights = zeros(nsamples);
    this.bias = 0.0;
    this.errors = zeros(nsamples);

    bool isAll = true;
    foreach (_; iota(maxIter)) {
      size_t nchanged = 0;
      foreach (i, a; multipliers) {
        if (isAll || (multTolerance < a && a < penalty - multTolerance)) {
          nchanged += updateSMO(i);
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
      weights[i] = outputs[i] * multipliers[i];
    }
  }

  Real decision_function(Real[] input) {
    assert(input.length == ndim);
    auto xs = trainInputs.sliced(weights.length, ndim);
    return svIndices.map!(i => weights[i] * dotProduct(xs[i], input)).sum - bias;
  }


private:
  Real penalty = 1.0;
  Real multTolerance = 1e-3;
  Real kktTolerance = 1e-3;
  bool isLinear = true;
  Real[] weights, multipliers, errors;
  Real bias;
  size_t[] svIndices;
  size_t nsamples, ndim;
  Real[] trainInputs, trainOutputs;

  auto xs() {
    return this.trainInputs.sliced(nsamples, ndim);
  }

  bool isKKT(size_t i) {
    auto a = multipliers[i];
    auto e = (multTolerance < a && a < (penalty - multTolerance))
      ? errors[i]
      : f(i) - trainOutputs[i];
    
    auto yfi = e * trainOutputs[i];      // yf(x)-1
    return (a < (penalty - multTolerance) && yfi < -kktTolerance) || (a > multTolerance && yfi > kktTolerance);
  }

  double f(const size_t i) {
    auto svs = iota(trainInputs.length).filter!(j => multipliers[j] == 0.0);
    return svs.map!(j => multipliers[j] * trainOutputs[j] * dotProduct(xs[j], xs[i])).sum - bias;
  }

  size_t updateSMO(size_t i) {
    if (isKKT(i)) {
      // TODO: impl SMO
      return 0;
    }

    return 0;
  }
}

void main()
{
  auto svm = new SVM;
	writeln("Edit source/app.d to start your project.");
}
