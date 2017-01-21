import std.array : array;
import std.range : iota, repeat, enumerate, generate, take;
import std.conv : to;
import std.numeric : dotProduct;
import std.variant : Algebraic;
import std.algorithm : map, sum, filter, fold;
import std.math : fabs, sgn;
import std.stdio : writeln, writef;

import mir.ndslice : Slice, sliced;
import mir.random : unpredictableSeed, Random;
import mir.random.algorithm : range;
import mir.random.variable : Bernoulli2Variable, UniformVariable, NormalVariable;


class SVM {
  alias Real = double;
  alias kernel = dotProduct;

  static auto zeros(T=Real)(size_t n) pure nothrow {
    return repeat(0.to!T, n).array;
  }

  this(Slice!(2, Real*) inputs, Slice!(1, Real*) outputs, ulong seed=unpredictableSeed) {
    this(inputs.ptr, outputs.ptr, inputs.shape[0], inputs.shape[1], seed);
  }

  this(Real* inputs, Real* outputs, size_t nsamples, size_t ndim, ulong seed=unpredictableSeed) {
    this.rng = Random(seed);
    this.nsamples = nsamples;
    this.ndim = ndim;
    this.trainInputs = inputs[0 .. nsamples * ndim];
    this.trainOutputs = outputs[0 .. nsamples];
    this.multipliers = zeros(nsamples);
    this.weights = zeros(nsamples);
    this.bias = 0.0;
    this.errors = zeros(nsamples);
  }

  void fit(Real penalty=1.0, Real tolerance=1e-3, int maxIter=1000, bool isLinear=true) {
    this.multTolerance = multTolerance;
    this.penalty = penalty;
    this.isLinear = isLinear;

    bool isAll = true;
    foreach (_; iota(maxIter)) {
      size_t nchanged = 0;
      foreach (i, a; multipliers) {
        if (isAll || insidePenalty(a)) {
          if (update(i)) { ++nchanged; }
        }
      }
      if (isAll) {
        isAll = false;
        if (nchanged == 0) break;
      } else if (nchanged == 0) {
        isAll = true;
      }
      writef("%d,", nchanged);
    }

    // get support vectors
    svIndices = [];
    foreach (i, a; multipliers) {
      if (a != 0.0) svIndices ~= [i];
    }
    // get a weight vector
    foreach (i; svIndices) {
      weights[i] = trainOutputs[i] * multipliers[i];
    }
  }

  Real decision_function(Real* testInput) {
    auto t = testInput[0 .. ndim];
    return svIndices.map!(i => weights[i] * kernel(xs[i], t)).sum - bias;
  }

  // settings
  const size_t nsamples, ndim;
  const Real[] trainInputs, trainOutputs;
  Real penalty = 1.0;
  Real multTolerance = 1e-3;
  Real kktTolerance = 1e-3;
  bool isLinear = true;
  Random rng;

  // to be learned
  Real[] weights, multipliers, errors;
  Real bias;
  size_t[] svIndices;

  auto xs() {
    return this.trainInputs.sliced(nsamples, ndim);
  }

  bool insidePenalty(Real multiplier) {
    return multTolerance < multiplier && multiplier < (penalty - multTolerance);
  }

  bool needToOptimize(Real multiplier, Real yfi) {
    return // check KKT condition
      (multiplier < (penalty - multTolerance) && yfi < -kktTolerance) ||
      (multiplier > multTolerance             && yfi > kktTolerance);
  }

  auto f(const size_t i) {
    auto svs = iota(this.nsamples).filter!(j => (multipliers[j] == 0.0));
    return svs.map!(j => multipliers[j] * trainOutputs[j] * dotProduct(xs[j], xs[i])).sum - bias;
  }

  size_t randomIndex() {
    return UniformVariable!size_t(0, nsamples - 1)(rng);
  }

  auto randomIndices() {
    const off = randomIndex();
    return iota(nsamples).map!(i => (i + off) % nsamples);
  }

  alias MaybeIndex = Algebraic!(bool, size_t);

  auto randomSearchMaxErrorIndex(Real error) {
    Real maxdiff = -Real.infinity;
    MaybeIndex resultIndex;
    foreach(i; randomIndices) {
      const mi = multipliers[i];
      if (insidePenalty(mi)) {
        const diff = fabs(errors[i] - error);
        if (diff > maxdiff) {
          maxdiff = diff;
          resultIndex = i;
        }
      }
    }
    return resultIndex;
  }

  Real currentError;
  bool update(size_t i) {
    const mi = multipliers[i];
    currentError = insidePenalty(mi) ? errors[i] : f(i) - trainOutputs[i];
    const yfi = currentError * trainOutputs[i];
    if (needToOptimize(mi, yfi)) { return optimize(i); }
    return false;
  }

  bool optimize(size_t i) {
    const imax = randomSearchMaxErrorIndex(currentError);
    if (imax.hasValue && stepSMO(i, *imax.peek!size_t)) {
      return true;
    }
    foreach (j; randomIndices) {
      if (insidePenalty(multipliers[j]) && stepSMO(i, j)) {
        return true;
      }
    }
    foreach (j; randomIndices) {
      if (!insidePenalty(multipliers[j]) && stepSMO(i, j)) {
        return true;
      }
    }
    return false;
  }

  bool stepSMO(size_t i, size_t j) {
    // TODO: impl SMO
    if (i == j) { return false; }

    return true;
  }
}
