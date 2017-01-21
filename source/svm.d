import std.array : array;
import std.range : repeat;
import std.conv : to;
import std.numeric : dotProduct;
import std.algorithm : map, sum;
import std.math : sgn;

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

  this(Real* inputs, Real* outputs, size_t nsamples, size_t ndim) {
    this.nsamples = nsamples;
    this.ndim = ndim;
    this.trainInputs = inputs[0 .. nsamples * ndim];
    this.trainOutputs = outputs[0 .. nsamples];
    this.multipliers = zeros(nsamples);
    this.weights = zeros(nsamples);
    this.bias = 0.0;
    this.errors = zeros(nsamples);
  }

  Real decision_function(Real* testInput) {
    return decision_function(testInput.sliced(ndim));
  }

  Real decision_function(Slice!(1, Real*) testInput) {
    return svIndices.map!(i => weights[i] * kernel(xs[i], testInput)).sum - bias;
  }

  const size_t nsamples, ndim;
  // TODO: make this ndslice
  const Real[] trainInputs, trainOutputs;
  Real penalty = 1.0;

  // to be learned
  Real[] weights, multipliers, errors;
  Real bias;
  size_t[] svIndices;

  auto xs() {
    return this.trainInputs.sliced(nsamples, ndim);
  }
}
