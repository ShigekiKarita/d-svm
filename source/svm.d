import std.array : array;
import std.range : repeat, iota, zip, ElementType;
import std.conv : to;
import std.algorithm : map, sum, filter;

import mir.ndslice : Slice, sliced;
import mir.random : unpredictableSeed, Random;
import mir.random.algorithm : range;
import mir.random.variable : Bernoulli2Variable, UniformVariable, NormalVariable;

import tensor : zeros;
import kernel;


class SVM(alias kernel, Real) {

  this(Real* inputs, Real* outputs, size_t nsamples, size_t ndim, ulong seed=unpredictableSeed) {
    this.nsamples = nsamples;
    this.ndim = ndim;
    this.trainInputs = inputs[0 .. nsamples * ndim];
    this.trainOutputs = outputs[0 .. nsamples];
    this.multipliers = zeros(nsamples);
    this.weights = zeros(nsamples);
    this.bias = 0.0;
    this.errors = zeros(nsamples);
    this.rng = Random(seed);
  }

  this(Slice!(2, Real*) inputs, Slice!(1, Real*) outputs, ulong seed=unpredictableSeed) {
    this(inputs.ptr, outputs.ptr, inputs.shape[0], inputs.shape[1], seed);
  }

  Real decision_function(const Real* testInput) {
    auto t = testInput.sliced(ndim);
    return svIndices.map!(i => weights[i] * kernel(xs[i], t)).sum - bias;
  }

  Real decision_function(const Real[] testInput) {
    return svIndices.map!(i => weights[i] * kernel(xs[i], testInput)).sum - bias;
  }


  const size_t nsamples, ndim;
  // TODO: make this ndslice
  const Real[] trainInputs, trainOutputs;

protected:
  Real penalty = 1.0;
  Random rng;

  // to be learned
  Real[] weights, multipliers, errors;
  Real bias;
  size_t[] svIndices;

  const f(const size_t i) {
    auto svs = iota(this.nsamples).filter!(j => (multipliers[j] == 0.0));
    return svs.map!(j => multipliers[j] * trainOutputs[j] * kernel(xs[j], xs[i])).sum - bias;
  }

  const xs() {
    return this.trainInputs.sliced(nsamples, ndim);
  }
}
