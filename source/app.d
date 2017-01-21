import std.array : array;
import std.range : iota, take, zip;
import std.algorithm : map, fold, count;
import std.math : sgn;
import std.stdio : writeln, writef;

import mir.ndslice : sliced;
import mir.random : unpredictableSeed, Random;
import mir.random.algorithm : range;
import mir.random.variable : Bernoulli2Variable, UniformVariable, NormalVariable;

import optimizer : SMO;


auto randNormal(S ...)(S shape) {
  auto len = [shape].fold!((a, b) => a * b);
  auto rng = Random(unpredictableSeed);
  return rng
    .range(NormalVariable!double(0, 1))
    .take(len)
    .array.sliced(shape);
}

auto randBin(S ...)(S shape) {
  auto len = [shape].fold!((a, b) => a * b);
  auto rng = Random(unpredictableSeed);
  return rng
    .range(Bernoulli2Variable.init)
    .take(len)
    .map!(a => a.sgn!double * 2 - 1)
    .array.sliced(shape);
}

auto accuracy(R1, R2)(R1 actual, R2 expect) {
  assert(actual.length == expect.length);
  double ok = zip(actual, expect).count!"a[0] == a[1]";
  return ok / expect.length;
}


void main() {
  auto nsamples = 10;
  auto ndim = 100;
  auto xs = randNormal(nsamples, ndim);
  auto ys = randBin(nsamples);

  auto svm = new SMO(xs, ys);
  svm.fit();
  auto pred = iota(nsamples).map!(i => svm.decision_function(xs[i].ptr).sgn);
  writeln("\nfitting result:");
  writeln(ys);
  writeln(pred);
  writef("accuracy: %f\n", accuracy(pred, ys));
}
