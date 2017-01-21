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

import svm;

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


void main()
{
  auto nsamples = 3;
  auto ndim = 4;
  auto xs = randNormal(nsamples, ndim);
  auto ys = randBin(nsamples);
  xs.writeln;
  ys.writeln;

  auto svm = new SVM(xs, ys);
  svm.fit();
  svm.decision_function(xs[0].ptr).writeln;
}
