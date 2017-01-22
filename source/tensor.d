import std.array : array;
import std.range : repeat, take;
import std.conv : to;
import std.algorithm : fold, map;
import std.math : sgn;

import mir.ndslice : sliced;
import mir.random : unpredictableSeed, Random;
import mir.random.algorithm : range;
import mir.random.variable : Bernoulli2Variable, UniformVariable, NormalVariable;


auto zeros(T=double)(size_t n) pure nothrow {
  return repeat(0.to!T, n).array;
}

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

