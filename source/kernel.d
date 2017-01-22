import std.numeric : dotProduct;
import std.range : zip, ElementType;
import std.algorithm : reduce;
import std.math : exp;


alias linearKernel = dotProduct;

auto gaussianKernel(Range1, Range2)(Range1 a, Range2 b) {
  alias Real = ElementType!Range1;
  Real r = reduce!((a, b) => a + (b[0] - b[1]) ^^ 2)(Real(0), zip(a, b));
  return exp(- r / 2.0);
}

