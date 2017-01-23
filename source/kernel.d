import std.numeric : dotProduct;
import std.range : zip, ElementType;
import std.algorithm : reduce;
import std.math : exp;


alias linearKernel = dotProduct;

auto gaussianKernel(Range1, Range2)(Range1 a, Range2 b, ElementType!Range1 denom=2.0) {
  alias Real = ElementType!Range1;
  Real r = reduce!((a, b) => a + (b[0] - b[1]) ^^ 2)(Real(0), zip(a, b));
  return exp(- r / denom);
}

auto polynomialKernel(Range1, Range2)
  (Range1 a, Range2 b, ElementType!Range1 bias=0.0, ElementType!Range1 power=2.0) {
  return (dotProduct(a, b) + bias) ^^ power;
}

