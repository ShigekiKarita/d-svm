import std.array : array;
import std.numeric : dotProduct;
import std.algorithm;
import std.typecons;
// import std.algorithm : maxIndex;


abstract class StructualSVM(RealType, YType, XType) {
  alias Real = RealType;
  alias Y = YType;
  alias X = XType;

  Real[] joint_feature(Y y, X x);
  Real yLoss(Y expect, Y actual);

  this(size_t ndim) {
    this.ndim = ndim;
    this.weight.length = ndim;
  }

  Real wEnergy() {
    // i.e. logPrior of spherical Gaussian N(0, I);
    return dotProduct(weight, weight) / 2;
  }
  Real logPosterior(Y yi, X xi) {
    // actually upper-bound of log-posterior
    return ys.map!(y => yLoss(y, yi) + score(y, xi)).maxElement - score(yi, xi);
  }
  Real wRisk(Tuple!(Y, X)[] yxs) {
    // i.e. objective function
    return wEnergy() + penalty * yxs.map!(yx => logPosterior(yx[0], yx[1])).sum;
  }
  Real score(Y y, X x) {
    return dotProduct(weight, joint_feature(y, x));
  }
  Y predict(X x) {
    return ys.map!(y => score(y, x)).maxIndex;
  }

  size_t ndim;
  Real penalty = 1.0;
  Real[] weight;
  Y[] ys; // TODO: generalize?
}

class BinarySVM(Real=double) : StructualSVM!(Real, Real, Real[]) {

  this(size_t ndim) {
    this.ys = [-1, 1];
    super(ndim);
  }
  override Real[] joint_feature(Y y, X x) {
    return x.map!(xn => xn * y / 2).array;
  }
  override Real yLoss(Y expect, Y actual) {
    return 1.0 - (expect == actual ? 1.0 : 0.0);
  }
}

unittest {
  import std.stdio;
  auto svm = new BinarySVM!double(2);
  writeln("success");
}
