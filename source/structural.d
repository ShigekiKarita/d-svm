import std.array : array;
import std.numeric : dotProduct;
import std.algorithm;
import std.typecons;
// import std.algorithm : maxIndex;


struct SVMParameters(Real, Y, X) {
  const size_t ndim;
  Real penalty = 1.0;
  Real[] weight;
  Y[] ys; // TODO: generalize?
}

abstract class StructuralSVM(RealType, YType, XType) {
  // FIXME: YType -> ElementType!Ys
  alias Real = RealType;
  alias Y = YType;
  alias X = XType;
  alias P = SVMParameters!(Real, Y, X);

  Real[] jointFeature(Y y, X x);
  Real yLoss(Y expect, Y actual);

  this(size_t ndim) {
    params.ndim = ndim;
    params.weight.length = ndim;
  }

private:
  P params;

  Real margin(X x, Y[] trainYs) {
    // this is just for debug
    return trainYs.map!(y => score(y, x)).minElement
       - params.ys.map!(y => score(y, x)).maxElement;
  }
  Real wEnergy() {
    // a.k.a. 0.5 * |W|^2
    // i.e. logPrior of spherical Gaussian N(0, I);
    return params.weight.dotProduct(params.weight) / 2;
  }
  Real logPosterior(Y yi, X xi) {
    // a.k.a. constraint or slack variable \eta_t
    // actually upper-bound of log-posterior
    return maxElement(params.ys.map!(y => yLoss(y, yi) + score(y, xi))) - score(yi, xi);
  }
  Real wRisk(Tuple!(Y, X)[] yxs) {
    // i.e. objective function
    return wEnergy() + params.penalty * yxs.map!(yx => logPosterior(yx[0], yx[1])).sum;
  }
  Real score(Y y, X x) {
    return params.weight.dotProduct(jointFeature(y, x));
  }
  Y predict(X x) {
    auto ys = params.ys;
    return ys[ys.map!(y => score(y, x)).maxIndex];
  }
}

class BinarySVM(Real=double) : StructuralSVM!(Real, Real, Real[]) {
  this(size_t ndim) {
    super(ndim);
    params.ys = [-1, 1];
  }
  override Real[] jointFeature(Y y, X x) {
    return x.map!(xn => xn * y / 2).array;
  }
  override Real yLoss(Y expect, Y actual) {
    return 1.0 - (expect == actual ? 1.0 : 0.0);
  }
}

unittest {
  import std.stdio;
  auto bsvm = new BinarySVM!double(2);
  // auto msvm = new MultinomialSVM!double(2, 10);
  writeln("success");
}


class NslackCuttingPlace(alias SVM, Real = double, Xs, Ys) {
  Xs xsTrain;
  Ys ysTrain;
  ElementType!Ys[] constraints;
  Real[] multipliers;
  SVM svm;

  this(SVM svm, Xs xs, Ys ys) {
    this.svm = svm;
    this.xsTrain = xsTrain;
    this.ysTrain = ysTrain;
    this.multipliers.length = ysTrain.length;
  }

  void fit(Real tolerance = 1e-5) {
    auto ysParam = svm.param.ys;
    bool changed = false;

    while (!changed) {
      foreach (xt, yt, m; zip(xsTrain, ysTrain, multipliers)) {
        auto ymax = ysParam.map!(yp => svm.yLoss(yp, yt) + svm.score(yp, xt)).maxElement;
        if (ymax > m + tolerance) {
          changed = true; // FIXME: see below.
          constraints ~= [ymax]; // FIXME: needs to check doubling?
          // TODO: solve QP as (w, ms) = argmin svm.wRisk();
        }
      }
    }
  }
}
