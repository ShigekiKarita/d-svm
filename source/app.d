import std.range : iota, zip;
import std.algorithm : count, map;
import std.random : dice;
import std.math;
import std.stdio;
import std.datetime;
import std.conv : to;

import tensor;
import kernel;
import optimizer;


auto accuracy(R1, R2)(R1 expect, R2 actual) {
  if (expect.length != actual.length) {
    throw new Exception("expect.length != actual.length");
  }
  double ok = zip(actual, expect).count!"a[0] == a[1]";
  return ok / expect.length;
}

auto rescale(double[] gridPreds) {
  import std.algorithm;
  auto gpmin = minElement(gridPreds);
  auto gpmax = maxElement(gridPreds);
  auto gpabs = max(gpmin.fabs, gpmax.fabs) * 2;
  return gridPreds.map!(p => p / gpabs + 0.5);
}

void plotSurface(C, Xs, Ys)(string name, C svm, Xs xs, Ys ys, size_t resolution=100) {
  import std.algorithm; // : cartesianProduct;
  import std.string;
  import std.stdio : writeln;
  import std.array : array;

  import ggplotd.aes : aes;
  import ggplotd.geom : geomPolygon, geomPoint;
  import ggplotd.ggplotd : GGPlotD, putIn, title;
  import ggplotd.colour : colourGradient;
  import ggplotd.colourspace : XYZ;

  const n = svm.nsamples;
  const xmin = minElement(xs[0..$, 0]);
  const xmax = maxElement(xs[0..$, 0]);
  const ymin = minElement(xs[0..$, 1]);
  const ymax = maxElement(xs[0..$, 1]);
  const xstep = (xmax - xmin) / resolution;
  const ystep = (ymax - ymin) / resolution;
  auto grid = cartesianProduct(iota(xmin, xmax, xstep), iota(ymin, ymax, ystep)).array;
  auto gridPreds = grid.map!(i => svm.decision_function([i[0], i[1]])).array.rescale;

  auto gg =
    iota(grid.length)
    .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], grid[i][1], gridPreds[i], 1.0))
    .geomPoint
    .putIn(GGPlotD());

  gg = iota(n)
    .map!(i => aes!("x", "y", "colour", "size")(xs[i,0], xs[i,1], ys[i] == 1 ? 1 : 0, 1.0))
    .geomPoint
    .putIn(gg);

  gg = colourGradient!XYZ( "cornflowerBlue-white-crimson" )
    .putIn(gg);

  auto expect = xs.map!(x => svm.decision_function(x.ptr).sgn);
  auto tstr = "%s (fit accuracy: %.2f)".format(name, accuracy(expect, ys));
  tstr.writeln;
  gg.put(title(tstr));
  gg.save("./resource/" ~ name ~ ".png");
}

void testArtificialData() {
  auto nsamples = 200;
  auto ndim = 2;
  auto xs = randNormal(nsamples, ndim);
  auto ys = randBin(nsamples);
  foreach (ref x, y; zip(xs, ys)) {
    if (y == 1.0) {
      x[0] += 2.0 * dice(0.5, 0.5) - 2.0;
      x[0] *= 4.0;
    } else {
      x[1] += 2.0 * dice(0.5, 0.5) - 2.0;
      x[1] *= 4.0;
    }
  }

  void exec(string name)() {
    mixin("auto svm = new SMO!(" ~ name ~ ")(xs, ys);");
    auto r = benchmark!(() => svm.fit())(1);
    writeln(to!Duration(r[0]));
    plotSurface(name, svm, xs, ys);
  }

  exec!"gaussianKernel";
  exec!"linearKernel";
  exec!"polynomialKernel";
}

void testMnist() {
  // TODO: impl one vs rest SVM wrapper
  import dataset;

  writeln("small MNIST test");
  const nstride = 1;
  auto train = new Mnist("train", 400);
  auto ovr = new OneVsRestSMO!gaussianKernel(10, train.xs(nstride), train.ys);
  auto r = benchmark!(() => ovr.fit())(1);
  writeln(to!Duration(r[0]));

  auto ytrain = train.xs(nstride).map!(x => ovr.decision_function(x.ptr));
  writef("train accuracy %f\n", accuracy(ytrain, train.ys));

  auto test = new Mnist("t10k", 100);
  auto expect = test.xs(nstride).map!(x => ovr.decision_function(x.ptr));
  writef("test accuracy %f\n", accuracy(expect, test.ys));

  // auto svm = new SMO!gaussianKernel(train.xs, train.ys);
  // auto r = benchmark!(() => svm.fit())(1);
  // writeln(to!Duration(r[0]));
  // auto test = Mnist.test();
  // auto expect = test.xs.map!(x => svm.decision_function(x.ptr).sgn);
  // writef("accuracy %f\n", accuracy(expect, test.ys));
}

version(unittest) {} else {
  void main() {
    // testArtificialData();
    testMnist();
  }
} // not unittest
