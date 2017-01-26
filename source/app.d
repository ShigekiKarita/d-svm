import std.range : iota, zip;
import std.algorithm : count, map;
import std.random : dice;

import tensor;
import kernel;
import optimizer;


auto accuracy(S, R1, R2)(S svm, R1 xs, R2 actual) {
  auto expect = iota(actual.length).map!(i => svm.decision_function(xs[i].ptr).sgn);
  double ok = zip(actual, expect).count!"a[0] == a[1]";
  return ok / expect.length;
}

double sigmoid(double x, double scale=0.1, double margin=0.05) {
  return 1.0 / (1.0 + exp(scale * x)) * (1.0 - margin * 2.0) + margin;
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
  auto gridPreds = grid.map!(i => svm.decision_function([i[0], i[1]])).array;
  auto gpmin = minElement(gridPreds);
  auto gpnorm = maxElement(gridPreds) - gpmin;

  auto gg =
    iota(grid.length)
    .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], grid[i][1], sigmoid(gridPreds[i]), 1.0))
    .geomPoint
    .putIn(GGPlotD());

  gg = iota(n)
    .map!(i => aes!("x", "y", "colour", "size")(xs[i,0], xs[i,1], ys[i] == 1 ? 0 : 1, 1.0))
    .geomPoint
    .putIn(gg);

  gg = colourGradient!XYZ( "cornflowerBlue-white-crimson" )
    .putIn(gg);

  auto tstr = "%s (fit accuracy: %.2f)".format(name, accuracy(svm, xs, ys));
  tstr.writeln;
  gg.put(title(tstr));
  gg.save("./resource/" ~ name ~ ".png");
}


void main() {
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
    import std.datetime;
    import std.stdio;
    import std.conv : to;
    mixin("auto svm = new SMO!(" ~ name ~ ")(xs, ys);");
    auto r = benchmark!(() => svm.fit())(1);
    writeln(to!Duration(r[0]));
    plotSurface(name, svm, xs, ys);
  }

  exec!"gaussianKernel";
  exec!"linearKernel";
  exec!"polynomialKernel";
}

