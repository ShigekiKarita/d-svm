import std.range : iota, zip;
import std.algorithm : count, map;
import std.stdio : writeln, writef;

import tensor;
import kernel;
import optimizer;


auto accuracy(R1, R2)(R1 actual, R2 expect) {
  assert(actual.length == expect.length);
  double ok = zip(actual, expect).count!"a[0] == a[1]";
  return ok / expect.length;
}

void plotSurface(C, Xs)(C svm, Xs xs, size_t resolution=100) {
  import ggplotd.aes : aes;
  import ggplotd.geom : geomPolygon;
  import ggplotd.ggplotd : GGPlotD, putIn;

  auto points = iota(svm.nsamples).map!(i => [xs[i, 0], xs[i, 1], svm.decision_function(xs[i].ptr)]);
  auto gg = points
    .map!((a) => aes!("x", "y", "colour")(a[0], a[1], a[2]))
    .geomPolygon
    .putIn(GGPlotD());
  gg.save( "polygon.png" );
}

void main() {
  auto nsamples = 50;
  auto ndim = 2;
  auto xs = randNormal(nsamples, ndim);
  auto ys = randBin(nsamples);

  auto svm = new SMO!gaussianKernel(xs, ys);
  svm.fit();
  auto pred = iota(nsamples).map!(i => svm.decision_function(xs[i].ptr).sgn);
  writeln("\nfitting result:");
  writeln(ys);
  writeln(pred);
  writef("accuracy: %f\n", accuracy(pred, ys));
  plotSurface(svm, xs);
}

