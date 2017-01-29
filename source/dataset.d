// import std.zlib;
import std.process : executeShell;
import std.string : format;
import std.file;
import std.stdio;
import std.array : array;
import std.algorithm : map;
import std.range : stride;
import std.math : ceil;

import mir.ndslice : sliced;


class Mnist {
  immutable ndim = 28 ^^ 2;
  const size_t nsamples;
  ubyte[] target;
  ubyte[] data;

  auto xs(size_t nstride = 3, double scale = 1.0 / 255) pure {
    const newdim = cast(size_t) ceil(cast(double) ndim / nstride);
    return data.stride(nstride).map!(i => scale * i).array.sliced(nsamples, newdim);
  }

  auto ys(int c = 0) pure {
    return target.map!(c => c == 0 ? 1.0 : -1.0).array.sliced(nsamples);
  }

  this(string prefix, size_t nsamples, string root = "./resource/") {
    // TODO: read nsamples from file
    this.nsamples = nsamples;
    const image = root ~ prefix ~ "-images.idx3-ubyte";
    const label = root ~ prefix ~ "-labels.idx1-ubyte";
    if (!image.exists || !label.exists) {
        const script = "download_mnist.sh";
        const cmd = "cd %s && sh %s".format(root, script);
        writeln(cmd ~ "\n...downloading MNIST dataset...");
        auto dl = executeShell(cmd);
        if (dl.status != 0) {
          writeln("failed to execute " ~ script);
        }
        writeln(dl.output);
    }
    this.data.length = nsamples;
    this.target.length = nsamples;

    auto flabel = File(label, "rb");
    auto fimage = File(image, "rb");

    flabel.seek(8);
    fimage.seek(16);
    data = fimage.rawRead(new ubyte[nsamples * ndim]);
    target = flabel.rawRead(new ubyte[nsamples]);
  }
  

  static ref auto train() {
    return new Mnist("train", 60000);
  }

  static ref auto test() {
    return new Mnist("t10k", 10000);
  }
}

unittest {
  auto train = Mnist.train();
  assert(train.target[0 .. 3] == [5, 0, 4]);
  assert(train.target[$-3 .. $] == [5, 6, 8]);

  auto test = Mnist.test();
  assert(test.target[0 .. 3] == [7, 2, 1]);
  assert(test.target[$-3 .. $] == [4, 5, 6]);
}

