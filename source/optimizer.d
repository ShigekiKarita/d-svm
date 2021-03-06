import std.variant : Algebraic;
import std.range : iota, enumerate, generate, take;
import std.algorithm : map, sum, filter;
import std.math : fabs;

import mir.ndslice : Slice, sliced;
import mir.random : unpredictableSeed, Random;
import mir.random.algorithm; // : range;
import mir.random.variable : Bernoulli2Variable, UniformVariable, NormalVariable;

import std.algorithm;
import std.math : fabs;

import svm : SVM;


class OneVsRestSMO(alias Kernel, Real = double) {
  import std.array : array;

  SMO!Kernel[] svms;
  const size_t nclass;

  this(size_t nclass, Slice!(2, Real*) inputs, Slice!(1, Real*) outputs, ulong seed=unpredictableSeed) {
    this.nclass = nclass;
    svms.length = nclass;
    foreach (i, ref s; svms) {
      auto bins = outputs.map!(c => c == i ? 1.0 : -1.0).array.sliced(outputs.length);
      s = new SMO!Kernel(inputs, bins);
    }
  }

  ref auto fit(Args ...)(Args args) {
    foreach(i, ref s; svms) {
      // import std.stdio;
      // writefln("%d-th svm", i);
      s.fit(args);
    }
    return this;
  }

  auto decision_function(const Real* testInput) {
    return svms.map!(s => s.decision_function(testInput)).maxIndex;
  }
}



class SMO(alias kernel, Real = double) : SVM!(kernel, Real) {
  this(Args ...)(Args args) {
    super(args);
  }

  ref auto fit(Real penalty=1.0, int maxIter=10000, Real tolerance=1e-6) {
    this.multTolerance = multTolerance;
    this.penalty = penalty;

    bool isAll = true;
    foreach (_; iota(maxIter)) {
      size_t nchanged = 0;
      foreach (i; iota(nsamples)) {
        if (isAll || insidePenalty(i)) {
          if (update(i)) { ++nchanged; }
        }
      }
      if (isAll) {
        isAll = false;
        if (nchanged == 0) break;
      } else if (nchanged == 0) {
        isAll = true;
      }
      // writef("%d,", nchanged);
    }

    // get support vectors
    svIndices = [];
    foreach (i, a; multipliers) {
      if (a != 0.0) { svIndices ~= [i]; }
    }
    // get a weight vector
    foreach (i; svIndices) {
      weights[i] = trainOutputs[i] * multipliers[i];
    }

    return this;
  }

private:
  Real multTolerance = 1e-6;
  Real kktTolerance = 1e-6;

  bool insidePenalty(size_t i) {
    const multiplier = multipliers[i];
    return multTolerance < multiplier && multiplier < (penalty - multTolerance);
  }

  bool needToOptimize(size_t i) {
    const multiplier = multipliers[i];
    const yfi = currentError * trainOutputs[i];
    return // check KKT condition
      (multiplier < (penalty - multTolerance) && yfi < -kktTolerance) ||
      (multiplier > multTolerance             && yfi > kktTolerance);
  }

  size_t randomIndex() {
    return UniformVariable!size_t(0, nsamples - 1)(rng);
  }

  auto randomIndices() {
    const off = randomIndex();
    return iota(nsamples).map!(i => (i + off) % nsamples);
  }

  alias MaybeIndex = Algebraic!(bool, size_t);

  auto randomSearchMaxErrorIndex(Real error) {
    Real maxdiff = -Real.infinity;
    MaybeIndex resultIndex;
    foreach(i; randomIndices) {
      if (insidePenalty(i)) {
        const diff = fabs(errors[i] - error);
        if (diff > maxdiff) {
          maxdiff = diff;
          resultIndex = i;
        }
      }
    }
    return resultIndex;
  }

  Real currentError;
  bool update(size_t i) {
    currentError = insidePenalty(i) ? errors[i] : f(i) - trainOutputs[i];
    if (needToOptimize(i)) { return optimize(i); }
    return false;
  }

  bool optimize(size_t i) {
    const imax = randomSearchMaxErrorIndex(currentError);
    if (imax.hasValue && stepSMO(i, *imax.peek!size_t)) {
      return true;
    }
    foreach (j; randomIndices) {
      if (insidePenalty(j) && stepSMO(i, j)) {
        return true;
      }
    }
    foreach (j; randomIndices) {
      if (!insidePenalty(j) && stepSMO(i, j)) {
        return true;
      }
    }
    return false;
  }

  bool stepSMO(const size_t i, const size_t j) {
    // TODO: clean and refactor
    if (i == j) return false;

    const ai_old = multipliers[i];
    const aj_old = multipliers[j];
    Real ai_new;
    Real aj_new;
    Real U, V;

    if (trainOutputs[i] != trainOutputs[j]) {
      U = max(0.0, ai_old - aj_old);
      V = min(penalty, penalty + ai_old - aj_old);
    } else {
      U = max(0.0, ai_old + aj_old - penalty);
      V = min(penalty, ai_old + aj_old);
    }

    if (U == V) { return false; }

    const kii = kernel(xs[i], xs[i]);
    const kjj = kernel(xs[j], xs[j]);
    const kij = kernel(xs[i], xs[j]);
    const k = kii + kjj - 2.0*kij;
    const Ej = insidePenalty(j) ? errors[j] : f(j) - trainOutputs[j];
    bool biasClip = false;

    if (k <= 0.0) {
      // get objective when multipliers[i] = U
      ai_new = U;
      aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);
      multipliers[i] = ai_new; // to be reverted
      multipliers[j] = aj_new;
      auto v1 = f(j) + bias - trainOutputs[j] * aj_old * kjj - trainOutputs[i] * ai_old * kij;
      auto v2 = f(i) + bias - trainOutputs[j] * aj_old * kij - trainOutputs[i] * ai_old * kii;
      const Lobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
        - trainOutputs[j] * trainOutputs[i] * kij * aj_new * ai_new
        - trainOutputs[j] * aj_new * v1 - trainOutputs[i] * ai_new * v2;

      // get objective when multipliers[i] = V
      ai_new = V;
      aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);
      multipliers[i] = ai_new; // to be reverted
      multipliers[j] = aj_new;
      v1 = f(j) + bias - trainOutputs[j] * aj_old * kjj - trainOutputs[i] * ai_old * kij;
      v2 = f(i) + bias - trainOutputs[j] * aj_old * kij - trainOutputs[i] * ai_old * kii;
      const Hobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
        - trainOutputs[j] * trainOutputs[i] * kij * aj_new * ai_new
        - trainOutputs[j] * aj_new * v1 - trainOutputs[i] * ai_new * v2;

      if (Lobj > Hobj + multTolerance) {
        biasClip = true;
        ai_new = U;
      } else if (Lobj < Hobj - multTolerance) {
        biasClip = true;
        ai_new = V;
      } else {
        biasClip = true;
        ai_new = ai_old;
      }

      // revert
      multipliers[i] = ai_old;
      multipliers[j] = aj_old;
    } else {
      ai_new = ai_old + (trainOutputs[i] * (Ej - currentError) / k);

      if (ai_new > V) {
        biasClip = true;
        ai_new = V;
      } else if (ai_new < U) {
        biasClip = true;
        ai_new = U;
      }
    }

    if (fabs(ai_new - ai_old) < multTolerance * (ai_new + ai_old + multTolerance)) {
      return false;
    }

    // update multipliers[j]
    aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);

    // update bias
    const old_b = bias;
    if (insidePenalty(i)) {
      bias += currentError + (ai_new - ai_old) * trainOutputs[i] * kii +
        (aj_new - aj_old) * trainOutputs[j] * kij;
    } else if (insidePenalty(j)) {
      bias += Ej + (ai_new - ai_old) * trainOutputs[i] * kij +
        (aj_new - aj_old) * trainOutputs[j] * kjj;
    } else {
      bias += (currentError + (ai_new - ai_old) * trainOutputs[i] * kii +
               (aj_new - aj_old) * trainOutputs[j] * kij +
               Ej + (ai_new - ai_old) * trainOutputs[i] * kij +
               (aj_new - aj_old) * trainOutputs[j] * kjj) / 2.0;
    }

    // update errors
    foreach (n; iota(nsamples)) {
      if (n == i || n == j) {
        continue;
      } else if (insidePenalty(n)) {
        errors[n] += trainOutputs[j] * (aj_new - aj_old) * kernel(xs[j], xs[n])
          + trainOutputs[i] * (ai_new - ai_old) * kernel(xs[i], xs[n])
          + old_b - bias;
      }
    }

    multipliers[i] = ai_new;
    multipliers[j] = aj_new;
    errors[i] = (biasClip && insidePenalty(i)) ? f(i) - trainOutputs[i] : 0.0;
    errors[j] = f(j) - trainOutputs[j];

    return true;
  }
}


