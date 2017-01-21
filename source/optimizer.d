import std.algorithm;
import std.math : fabs;

import svm;


class SMO : SVM {
  this(Args ...)(Args args) {
    super(args);
  }

  size_t update(const size_t i, const size_t j)
  {
    if (i == j) return 0;
    
    const double ai_old = multipliers[i];
    const double aj_old = multipliers[j];
    double ai_new;
    double aj_new;
    double U, V;

    if (trainOutputs[i] != trainOutputs[j])
      {
        U = max(0.0, ai_old - aj_old);
        V = min(penalty, penalty + ai_old - aj_old);
      }
    else
      {
        U = max(0.0, ai_old + aj_old - penalty);
        V = min(penalty, ai_old + aj_old);
      }
    
    if (U == V) return 0;
    
    const double kii = kernel(xs[i], xs[i]);
    const double kjj = kernel(xs[j], xs[j]);
    const double kij = kernel(xs[i], xs[j]);
    const double k = kii + kjj - 2.0*kij;

    double Ej;
    if (multipliers[j] > multTolerance && multipliers[j] < (penalty - multTolerance))
      {
        Ej = errors[j];
      }
    else
      {
        Ej = f(j) - trainOutputs[j];
      }
    
    bool biasClip = false;
    
    if (k <= 0.0)
      {
        // ai = U のときの目的関数の値
        ai_new = U;
        aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);
        multipliers[i] = ai_new; // 仮置き
        multipliers[j] = aj_new;
        double v1 = f(j) + bias - trainOutputs[j] * aj_old * kjj - trainOutputs[i] * ai_old * kij;
        double v2 = f(i) + bias - trainOutputs[j] * aj_old * kij - trainOutputs[i] * ai_old * kii;
        double Lobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
          - trainOutputs[j] * trainOutputs[i] * kij * aj_new * ai_new
          - trainOutputs[j] * aj_new * v1 - trainOutputs[i] * ai_new * v2;
        // ai = V のときの目的関数の値
        ai_new = V;
        aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);
        multipliers[i] = ai_new; // 仮置き
        multipliers[j] = aj_new;
        v1 = f(j) + bias - trainOutputs[j] * aj_old * kjj - trainOutputs[i] * ai_old * kij;
        v2 = f(i) + bias - trainOutputs[j] * aj_old * kij - trainOutputs[i] * ai_old * kii;
        double Hobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
          - trainOutputs[j] * trainOutputs[i] * kij * aj_new * ai_new
          - trainOutputs[j] * aj_new * v1 - trainOutputs[i] * ai_new * v2;
        
        if (Lobj > Hobj + multTolerance)
          {
            biasClip = true;
            ai_new = U;
          }
        else if (Lobj < Hobj - multTolerance)
          {
            biasClip = true;
            ai_new = V;
          }
        else
          {
            biasClip = true;
            ai_new = ai_old;
          }
        multipliers[i] = ai_old; // 元に戻す
        multipliers[j] = aj_old;
      }
    else
      {
        ai_new = ai_old + (trainOutputs[i] * (Ej - currentError) / k);
        if (ai_new > V)
          {
            biasClip = true;
            ai_new = V;
          }
        else if (ai_new < U)
          {
            biasClip = true;
            ai_new = U;
          }
      }
    if (fabs(ai_new - ai_old) < multTolerance * (ai_new + ai_old + multTolerance))
      {
        return 0;
      }
    
    // multipliers[j]更新
    aj_new = aj_old + trainOutputs[i] * trainOutputs[j] * (ai_old - ai_new);
    // bias更新
    double old_b = bias;
    if (multipliers[i] > multTolerance && multipliers[i] < (penalty - multTolerance))
      {
        bias += currentError + (ai_new - ai_old) * trainOutputs[i] * kii +
          (aj_new - aj_old) * trainOutputs[j] * kij;
      }
    else if (multipliers[j] > multTolerance && multipliers[j] < (penalty - multTolerance))
      {
        bias += Ej + (ai_new - ai_old) * trainOutputs[i] * kij +
          (aj_new - aj_old) * trainOutputs[j] * kjj;
      }
    else
      {
        bias += (currentError + (ai_new - ai_old) * trainOutputs[i] * kii +
              (aj_new - aj_old) * trainOutputs[j] * kij +
              Ej + (ai_new - ai_old) * trainOutputs[i] * kij +
              (aj_new - aj_old) * trainOutputs[j] * kjj) / 2.0;
      }
    // err更新
    for (int m = 0; m < nsamples; m++)
      {
        if (m == i || m == j) continue;

        else if (multipliers[m] > multTolerance && multipliers[m] < (penalty - multTolerance))
          {
            errors[m] = errors[m] + trainOutputs[j] * (aj_new - aj_old) * kernel(xs[j], xs[m])
              + trainOutputs[i] * (ai_new - ai_old) * kernel(xs[i], xs[m])
              + old_b - bias;
          }
      }
    
    multipliers[i] = ai_new;
    multipliers[j] = aj_new;
    
    if (biasClip && ai_new > multTolerance && ai_new < penalty - multTolerance)
      {
        errors[i] = f(i) - trainOutputs[i];
      }
    else
      {
        errors[i] = 0.0;
      }
    errors[j] = f(j) - trainOutputs[j];
    
    return 1;
  }
}
