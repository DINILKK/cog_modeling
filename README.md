# Bayesian Prospect Theory on choices13k

A cognitive modeling project fitting a hierarchical Bayesian Prospect Theory model
to human risky decision-making data from the choices13k dataset (Peterson et al., 2021, *Science*).

## What This Project Does

Humans don't make decisions like rational agents. They:
- **Overweight small probabilities** and underweight large ones (buying lottery tickets)
- **Show diminishing sensitivity** to outcomes as magnitudes grow
- **Add noise** to their choices that varies across individuals

This project estimates these biases as posterior distributions using Bayesian inference in PyMC.

## Model

**Prospect Theory (Tversky & Kahneman, 1992)** with:

| Parameter | Meaning | Prior |
|-----------|---------|-------|
| `alpha` | Value function curvature (risk attitude) | Beta(2,2) |
| `gamma` | Probability weighting curvature | Beta(2,2) |
| `delta` | Probability weighting elevation | HalfNormal(1) |
| `beta`  | Decision noise (softmax temperature) | HalfNormal(2) |
| `kappa` | Beta-likelihood concentration | HalfNormal(10) |

**Probability Weighting (Prelec, 1998):**
```
w(p) = exp(-delta * (-log(p))^gamma)
```

**Choice Rule:**
```
P(choose B) = sigmoid(beta * (V(B) - V(A)))
```

## Setup

```bash
pip install pymc arviz matplotlib pandas numpy
```

## Dataset

The choices13k dataset is openly available at:
https://github.com/jcpeterson/choices13k

> **Note:** The dataset is not included in this repository. Clone it separately and place it in a `choices13k/` folder in the project root before running the script:
> ```bash
> git clone https://github.com/jcpeterson/choices13k.git
> ```

**Full citation:**
Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021).
Using large-scale experiments and machine learning to discover theories of human decision-making.
*Science*, 372(6547), 1209–1214. https://doi.org/10.1126/science.abe2629

## Run

```bash
python prospect_theory_model.py
```

## Outputs

- `posterior_distributions.png` — Posterior of all parameters
- `probability_weighting.png` — Fitted probability weighting function with uncertainty bands
- `posterior_predictive_check.png` — Predicted vs observed choice rates

## Key Results

| Parameter | Posterior Mean | Interpretation |
|-----------|---------------|----------------|
| alpha | 0.782 | Risk averse value function (< 1 = concave) |
| gamma | 0.879 | Inverse-S probability weighting (< 1 = distortion) |
| delta | 0.881 | Slight underweighting of probabilities overall |
| beta  | 0.243 | Low decision noise — consistent choices |

All r-hat values = 1.0, indicating perfect chain convergence.

## References

- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory. *Journal of Risk and Uncertainty*, 5(4), 297–323.
- Prelec, D. (1998). The probability weighting function. *Econometrica*, 66(3), 497–527.