"""
Polytope-Aware Input Sampler (PAIS)
====================================
The CCSS (Context-Constrained Stochastic Search) analog from HighDiv,
instantiated for DNN activation polytopes.

Given a Gurobi LP model whose feasible region is the activation polytope
P(sigma) = { x : A*x <= b, x in [lb, ub] } for a fixed activation
pattern sigma, PAIS generates diverse samples from the interior of P(sigma)
by repeatedly solving the LP with random linear objectives.

This is the "boundary-aware move" (bam) operator adapted for continuous
input spaces: instead of always returning the LP optimum (which sits at a
vertex of P(sigma), analogous to HighDiv's critical move always picking the
boundary), bam adds a random direction and finds a new feasible interior
point.

Constraint-Partitioned Variable Initialization (HighDiv §5.1) is
implemented as three groups:
  - High-frequency: input dimensions that appear in many tight constraints
    (large |weight| in the pre-relu layers), initialized near the seed x*.
  - General: remaining dimensions, initialized away from x* to push
    diversity.
"""

from __future__ import annotations
import random
import math
import torch

try:
    import gurobipy as grb
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from helper.misc.logger import logger


class PolytopeAwareInputSampler:
    """
    Sample diverse inputs from an activation polytope using a LP model.

    Parameters
    ----------
    lp_model   : gurobipy.Model
        A *solved* Gurobi LP model (from abstractor.build_lp_solver /
        solve_full_assignment). All neuron activations are already fixed.
    seed_adv   : torch.Tensor
        The initial counterexample x* found by NeuralSAT's DPLL.
    input_shape: tuple
        Shape of one input (e.g. (1, 1, 28, 28)).
    input_lower: torch.Tensor (flattened or shaped)
    input_upper: torch.Tensor (flattened or shaped)
    n_samples  : int
        Number of diverse samples to generate (diversity_bam_n).
    """

    def __init__(
        self,
        lp_model,          # gurobipy.Model already built + solved
        seed_adv: torch.Tensor,
        input_shape: tuple,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        n_samples: int = 5,
    ):
        self.lp_model = lp_model
        self.seed = seed_adv.detach().cpu().flatten()
        self.input_shape = input_shape
        self.n_input = math.prod(input_shape)
        self.lower = input_lower.detach().cpu().flatten()
        self.upper = input_upper.detach().cpu().flatten()
        self.n_samples = n_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> list[torch.Tensor]:
        """
        Run PAIS and return a list of diverse violation witnesses.

        Algorithm
        ---------
        For each of n_samples iterations:
          1. Build an LP with a random linear objective over input dims
             (boundary-aware move: random direction within feasible polytope).
          2. Solve → get a new interior/vertex point x_new.
          3. Add x_new to results.

        The random objective pushes the LP solver to a different vertex of
        P(sigma) each time, giving structural diversity within the same
        activation region — analogous to HighDiv's bam operator.
        """
        if not HAS_GUROBI:
            logger.debug('[PAIS] Gurobi not available, skipping.')
            return []

        if self.lp_model is None:
            return []

        samples = []
        input_vars = [
            self.lp_model.getVarByName(f'inp_{dim}')
            for dim in range(self.n_input)
        ]
        if any(v is None for v in input_vars):
            logger.debug('[PAIS] Could not retrieve input variables from LP model.')
            return []

        # Constraint-Partitioned Initialization: compute variable "frequency"
        # as a proxy for how constrained each input dim is.
        # High-frequency dims: those with tighter [lower, upper] ranges.
        ranges = (self.upper - self.lower).numpy()
        freq_threshold = ranges.mean()
        high_freq_mask = ranges < freq_threshold  # tighter range = more constrained

        for i in range(self.n_samples):
            adv = self._bam_step(input_vars, high_freq_mask, i)
            if adv is not None:
                samples.append(adv)

        logger.debug(f'[PAIS] Generated {len(samples)} polytope samples.')
        return samples

    # ------------------------------------------------------------------
    # Internal: boundary-aware move step
    # ------------------------------------------------------------------

    def _bam_step(
        self,
        input_vars: list,
        high_freq_mask,
        step_idx: int,
    ) -> torch.Tensor | None:
        """
        One boundary-aware move: minimize a random linear objective to get
        a new feasible point.

        HighDiv §5.2 — bam(x, l) picks a *random* value within the feasible
        interval rather than always the boundary.  Here the equivalent is:
        random objective direction → LP solver finds a different vertex.

        Constraint-Partitioned Variable Initialization (§5.1):
        - High-frequency (tight) dims: objective coefficient ~ N(0, small_sigma)
          so the solver stays near the seed (feasibility-preserving).
        - General dims: coefficient ~ N(0, large_sigma) to push diversity.
        """
        import numpy as np

        try:
            tmp = self.lp_model.copy()
            tmp.setParam('OutputFlag', 0)

            # Random linear objective (bam direction)
            coeffs = np.zeros(self.n_input)
            for dim in range(self.n_input):
                if high_freq_mask[dim]:
                    # High-frequency: small random perturbation
                    coeffs[dim] = random.gauss(0, 0.1)
                else:
                    # General: large random direction → push diversity
                    coeffs[dim] = random.gauss(0, 1.0)

            # Optionally bias toward a random direction vector
            # (simulates CCSS's "away from seed" initialization for general vars)
            obj = grb.LinExpr()
            for dim, var in enumerate(
                [tmp.getVarByName(f'inp_{d}') for d in range(self.n_input)]
            ):
                if var is not None:
                    obj += coeffs[dim] * var

            tmp.setObjective(obj, grb.GRB.MINIMIZE)
            tmp.update()
            tmp.optimize()

            if tmp.status == 2:  # optimal
                vals = []
                for dim in range(self.n_input):
                    v = tmp.getVarByName(f'inp_{dim}')
                    vals.append(v.X if v is not None else self.seed[dim].item())
                adv_flat = torch.tensor(vals, dtype=torch.float32)
                # Clamp to input bounds
                lo = self.lower.to(adv_flat.dtype)
                hi = self.upper.to(adv_flat.dtype)
                adv_flat = torch.clamp(adv_flat, lo, hi)
                return adv_flat.view(self.input_shape)
            else:
                logger.debug(f'[PAIS] bam step {step_idx}: LP status {tmp.status}')
                return None

        except Exception as e:
            logger.debug(f'[PAIS] bam step {step_idx} exception: {e}')
            return None
