from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
from copy import deepcopy

import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


VariableName = str
StateName = str


@dataclass
class BNSimulator:
    """
    A simulator for a discrete Bayesian Network.

    This class:
        - wraps a pgmpy BayesianModel
        - provides observational and interventional sampling
        - exposes minimal introspection and query utilities

    Typical usage:

        >>> sim = BNSimulator.from_bif("asia.bif")
        >>> df_obs = sim.sample_observational(1000)
        >>> df_do = sim.sample_interventional({"smoke": "yes"}, 500)
    """

    model: "pgmpy.models.BayesianModel"
    name: str = "bn_world"

    def __post_init__(self):
        # Cache state names and cardinalities
        self._state_names: Dict[VariableName, List[StateName]] = {}
        self._cardinality: Dict[VariableName, int] = {}

        for node in self.model.nodes():
            cpd = self.model.get_cpds(node)
            if cpd is None:
                raise ValueError(f"Node {node} has no CPD in the model.")
            self._state_names[node] = list(cpd.state_names[node])
            self._cardinality[node] = len(self._state_names[node])

        # Sampler + inference engine
        self._sampler = BayesianModelSampling(self.model)
        self._inference = VariableElimination(self.model)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_bif(cls, bif_path: str, name: Optional[str] = None) -> "BNSimulator":
        """
        Create a BNSimulator from a .bif file.
        """
        reader = BIFReader(bif_path)
        model = reader.get_model()
        model.check_model()
        return cls(model=model, name=name or bif_path)

    # -------------------------------------------------------------------------
    # Sampling API
    # -------------------------------------------------------------------------

    def sample_observational(
        self,
        n: int,
        variables: Optional[Sequence[VariableName]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Draw n observational samples from the BN.

        Args:
            n: number of i.i.d. samples
            variables: optional subset of variables to return (columns)
            seed: optional random seed

        Returns:
            pandas DataFrame with one row per sample.
        """
        df = self._sampler.forward_sample(size=n, seed=seed)
        if variables is not None:
            df = df[list(variables)]
        return df.reset_index(drop=True)

    def sample_interventional(
        self,
        interventions: Dict[VariableName, StateName],
        n: int,
        variables: Optional[Sequence[VariableName]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Draw n samples under do(interventions).

        interventions: e.g. {"smoke": "yes", "asia": "no"}

        Implements do() by:
            - removing all incoming edges to the intervened variable(s)
            - replacing their CPDs with delta distributions at the fixed state
        """
        intervened_model = self._apply_do(self.model, interventions)
        intervened_sampler = BayesianModelSampling(intervened_model)
        df = intervened_sampler.forward_sample(size=n, seed=seed)
        if variables is not None:
            df = df[list(variables)]
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # do-operator implementation
    # -------------------------------------------------------------------------

    def _apply_do(
        self,
        model,
        interventions: Dict[VariableName, StateName],
    ):
        """
        Create a new pgmpy model with interventions applied.

        This does not modify self.model in place.
        """
        new_model = deepcopy(model)

        for var, state_name in interventions.items():
            if var not in new_model.nodes():
                raise ValueError(f"Unknown variable in intervention: {var}")

            # Ensure state is valid
            states = self.get_state_names(var)
            if state_name not in states:
                raise ValueError(
                    f"Invalid state {state_name!r} for variable {var!r}. "
                    f"Valid states: {states}"
                )

            # Remove all incoming edges to var
            for parent in list(new_model.get_parents(var)):
                new_model.remove_edge(parent, var)

            # Replace its CPD with a delta distribution at state_name
            old_cpd = new_model.get_cpds(var)
            card = len(states)
            idx = states.index(state_name)

            # Single-column CPD: P(var | (no parents)) = delta
            values = [[1.0] if i == idx else [0.0] for i in range(card)]

            new_cpd = TabularCPD(
                variable=var,
                variable_card=card,
                values=values,
                state_names={var: states},
            )

            new_model.remove_cpds(old_cpd)
            new_model.add_cpds(new_cpd)

        new_model.check_model()
        return new_model

    # -------------------------------------------------------------------------
    # Introspection helpers
    # -------------------------------------------------------------------------

    def get_nodes(self) -> List[VariableName]:
        """Return list of variable names."""
        return list(self.model.nodes())

    def get_edges(self) -> List[tuple]:
        """Return list of directed edges (u, v)."""
        return list(self.model.edges())

    def get_state_names(self, var: VariableName) -> List[StateName]:
        """Return list of state names for a variable."""
        return self._state_names[var]

    def get_cardinality(self, var: VariableName) -> int:
        """Return the number of states for a variable."""
        return self._cardinality[var]

    # -------------------------------------------------------------------------
    # Basic probability queries
    # -------------------------------------------------------------------------

    def query_marginal(
        self,
        var: VariableName,
        evidence: Optional[Dict[VariableName, StateName]] = None,
    ) -> pd.Series:
        """
        Return P(var | evidence) as a pandas Series indexed by state name.

        Example:
            sim.query_marginal("dysp", {"smoke": "yes"})
        """
        if evidence is None:
            q = self._inference.query(variables=[var])
        else:
            q = self._inference.query(variables=[var], evidence=evidence)

        # q is a DiscreteFactor; convert to Series with state names as index
        states = self.get_state_names(var)
        probs = [q.values[i] for i in range(len(states))]
        return pd.Series(probs, index=states, name=var)

    def joint_log_prob(self, assignment: Dict[VariableName, StateName]) -> float:
        """
        Compute log P(assignment) for a full assignment of all variables.

        assignment: e.g. {"asia":"yes", "tub":"no", ..., "dysp":"yes"}

        NOTE: For now, requires a full assignment (all variables). You can
        extend this later.
        """
        # pgmpy has log_probability for DynamicBayesianNetwork; here we
        # implement a simple manual log prob via the CPDs.
        import math

        missing = set(self.get_nodes()) - set(assignment.keys())
        if missing:
            raise ValueError(
                f"Assignment missing variables: {sorted(missing)}. "
                f"Currently this method expects full assignments."
            )

        logp = 0.0
        for var in self.get_nodes():
            cpd = self.model.get_cpds(var)
            var_state = assignment[var]

            # Build the "evidence" configuration in the order expected by CPD
            parent_vars = list(cpd.variables[1:])  # first is var itself
            parent_states = [assignment[p] for p in parent_vars]

            # cpds in pgmpy allow accessing values via state names
            p = cpd.get_value(**{var: var_state, **dict(zip(parent_vars, parent_states))})
            logp += math.log(p + 1e-32)

        return logp


# -----------------------------------------------------------------------------
# Simple CLI test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick test for BNSimulator using a .bif file (e.g., ASIA)."
    )
    parser.add_argument("bif_path", type=str, help="Path to a .bif file")
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of samples to draw for quick demo (default: 5)",
    )
    args = parser.parse_args()

    sim = BNSimulator.from_bif(args.bif_path)
    print(f"Loaded BN from {args.bif_path}")
    print("Nodes:", sim.get_nodes())
    print("Edges:", sim.get_edges())
    print()

    print(f"Observational samples (n={args.n}):")
    df_obs = sim.sample_observational(args.n)
    print(df_obs)
    print()

    # If a variable named "smoke" exists, show an intervention example
    if "smoke" in sim.get_nodes():
        print(f"Interventional samples under do(smoke='yes') (n={args.n}):")
        df_do = sim.sample_interventional({"smoke": "yes"}, args.n)
        print(df_do)
        print()

        print("P(dysp | smoke='yes') marginal (if dysp exists):")
        if "dysp" in sim.get_nodes():
            print(sim.query_marginal("dysp", {"smoke": "yes"}))
    else:
        print("No 'smoke' variable; skipping intervention demo.")
