# episode_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import random

from simulator import BNSimulator
from world_model import WorldAgent, WorldResponse


@dataclass
class EpisodeSpec:
    """
    What the ScientistAgent is allowed to know at reset time.
    This is the "interface layer" you mentioned (reveal/hide variables, etc.).
    """
    story: str
    visible_variables: List[str]
    variable_states: Dict[str, List[str]]   # only for visible vars
    variable_descriptions: Dict[str, str]   # only for visible vars
    # Optional: give the agent a goal question; for curiosity-only, can be None
    question: Optional[str] = None
    # Fixed horizon, no budget
    horizon: int = 15


@dataclass
class EpisodeManager:
    """
    Environment wrapper: owns the simulator/world, decides what the scientist sees,
    and runs an interaction trajectory.
    """
    world: WorldAgent
    seed: int = 0

    # “hiding nodes” knobs
    reveal_fraction: float = 1.0   # 1.0 = reveal all variable names
    min_reveal: int = 8            # if reveal_fraction small, still reveal at least this many
    horizon: int = 15

    # optional: allow extension in the world
    allow_extension: bool = False

    # internal
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

        # Ensure world extension setting matches episode config
        self.world.allow_extension = self.allow_extension

    def reset(self, *, question: Optional[str] = None) -> EpisodeSpec:
        """
        Defines the initial interface to the scientist:
        - selects which variables are visible (names + states + descriptions)
        - provides story and optional question
        """
        all_vars = list(self.world.simulator.get_nodes())
        self._rng.shuffle(all_vars)

        k = max(self.min_reveal, int(round(len(all_vars) * self.reveal_fraction)))
        k = min(k, len(all_vars))
        visible = sorted(all_vars[:k])

        states = {v: self.world.simulator.get_state_names(v) for v in visible}
        descs = {v: self.world.var_descriptions.get(v, "") for v in visible}

        return EpisodeSpec(
            story=self.world.story,
            visible_variables=visible,
            variable_states=states,
            variable_descriptions=descs,
            question=question,
            horizon=self.horizon,
        )

    def step(self, query: str, *, t: int) -> WorldResponse:
        """
        Execute one scientist action in the world.
        We pass a deterministic seed to keep runs reproducible.
        """
        # Derive a deterministic per-step seed
        step_seed = self.seed * 10_000 + t
        return self.world.handle(query, seed=step_seed)

    def run(self, scientist, *, question: Optional[str] = None) -> Tuple[EpisodeSpec, List[WorldResponse]]:
        """
        Run a full episode of length horizon, no budget.
        """
        spec = self.reset(question=question)
        traj: List[WorldResponse] = []

        scientist.reset(spec)

        for t in range(spec.horizon):
            query = scientist.choose_next_query(t=t)
            out = self.step(query, t=t)
            scientist.observe(out, t=t)
            traj.append(out)

        return spec, traj
