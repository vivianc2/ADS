"""
world_model.py

A story-aware, LLM-driven "World Agent" interface on top of a Bayesian-network simulator.

Key guarantees (per your requirements):
- Always uses an LLM for parsing / decisions. No heuristic fallback parsing.
- Uses Qwen/Qwen2.5-3B-Instruct by default.
- If no LLM backend is provided (and loading is disabled), raises a clear error.

Recommended usage:

from simulator import BNSimulator
from world_model import WorldAgent, HFChatLLM

sim = BNSimulator.from_bif("asia.bif")
llm = HFChatLLM(model_name="Qwen/Qwen2.5-3B-Instruct")
world = WorldAgent(simulator=sim, story="Chest clinic ...", var_descriptions=..., llm=llm)

resp = world.handle("Give me 50 observational samples of dysp and smoke.")
print(resp.narrative)
print(resp.data.head())

Notes:
- This module is intentionally strict: if the LLM outputs invalid JSON or violates the schema, it raises.
- Mechanical validation is deterministic (node exists, state valid, n positive, etc.).
- BN extension (optional) is atomic: build candidate model via deepcopy, check_model(), then swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
import itertools
import json
import re

import pandas as pd

# HuggingFace / torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# pgmpy
from pgmpy.factors.discrete import TabularCPD

# Local
from simulator import BNSimulator


# -----------------------------
# Exceptions
# -----------------------------

class WorldModelError(RuntimeError):
    """Base error for world_model failures."""


class LLMNotProvidedError(WorldModelError):
    """Raised when no LLM backend is available but required."""


class LLMParseError(WorldModelError):
    """Raised when the LLM output can't be parsed into the required JSON schema."""
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class RequestValidationError(WorldModelError):
    """Raised when a parsed request is mechanically invalid (node/state/n/etc.)."""


class PolicyError(WorldModelError):
    """Raised when a request is disallowed by world policy/realism."""


class BNExtensionError(WorldModelError):
    """Raised when BN extension fails or produces an invalid BN."""


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class ParsedCommand:
    action: str  # "observational" | "interventional"
    n: int
    variables: Optional[List[str]]  # None means "all variables"
    interventions: Dict[str, str]   # empty if observational
    # optional fields for logging/debug
    notes: Optional[str] = None


@dataclass(frozen=True)
class WorldResponse:
    parsed: ParsedCommand
    data: pd.DataFrame
    narrative: str
    added_variables: List[Dict[str, Any]] = field(default_factory=list)  # extension info, if any


# -----------------------------
# LLM backend (HuggingFace Qwen2.5 Instruct)
# -----------------------------

@dataclass
class HFChatLLM:
    """
    Minimal chat-style wrapper for HuggingFace instruct models.
    Strictly returns only newly generated tokens (not the prompt).
    """
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None

    # generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95

    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    _device: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        if self.torch_dtype is None:
            # Reasonable default: fp16 on cuda, fp32 on cpu
            self.torch_dtype = torch.float16 if self._device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self._device)
        self.model.eval()

    def chat(self, system: str, user: str, *, max_new_tokens: Optional[int] = None) -> str:
        """
        Returns assistant text (new tokens only). No fallback transformations.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Qwen chat templates generally support apply_chat_template.
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = enc.to(self._device)
        attention_mask = torch.ones_like(input_ids)            

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (critical for reliable JSON extraction)
        new_tokens = out_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text


# -----------------------------
# WorldAgent
# -----------------------------

@dataclass
class WorldAgent:
    """
    Story-aware, policy-aware interface on top of BNSimulator.

    Requirements enforced:
    - Always uses LLM to parse the user's natural-language request.
    - If llm is None: error.
    """
    simulator: BNSimulator
    story: str
    var_descriptions: Dict[str, str]
    llm: HFChatLLM

    # policy/guardrails
    allow_interventions: bool = True
    # optional per-variable intervention policy:
    #   "allow" | "deny" | "ask"  (ask -> disallow unless user explicitly requests and LLM justifies)
    intervention_policy: Dict[str, str] = field(default_factory=dict)

    # extension (default to allow)
    allow_extension: bool = True
    max_extension_vars_per_request: int = 2

    # sampling limits
    max_samples: int = 5000

    def __post_init__(self) -> None:
        if self.llm is None:
            raise LLMNotProvidedError("No LLM backend provided. WorldAgent requires an LLM (e.g., HFChatLLM).")

    # ---------- Public API ----------

    def handle(self, user_query: str, *, seed: Optional[int] = None) -> WorldResponse:
        """
        Main entry point: parse -> validate policy -> (optional extend) -> sample -> narrate.
        """
        parsed = self._llm_parse_command(user_query)
        self._validate_mechanical(parsed)

        # policy checks
        self._validate_policy(parsed, user_query)

        added: List[Dict[str, Any]] = []
        unknown_vars = self._unknown_variables(parsed)
        if unknown_vars:
            if not self.allow_extension:
                raise RequestValidationError(
                    f"Unknown variables requested: {unknown_vars}. Extension is disabled."
                )
            if len(unknown_vars) > self.max_extension_vars_per_request:
                raise RequestValidationError(
                    f"Too many unknown variables requested ({len(unknown_vars)}). "
                    f"Limit is {self.max_extension_vars_per_request}."
                )
            added = self._extend_bn_atomically(unknown_vars, user_query)

            # After extension, re-validate mechanically (now nodes should exist)
            parsed = self._llm_parse_command(user_query)  # re-parse to allow LLM to include new vars cleanly
            self._validate_mechanical(parsed)

        # execute
        df = self._execute(parsed, seed=seed)

        # narrate (LLM-generated, no fallback)
        narrative = self._llm_narrate(user_query, parsed, df, added)

        return WorldResponse(parsed=parsed, data=df, narrative=narrative, added_variables=added)

    # ---------- LLM: strict JSON parsing ----------

    def _extract_tagged_json(self, text: str) -> Dict[str, Any]:
        """
        Strictly require <json> ... </json>.
        No fallback extraction.
        """
        m = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, flags=re.DOTALL)
        if not m:
            raise LLMParseError("LLM did not return <json>...</json> as required.", raw_output=text)
        payload = m.group(1)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            raise LLMParseError(f"Invalid JSON inside <json> tag: {e}", raw_output=text) from e

    def _llm_parse_command(self, user_query: str) -> ParsedCommand:
        """
        One-call parse: action, n, variables, interventions.
        Always uses LLM. No heuristic parsing.
        """
        nodes = self.simulator.get_nodes()
        # Keep descriptions short to avoid prompt bloat
        desc_lines = []
        for v in nodes:
            d = self.var_descriptions.get(v, "").strip()
            if d:
                d = re.sub(r"\s+", " ", d)
                if len(d) > 120:
                    d = d[:117] + "..."
                desc_lines.append(f"- {v}: {d}")
            else:
                desc_lines.append(f"- {v}")
        var_catalog = "\n".join(desc_lines)

        system = (
            "You are a scientific world-interface that converts natural-language experiment requests "
            "into a strict JSON command for a Bayesian-network simulator. "
            "You MUST output ONLY a single <json>...</json> block and nothing else."
        )

        user = f"""
STORY CONTEXT:
{self.story}

AVAILABLE VARIABLES (use these exact names when possible):
{var_catalog}

TASK:
Parse the user's request into a command with this schema:

<json>{{
  "action": "observational" | "interventional",
  "n": integer,                       // number of samples
  "variables": null | [string, ...],  // null means return all variables; otherwise list variable names
  "interventions": {{string: string}} // empty object if none; values are state names
}}</json>

RULES:
- If the user requests an intervention / do() / forced setting, use action="interventional".
- If there are no interventions, use action="observational" and interventions={{}}.
- If user does not specify n, choose a reasonable default n=50.
- If user does not specify which variables to return, set variables=null (all).
- If user requests variables not in the catalog, still include them verbatim in "variables"
  (do NOT invent synonyms). Do not add extra keys.

USER REQUEST:
{user_query}
""".strip()

        out = self.llm.chat(system=system, user=user)
        obj = self._extract_tagged_json(out)

        # strict schema validation (deterministic)
        if not isinstance(obj, dict):
            raise LLMParseError("Parsed JSON is not an object.", raw_output=out)

        action = obj.get("action")
        n = obj.get("n")
        variables = obj.get("variables")
        interventions = obj.get("interventions")
        if interventions is None:
            interventions = {}

        if action not in ("observational", "interventional"):
            raise LLMParseError("Field 'action' must be 'observational' or 'interventional'.", raw_output=out)
        if not isinstance(n, int):
            raise LLMParseError("Field 'n' must be an integer.", raw_output=out)
        if variables is not None:
            if not (isinstance(variables, list) and all(isinstance(x, str) for x in variables)):
                raise LLMParseError("Field 'variables' must be null or a list of strings.", raw_output=out)
        if not (isinstance(interventions, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in interventions.items())):
            raise LLMParseError("Field 'interventions' must be an object {string: string}.", raw_output=out)

        return ParsedCommand(
            action=action,
            n=n,
            variables=variables,
            interventions=interventions,
        )

    def _llm_narrate(
        self,
        user_query: str,
        parsed: ParsedCommand,
        df: pd.DataFrame,
        added: List[Dict[str, Any]],
    ) -> str:
        """
        LLM-generated narrative. No fallback. Keeps table content summarized.
        """
        # small summary stats to avoid dumping the whole table
        preview = df.head(8).to_markdown(index=False)

        system = (
            "You are the World Agent narrating scientific sampling results in the story context. "
            "Be concise, grounded, and do not fabricate data beyond what is shown."
        )

        user = f"""
STORY CONTEXT:
{self.story}

USER REQUEST:
{user_query}

PARSED COMMAND:
{{
  "action": "{parsed.action}",
  "n": {parsed.n},
  "variables": {json.dumps(parsed.variables)},
  "interventions": {json.dumps(parsed.interventions)}
}}

ADDED VARIABLES (if any):
{json.dumps(added, indent=2)}

SAMPLE PREVIEW (first rows):
{preview}

TASK:
Write a short narrative response:
- 2-6 sentences.
- Mention whether this is observational or interventional.
- If interventional, mention the do() setting(s).
- Avoid claiming anything not supported by the table preview.
""".strip()

        out = self.llm.chat(system=system, user=user, max_new_tokens=256)
        if not out.strip():
            raise LLMParseError("LLM returned empty narrative.", raw_output=out)
        return out.strip()

    # ---------- Validation & policy ----------

    def _validate_mechanical(self, cmd: ParsedCommand) -> None:
        if cmd.n <= 0:
            raise RequestValidationError("n must be positive.")
        if cmd.n > self.max_samples:
            raise RequestValidationError(f"n too large ({cmd.n}). Max is {self.max_samples}.")

        if cmd.action == "observational":
            if cmd.interventions:
                raise RequestValidationError("Observational action cannot include interventions.")
        elif cmd.action == "interventional":
            # if not cmd.interventions:
            #     raise RequestValidationError("Interventional action requires non-empty interventions.")
            pass

        # mechanical checks for known nodes / states (unknown handled separately)
        nodes = set(self.simulator.get_nodes())

        # interventions: if node exists, state must be valid
        for var, state in cmd.interventions.items():
            if var in nodes:
                valid_states = set(self.simulator.get_state_names(var))
                if state not in valid_states:
                    raise RequestValidationError(
                        f"Invalid state '{state}' for variable '{var}'. Valid: {sorted(valid_states)}"
                    )

        # variables: if node exists, ok; if unknown, may be extended later
        if cmd.variables is not None:
            for v in cmd.variables:
                if not isinstance(v, str) or not v.strip():
                    raise RequestValidationError("Variable names must be non-empty strings.")

    def _validate_policy(self, cmd: ParsedCommand, user_query: str) -> None:
        # Global interventions toggle
        if cmd.action == "interventional" and not self.allow_interventions:
            raise PolicyError("Interventions are not allowed in this world.")

        # Per-variable intervention policy (deterministic)
        if cmd.action == "interventional":
            for var in cmd.interventions.keys():
                rule = self.intervention_policy.get(var, "allow")
                if rule == "deny":
                    raise PolicyError(f"Intervention on '{var}' is disallowed by world policy.")
                if rule == "ask":
                    # Still deterministic: disallow unless you remove this mode or implement an explicit
                    # 'LLM justification' flow. For now, enforce strictness.
                    raise PolicyError(
                        f"Intervention on '{var}' requires special approval (policy='ask'), currently disallowed."
                    )

    def _unknown_variables(self, cmd: ParsedCommand) -> List[str]:
        nodes = set(self.simulator.get_nodes())
        unknown: List[str] = []

        # requested output variables
        if cmd.variables is not None:
            for v in cmd.variables:
                if v not in nodes:
                    unknown.append(v)

        # intervention variables
        for v in cmd.interventions.keys():
            if v not in nodes and v not in unknown:
                unknown.append(v)

        return unknown

    # ---------- Execution ----------

    def _execute(self, cmd: ParsedCommand, *, seed: Optional[int] = None) -> pd.DataFrame:
        variables: Optional[Sequence[str]] = cmd.variables  # None means all variables
        if cmd.action == "observational":
            return self.simulator.sample_observational(n=cmd.n, variables=variables, seed=seed)
        return self.simulator.sample_interventional(interventions=cmd.interventions, n=cmd.n, variables=variables, seed=seed)

    # ---------- BN extension (atomic) ----------

    def _extend_bn_atomically(self, unknown_vars: List[str], user_query: str) -> List[Dict[str, Any]]:
        """
        Ask the LLM to propose new variables, then atomically swap the simulator if valid.
        This is STRICT: if anything is missing/invalid, extension fails.
        """
        base_nodes = self.simulator.get_nodes()
        base_edges = self.simulator.get_edges()

        # Keep prompt small but informative
        catalog = "\n".join([f"- {v}: {self.var_descriptions.get(v, '').strip()}" for v in base_nodes])

        system = (
            "You are designing an extension to a discrete Bayesian network for a simulated world. "
            "You MUST output ONLY a single <json>...</json> block and nothing else."
        )

        user = f"""
STORY CONTEXT:
{self.story}

CURRENT VARIABLES:
{catalog}

CURRENT EDGES:
{base_edges}

UNKNOWN CONCEPTS REQUESTED BY USER:
{unknown_vars}

USER REQUEST (for context):
{user_query}

TASK:
For EACH unknown concept, propose a NEW binary variable with states ["no", "yes"] that is plausible
in the story and connect it to existing variables with a small set of parents.

Return this schema:

<json>{{
  "new_variables": [
    {{
      "name": string,                 // must exactly match one of unknown concepts
      "description": string,
      "states": ["no", "yes"],
      "parents": [string, ...],       // subset of CURRENT VARIABLES; may be empty
      "cpd_yes": [
        {{
          "given": {{parent: state, ...}},   // assignment for ALL parents
          "p_yes": number                   // 0..1
        }},
        ...
      ]
    }},
    ...
  ]
}}</json>

RULES:
- "name" must exactly be one of the UNKNOWN CONCEPTS (verbatim).
- "parents" should be small (0-3).
- "cpd_yes" MUST cover every combination of parent states (full table).
- Use only existing state names for parents, consistent with CURRENT VARIABLES.
- Do not add extra keys.
""".strip()

        out = self.llm.chat(system=system, user=user, max_new_tokens=900)
        obj = self._extract_tagged_json(out)

        if not isinstance(obj, dict) or "new_variables" not in obj:
            raise LLMParseError("Extension JSON must contain key 'new_variables'.", raw_output=out)
        new_vars = obj["new_variables"]
        if not (isinstance(new_vars, list) and all(isinstance(x, dict) for x in new_vars)):
            raise LLMParseError("'new_variables' must be a list of objects.", raw_output=out)

        # Map proposals by name
        proposals: Dict[str, Dict[str, Any]] = {}
        for p in new_vars:
            name = p.get("name")
            if not isinstance(name, str) or name not in unknown_vars:
                raise BNExtensionError(f"Extension proposal has invalid name: {name}")
            proposals[name] = p

        # Ensure every unknown var is covered
        missing = [u for u in unknown_vars if u not in proposals]
        if missing:
            raise BNExtensionError(f"LLM did not propose specs for: {missing}")

        # Build candidate BN
        candidate_model = deepcopy(self.simulator.model)

        added_info: List[Dict[str, Any]] = []
        for name in unknown_vars:
            spec = proposals[name]
            self._apply_one_extension(candidate_model, spec)
            added_info.append(
                {
                    "name": name,
                    "description": spec.get("description", ""),
                    "parents": spec.get("parents", []),
                    "states": spec.get("states", ["no", "yes"]),
                }
            )

        # Validate and swap atomically
        try:
            candidate_model.check_model()
        except Exception as e:
            raise BNExtensionError(f"Extended BN failed check_model(): {e}") from e

        # Swap simulator
        self.simulator = BNSimulator(model=candidate_model, name=getattr(self.simulator, "name", None))

        # Update descriptions
        for info in added_info:
            n = info["name"]
            d = info.get("description", "").strip()
            if d:
                self.var_descriptions[n] = d

        return added_info

    def _apply_one_extension(self, model: Any, spec: Dict[str, Any]) -> None:
        """
        Mutates the provided pgmpy model (candidate copy) by adding one variable + edges + CPD.
        """
        name = spec.get("name")
        description = spec.get("description")
        states = spec.get("states")
        parents = spec.get("parents")
        cpd_yes = spec.get("cpd_yes")

        if not isinstance(name, str) or not name.strip():
            raise BNExtensionError("New variable 'name' must be a non-empty string.")
        if name in model.nodes():
            raise BNExtensionError(f"Variable '{name}' already exists in the BN.")
        if not (isinstance(states, list) and states == ["no", "yes"]):
            raise BNExtensionError("New variable must be binary with states ['no','yes'].")
        if parents is None:
            parents = []
        if not (isinstance(parents, list) and all(isinstance(p, str) for p in parents)):
            raise BNExtensionError("'parents' must be a list of strings.")
        for p in parents:
            if p not in model.nodes():
                raise BNExtensionError(f"Parent '{p}' does not exist in the BN.")

        # Add node + edges
        model.add_node(name)
        for p in parents:
            model.add_edge(p, name)

        # Build CPD
        cpd = self._build_binary_cpd_from_rows(
            var=name,
            parents=parents,
            parent_states={p: self.simulator.get_state_names(p) for p in parents},
            rows=cpd_yes,
        )

        # Add CPD
        model.add_cpds(cpd)

    def _build_binary_cpd_from_rows(
        self,
        *,
        var: str,
        parents: List[str],
        parent_states: Dict[str, List[str]],
        rows: Any,
    ) -> TabularCPD:
        """
        Build a pgmpy TabularCPD for a binary variable with states ["no","yes"].
        The LLM supplies "p_yes" rows keyed by complete parent assignments.
        """
        if parents is None:
            parents = []

        if not parents:
            # No parents: expect either empty rows or one row with given={}
            if rows is None:
                raise BNExtensionError(f"cpd_yes missing for '{var}'.")
            if not isinstance(rows, list):
                raise BNExtensionError(f"cpd_yes must be a list for '{var}'.")
            p_yes = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                given = r.get("given", {})
                if given == {}:
                    p_yes = r.get("p_yes")
                    break
            if not isinstance(p_yes, (int, float)) or not (0.0 <= float(p_yes) <= 1.0):
                raise BNExtensionError(f"cpd_yes must include one row with given={{}} and valid p_yes for '{var}'.")
            values = [[1.0 - float(p_yes)], [float(p_yes)]]
            return TabularCPD(variable=var, variable_card=2, values=values, state_names={var: ["no", "yes"]})

        # With parents: need full table
        if rows is None or not isinstance(rows, list):
            raise BNExtensionError(f"cpd_yes must be a non-empty list for '{var}' with parents.")

        # Build map from assignment tuple -> p_yes
        assign_to_p: Dict[Tuple[str, ...], float] = {}
        parent_order = list(parents)

        for r in rows:
            if not isinstance(r, dict):
                raise BNExtensionError(f"Invalid cpd_yes row (not an object) for '{var}'.")
            given = r.get("given")
            p_yes = r.get("p_yes")
            if not isinstance(given, dict):
                raise BNExtensionError(f"Row missing 'given' dict for '{var}'.")
            if not isinstance(p_yes, (int, float)) or not (0.0 <= float(p_yes) <= 1.0):
                raise BNExtensionError(f"Row has invalid p_yes for '{var}'.")
            # must specify ALL parents
            for p in parent_order:
                if p not in given:
                    raise BNExtensionError(f"Row missing parent '{p}' assignment in cpd_yes for '{var}'.")
                if given[p] not in parent_states[p]:
                    raise BNExtensionError(
                        f"Invalid state '{given[p]}' for parent '{p}' in cpd_yes for '{var}'. "
                        f"Valid: {parent_states[p]}"
                    )
            key = tuple(given[p] for p in parent_order)
            assign_to_p[key] = float(p_yes)

        # Determine column order: cartesian product of parent states in parent_order,
        # with last parent varying fastest (itertools.product does exactly that).
        combos = list(itertools.product(*[parent_states[p] for p in parent_order]))
        missing = [c for c in combos if c not in assign_to_p]
        if missing:
            raise BNExtensionError(
                f"cpd_yes does not cover all parent state combinations for '{var}'. Missing examples: {missing[:5]}"
            )

        p_yes_cols = [assign_to_p[c] for c in combos]
        p_no_cols = [1.0 - p for p in p_yes_cols]

        values = [p_no_cols, p_yes_cols]
        state_names = {var: ["no", "yes"]}
        for p in parent_order:
            state_names[p] = parent_states[p]

        return TabularCPD(
            variable=var,
            variable_card=2,
            values=values,
            evidence=parent_order,
            evidence_card=[len(parent_states[p]) for p in parent_order],
            state_names=state_names,
        )
