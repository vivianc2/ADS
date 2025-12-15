"""
world_model.py

WorldAgent: story-aware, LLM-driven interface on top of BNSimulator.

Key fixes vs earlier versions:
- Uses chat-template prompting (Qwen/Llama instruct models) so the model doesn't echo the prompt.
- Robustly extracts the *correct* JSON command object (not the giant variables-map JSON).
- Normalizes unknown variable names (e.g. "asthma history" -> "asthma_history") before adding to BN.
- Extends BN with: definition + parents + plausible CPD (binary yes/no).

Assumes you have simulator.py with BNSimulator providing:
- BNSimulator.from_bif(path)
- simulator.get_nodes() -> List[str]
- simulator.get_state_names(var) -> List[str]
- simulator.sample_observational(n, variables=None) -> pd.DataFrame
- simulator.sample_interventional(interventions, n, variables=None) -> pd.DataFrame
- simulator.model -> pgmpy BayesianNetwork/DiscreteBayesianNetwork-like object
- simulator.name -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from simulator import BNSimulator


# ----------------------------
# Helpers
# ----------------------------

def normalize_var_name(name: str) -> str:
    """
    Normalize a user-provided variable phrase into a BN-safe name.
    Examples:
      "asthma history" -> "asthma_history"
      "Air Pollution"  -> "air_pollution"
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "new_var"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------
# WorldAgent
# ----------------------------

@dataclass
class WorldAgent:
    simulator: BNSimulator
    story: str
    var_descriptions: Dict[str, str]

    # LLM config
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: Optional[str] = None
    max_new_tokens: int = 256

    # caches
    _intervention_policy: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[WorldAgent] Loading LLM: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        # Some tokenizers don't have pad token set
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ---------------------------------------------------------------------
    # LLM chat generation (IMPORTANT FIX)
    # ---------------------------------------------------------------------

    def _llm_chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Use chat template so the model doesn't echo the entire prompt.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # Heuristic: return only the suffix after the prompt (often works well)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    # ---------------------------------------------------------------------
    # World description
    # ---------------------------------------------------------------------

    def describe_world(self) -> str:
        nodes = self.simulator.get_nodes()
        lines = [self.story.strip(), "", "Variables in this world:"]
        for v in nodes:
            desc = self.var_descriptions.get(v, "(no description)")
            states = self.simulator.get_state_names(v)
            lines.append(f"- {v}: {desc} (states: {states})")
        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Main interface
    # ---------------------------------------------------------------------

    def handle(self, user_text: str) -> Dict[str, Any]:
        parsed = self._parse_request_llm(user_text)

        # Try extending world if unknown vars are present
        parsed, extension_msg = self.extend_bn_from_request(parsed, user_text)

        # Validate
        ok, msg = self.validate_query(parsed, user_text)
        if not ok:
            return {"parsed": parsed, "data": None, "response": msg}

        action = parsed["action"]

        if action == "describe":
            response = self.describe_world()
            if extension_msg:
                response = extension_msg + "\n\n" + response
            return {"parsed": parsed, "data": None, "response": response}

        if action == "sample_observational":
            df = self.simulator.sample_observational(n=parsed["n"], variables=parsed["variables"])
            response = self._format_samples(df, parsed, interventional=False)
            if extension_msg:
                response = extension_msg + "\n\n" + response
            return {"parsed": parsed, "data": df, "response": response}

        if action == "sample_interventional":
            df = self.simulator.sample_interventional(
                interventions=parsed["interventions"] or {},
                n=parsed["n"],
                variables=parsed["variables"],
            )
            response = self._format_samples(df, parsed, interventional=True)
            if extension_msg:
                response = extension_msg + "\n\n" + response
            return {"parsed": parsed, "data": df, "response": response}

        raise ValueError(f"Unknown action: {action}")

    # ---------------------------------------------------------------------
    # JSON extraction (IMPORTANT FIX)
    # ---------------------------------------------------------------------

    def _extract_all_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract ALL JSON objects from text using brace matching.
        This is more reliable than regex for nested objects.
        """
        objs: List[Dict[str, Any]] = []
        stack = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start is not None:
                        chunk = text[start:i+1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                objs.append(obj)
                        except Exception:
                            pass
                        start = None
        return objs

    def _is_command_schema(self, obj: Dict[str, Any]) -> bool:
        """
        Command schema MUST contain exactly these core keys (extra keys allowed but discouraged).
        We use this to avoid grabbing the huge variables-map dict by mistake.
        """
        required = {"action", "n", "variables", "interventions", "unknown_variables", "unknown_interventions"}
        if not required.issubset(set(obj.keys())):
            return False
        if obj.get("action") not in {"describe", "sample_observational", "sample_interventional"}:
            return False
        if not isinstance(obj.get("n"), int):
            return False
        # variables: list or None
        if obj.get("variables") is not None and not isinstance(obj.get("variables"), list):
            return False
        # interventions: dict or None
        if obj.get("interventions") is not None and not isinstance(obj.get("interventions"), dict):
            return False
        if not isinstance(obj.get("unknown_variables"), list):
            return False
        if not isinstance(obj.get("unknown_interventions"), dict):
            return False
        return True

    # ---------------------------------------------------------------------
    # Parsing via LLM
    # ---------------------------------------------------------------------

    def _parse_request_llm(self, text: str) -> Dict[str, Any]:
        valid_vars = self.simulator.get_nodes()
        state_names = {v: self.simulator.get_state_names(v) for v in valid_vars}

        system = (
            "You are a strict JSON-only parser for a Bayesian network simulator.\n"
            "You must output ONLY one JSON object that matches the exact schema.\n"
            "No extra commentary, no multiple candidates.\n"
        )

        # Keep this prompt SHORT. Long prompts make small models drift.
        # Also: we include examples to anchor behavior.
        user = f"""
World story (brief):
{self.story.strip()[:600]}

Known variables (names only):
{valid_vars}

State names:
{json.dumps(state_names, indent=2)}

User request:
{text.strip()}

Output exactly one JSON object with this schema:
{{
  "action": "describe" | "sample_observational" | "sample_interventional",
  "n": int,
  "variables": [string] | null,
  "interventions": object | null,
  "unknown_variables": [string],
  "unknown_interventions": object
}}

Rules:
- If request includes do(...), action must be "sample_interventional". Otherwise "sample_observational",
  unless explicitly asking to describe the world.
- variables should include ONLY known variable names; unknown phrases go in unknown_variables.
- interventions should include ONLY known variables with valid states; otherwise put them in unknown_interventions.
- If n not specified, use 10.
- Output ONLY the JSON object.

Examples:
1) "give me 5 samples of lung and dysp"
-> {{
  "action":"sample_observational","n":5,"variables":["lung","dysp"],
  "interventions":null,"unknown_variables":[],"unknown_interventions":{{}}
}}

2) "do(smoke=yes) give me 15 samples of lung and dysp"
-> {{
  "action":"sample_interventional","n":15,"variables":["lung","dysp"],
  "interventions":{{"smoke":"yes"}},"unknown_variables":[],"unknown_interventions":{{}}
}}

Now produce the JSON for the user request.
""".strip()

        out = self._llm_chat(system=system, user=user, max_new_tokens=256)

        # Extract json objects and pick the one matching schema
        objs = self._extract_all_json_objects(out)
        for obj in objs:
            if self._is_command_schema(obj):
                return self._sanitize_cmd(obj, valid_vars, state_names)

        # If we didn't find a command object, show the raw output for debugging
        raise ValueError(
            "Could not extract a valid JSON command from LLM output.\n"
            f"LLM output was:\n{out}"
        )

    def _sanitize_cmd(
        self,
        cmd: Dict[str, Any],
        valid_vars: List[str],
        state_names: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        # Normalize n
        if not isinstance(cmd.get("n"), int) or cmd["n"] <= 0:
            cmd["n"] = 10

        # Variables
        vars_in = cmd.get("variables", None)
        unknown_vars = cmd.get("unknown_variables", [])

        if vars_in is None:
            known_vars = None
        else:
            known_vars = [v for v in vars_in if v in valid_vars]
            # any not known -> unknown
            for v in vars_in:
                if v not in valid_vars and v not in unknown_vars:
                    unknown_vars.append(v)

        cmd["variables"] = known_vars

        # Interventions
        ints_in = cmd.get("interventions", None) or {}
        unknown_ints = cmd.get("unknown_interventions", {}) or {}

        known_ints: Dict[str, str] = {}
        for k, v in ints_in.items():
            if k in valid_vars and v in state_names.get(k, []):
                known_ints[k] = v
            else:
                unknown_ints[k] = v

        cmd["interventions"] = known_ints if known_ints else None
        cmd["unknown_variables"] = unknown_vars
        cmd["unknown_interventions"] = unknown_ints

        # Ensure keys exist
        cmd.setdefault("unknown_variables", [])
        cmd.setdefault("unknown_interventions", {})
        cmd.setdefault("interventions", None)
        cmd.setdefault("variables", None)

        return cmd

    # ---------------------------------------------------------------------
    # Query validation
    # ---------------------------------------------------------------------

    def validate_query(self, parsed: Dict[str, Any], user_text: str) -> Tuple[bool, str]:
        # unknown vars still there -> not extendable or not extended
        if parsed.get("unknown_variables") or parsed.get("unknown_interventions"):
            return False, (
                "In this world, I don't yet know how to interpret some of the variables you mentioned: "
                f"variables={parsed.get('unknown_variables', [])}, "
                f"intervention targets={list((parsed.get('unknown_interventions') or {}).keys())}. "
                "I couldn't safely extend the world to include them."
            )

        # intervention realism policy (LLM-decided)
        if parsed["action"] == "sample_interventional" and parsed.get("interventions"):
            policy = self._get_intervention_policy()
            for var in parsed["interventions"].keys():
                if policy.get(var) == "forbid":
                    return False, (
                        f"In this world, it's not realistic to directly intervene on '{var}'. "
                        "You can observe it, or intervene on upstream controllable factors."
                    )

        return True, ""

    def _get_intervention_policy(self) -> Dict[str, str]:
        if self._intervention_policy:
            return self._intervention_policy

        vars_ = self.simulator.get_nodes()
        descs = {v: self.var_descriptions.get(v, "") for v in vars_}

        system = "You are a careful realism judge. Output ONLY JSON."
        user = f"""
World story:
{self.story}

Variables:
{json.dumps(descs, indent=2)}

Classify each variable as one of:
- "allow" (reasonable to intervene on directly),
- "forbid" (not directly controllable / unethical / impossible),
- "diagnostic_only" (usually observed/measured, not intervened).

Return ONLY JSON mapping variable -> label.
""".strip()

        out = self._llm_chat(system=system, user=user, max_new_tokens=256)
        objs = self._extract_all_json_objects(out)
        policy = None
        for obj in objs:
            if isinstance(obj, dict) and all(k in vars_ for k in obj.keys()):
                policy = obj
                break

        if policy is None:
            policy = {v: "allow" for v in vars_}

        clean: Dict[str, str] = {}
        for v in vars_:
            lbl = policy.get(v, "allow")
            if lbl not in {"allow", "forbid", "diagnostic_only"}:
                lbl = "allow"
            clean[v] = lbl

        self._intervention_policy = clean
        return clean

    # ---------------------------------------------------------------------
    # New variable proposal (definition + parents + CPD)
    # ---------------------------------------------------------------------

    def _propose_new_variable_spec(self, raw_name: str) -> Optional[Dict[str, Any]]:
        """
        raw_name can contain spaces. We'll normalize it, but keep raw_name for semantics.
        """
        var_name = normalize_var_name(raw_name)

        known_vars = self.simulator.get_nodes()
        state_names = {v: self.simulator.get_state_names(v) for v in known_vars}
        descs = {v: self.var_descriptions.get(v, "") for v in known_vars}

        system = "You are expanding a Bayesian-network world. Output ONLY JSON."

        user = f"""
World story:
{self.story}

Existing variables (name -> description, states):
{json.dumps({v: {"description": descs[v], "states": state_names[v]} for v in known_vars}, indent=2)}

User asked about a new concept: "{raw_name}"
We will name the BN variable: "{var_name}"

Decide if this is plausible in this chest-clinic world.
If valid:
- Give a clear definition (1-2 sentences).
- Choose 0-2 parents from existing variables (causal influences).
- Provide a CPD as probabilities P({var_name}="yes" | parents) for EVERY parent combination.
- Variable is binary states ["yes","no"].
- Use parent-config keys exactly like:
  "()" for no parents
  "(smoke=yes)"
  "(smoke=yes, bronc=no)" etc. in the same parent order as your "parents" list.

Return ONLY JSON:
{{
  "valid": true/false,
  "name": "{var_name}",
  "raw_name": "{raw_name}",
  "description": "...",
  "parents": ["..."],
  "default_yes_prob": 0.3,
  "cpd": {{
    "()": 0.2
  }}
}}
If invalid, return:
{{"valid": false, "name":"{var_name}", "raw_name":"{raw_name}"}}
""".strip()

        out = self._llm_chat(system=system, user=user, max_new_tokens=512)
        objs = self._extract_all_json_objects(out)

        spec = None
        for obj in objs:
            if isinstance(obj, dict) and obj.get("name") == var_name and "valid" in obj:
                spec = obj
                break
        if spec is None or not spec.get("valid", False):
            return None

        # sanitize fields
        spec.setdefault("description", f"New variable derived from '{raw_name}'.")
        spec.setdefault("parents", [])
        spec.setdefault("default_yes_prob", 0.5)
        spec.setdefault("cpd", {})

        # parents must be subset of known
        parents = [p for p in spec["parents"] if p in known_vars]
        spec["parents"] = parents[:2]

        try:
            spec["default_yes_prob"] = float(spec["default_yes_prob"])
        except Exception:
            spec["default_yes_prob"] = 0.5
        spec["default_yes_prob"] = clamp01(spec["default_yes_prob"])

        # CPD parse
        clean_cpd = {}
        for k, v in (spec.get("cpd") or {}).items():
            try:
                clean_cpd[str(k)] = clamp01(float(v))
            except Exception:
                continue
        spec["cpd"] = clean_cpd

        return spec

    # ---------------------------------------------------------------------
    # BN extension (actual graph update)
    # ---------------------------------------------------------------------

    def extend_bn_from_request(self, parsed: Dict[str, Any], user_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Extend BN if unknown variables are requested.
        - Normalize unknown names.
        - Ask LLM for definition+parents+CPD.
        - Add node+edges+cpd.
        - Update parsed so the original request proceeds.
        """
        from itertools import product
        from pgmpy.factors.discrete import TabularCPD

        unknown_vars_raw: List[str] = parsed.get("unknown_variables") or []
        unknown_ints: Dict[str, Any] = parsed.get("unknown_interventions") or {}

        candidates_raw = list(dict.fromkeys(unknown_vars_raw + list(unknown_ints.keys())))
        if not candidates_raw:
            return parsed, ""

        model = self.simulator.model
        added_info: List[Dict[str, Any]] = []
        added_names: List[str] = []

        for raw in candidates_raw:
            new_name = normalize_var_name(raw)
            if new_name in model.nodes():
                # already exists, treat as known now
                continue

            spec = self._propose_new_variable_spec(raw)
            if spec is None:
                continue

            name = spec["name"]
            description = spec["description"]
            parents = spec["parents"]
            default_yes = spec["default_yes_prob"]
            cpd_map = spec["cpd"]

            # add node
            model.add_node(name)

            # add edges parent -> name
            for p in parents:
                model.add_edge(p, name)

            # build CPD
            parent_states = [self.simulator.get_state_names(p) for p in parents]
            p_yes_list: List[float] = []

            if not parents:
                p_yes = cpd_map.get("()", default_yes)
                p_yes_list = [clamp01(float(p_yes))]
            else:
                for combo in product(*parent_states):
                    key = "(" + ", ".join(f"{p}={s}" for p, s in zip(parents, combo)) + ")"
                    p_yes = cpd_map.get(key, default_yes)
                    p_yes_list.append(clamp01(float(p_yes)))

            p_no_list = [1.0 - p for p in p_yes_list]

            if not parents:
                cpd = TabularCPD(
                    variable=name,
                    variable_card=2,
                    values=[p_yes_list, p_no_list],
                    state_names={name: ["yes", "no"]},
                )
            else:
                cpd = TabularCPD(
                    variable=name,
                    variable_card=2,
                    values=[p_yes_list, p_no_list],
                    evidence=parents,
                    evidence_card=[len(s) for s in parent_states],
                    state_names={name: ["yes", "no"], **{p: self.simulator.get_state_names(p) for p in parents}},
                )

            model.add_cpds(cpd)

            # update world semantics
            self.var_descriptions[name] = f"{description} (user phrase: '{raw}')"
            added_names.append(name)
            added_info.append(
                {"name": name, "raw_name": raw, "description": self.var_descriptions[name],
                 "parents": parents, "default_yes_prob": default_yes}
            )

        if not added_names:
            # could not extend; leave parsed unknowns intact so validate_query will explain
            return parsed, ""

        # rebuild simulator
        self.simulator = BNSimulator(model=model, name=self.simulator.name + "+extended")

        # Update parsed to proceed with the original sampling request:
        # - remove added from unknown_variables
        # - include added names in variables if user requested them
        parsed["added_variables_info"] = added_info

        # Remove those raw names from unknown_variables (best effort)
        new_unknown_vars = []
        for raw in parsed.get("unknown_variables", []):
            nn = normalize_var_name(raw)
            if nn not in added_names:
                new_unknown_vars.append(raw)
        parsed["unknown_variables"] = new_unknown_vars

        # Move unknown interventions if now known
        new_unknown_ints = {}
        interventions = parsed.get("interventions") or {}
        for raw_k, v in (parsed.get("unknown_interventions") or {}).items():
            nk = normalize_var_name(raw_k)
            if nk in added_names:
                # new var is binary yes/no; coerce value
                vv = str(v).strip().lower()
                vv = vv if vv in {"yes", "no"} else "yes"
                interventions[nk] = vv
            else:
                new_unknown_ints[raw_k] = v
        parsed["interventions"] = interventions if interventions else None
        parsed["unknown_interventions"] = new_unknown_ints

        # If user requested variables list, add any newly created names if they were mentioned
        if parsed.get("variables") is not None:
            vars_now = list(parsed["variables"])
            for raw in candidates_raw:
                nn = normalize_var_name(raw)
                if nn in added_names and nn not in vars_now:
                    vars_now.append(nn)
            parsed["variables"] = vars_now

        # Build message
        msg_lines = ["I updated the world by adding new variables:"]
        for info in added_info:
            msg_lines.append(
                f"- {info['name']}: {info['description']} (parents={info['parents']}, baseline P(yes)={info['default_yes_prob']:.2f})"
            )
        return parsed, "\n".join(msg_lines)

    # ---------------------------------------------------------------------
    # Formatting responses
    # ---------------------------------------------------------------------

    def _format_samples(self, df: pd.DataFrame, parsed: Dict[str, Any], interventional: bool) -> str:
        n = parsed.get("n", len(df))
        n_show = min(n, 10)

        if interventional:
            interventions = parsed.get("interventions") or {}
            interv_str = ", ".join(f"{k}={v}" for k, v in interventions.items())
            intro = f"Here are {n} simulated cases under do({interv_str})."
        else:
            intro = f"Here are {n} observational simulated cases."

        vars_shown = parsed["variables"] if parsed.get("variables") is not None else list(df.columns)
        table_str = df[vars_shown].head(n_show).to_string(index=False)
        return intro + "\n\n" + table_str + f"\n\nShowing first {n_show} rows."

