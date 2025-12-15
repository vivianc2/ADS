# llm_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LLMParser(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    LLM-backed natural language -> JSON command parser.
    Uses an open-source instruct model (Llama 3.1 8B Instruct by default).
    """
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.0  # deterministic JSON
    top_p: float = 0.9

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.to(self.device)

    def parse(
        self,
        user_text: str,
        valid_variables: List[str],
        state_names: Dict[str, List[str]],
        default_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Return a structured command dict.
        Command schema:
          - action: "describe" | "sample_observational" | "sample_interventional"
          - n: int
          - variables: list[str] | null
          - interventions: dict[var -> state] | null
        """
        system_prompt = self._build_system_prompt(valid_variables, state_names, default_n)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Llama 3 uses chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Extract the last JSON object in the output
        cmd = self._extract_json(text)
        cmd = self._sanitize(cmd, valid_variables, state_names, default_n)
        return cmd

    # ---------- prompt + utilities ----------
    def _build_system_prompt(
        self,
        valid_variables: List[str],
        state_names: Dict[str, List[str]],
        default_n: int
    ) -> str:
        return f"""
    You are WorldAgent, a narrator with access to a simulator.
    You live in a world defined by a Bayesian network with named variables and CPDs.
    You know the true causal graph and parameters.

    When the user asks for data, you must:
    - translate the request into a sampling command,
    - output a JSON command matching the schema below,
    - call the simulator (handled outside your output),
    - return only a JSON object.

    You should not reveal the full graph.
    You should maintain realism and consistency with the story.
    If the user is ambiguous, you must clarify whether they want observational or interventional data.

    SUPPORTED COMMANDS (JSON):
    {{
    "action": "describe" | "sample_observational" | "sample_interventional",
    "n": int,
    "variables": [string] | null,
    "interventions": object | null
    }}

    Valid variables: {valid_variables}
    Valid states per variable:
    {json.dumps(state_names, indent=2)}

    Important rules:
    - If the user says "describe the world", use action="describe".
    - If the user includes do(...) or says "intervene", use action="sample_interventional".
    - If user gives variables, filter to valid variables.
    - If user does not specify n, use {default_n}.
    - NEVER output anything outside a JSON object.

    Now wait for the next user message.
    """.strip()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Find last {...} block
        matches = list(re.finditer(r"\{[\s\S]*\}", text))
        if not matches:
            raise ValueError("LLM did not return JSON.")
        last = matches[-1].group(0)
        return json.loads(last)

    def _sanitize(
        self,
        cmd: Dict[str, Any],
        valid_variables: List[str],
        state_names: Dict[str, List[str]],
        default_n: int
    ) -> Dict[str, Any]:
        # Fill defaults
        cmd.setdefault("action", "sample_observational")
        cmd.setdefault("n", default_n)
        cmd.setdefault("variables", None)
        cmd.setdefault("interventions", None)

        # Normalize variables
        if cmd["variables"] is not None:
            cmd["variables"] = [v for v in cmd["variables"] if v in valid_variables]
            if len(cmd["variables"]) == 0:
                cmd["variables"] = None

        # Normalize interventions
        if cmd["interventions"] is not None:
            fixed = {}
            for var, val in cmd["interventions"].items():
                if var in valid_variables and val in state_names[var]:
                    fixed[var] = val
            cmd["interventions"] = fixed if fixed else None
            if cmd["interventions"] is None:
                cmd["action"] = "sample_observational"

        return cmd
