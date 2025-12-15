from __future__ import annotations
from typing import Optional
from world_model import WorldAgent


def chat_loop(agent: WorldAgent):
    """
    Simple CLI chat loop. In Jupyter you can call agent.handle(...) directly,
    but this is handy for manual testing.
    """
    print("WorldAgent is ready. Type 'describe' or ask for samples. Type 'quit' to stop.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ["q", "quit", "exit"]:
            print("bye!")
            break

        out = agent.handle(user)
        print("\nWorldAgent:\n" + out["response"] + "\n")
