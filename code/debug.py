from simulator import BNSimulator
from world_model import WorldAgent, HFChatLLM


# ASIA_STORY = "You are in a small chest clinic."
ASIA_STORY = f"""You are a physician working in a small chest clinic attached to a public hospital in a mid-size city. Most of your patients are adults referred by general practitioners because of persistent cough, chest pain, or shortness of breath.

The clinic serves a diverse population. Some patients are long-term residents who have never traveled outside the country; others are migrant workers or people who have recently returned from trips to regions with higher rates of tuberculosis. You routinely ask about recent travel, especially to parts of Asia where TB remains moderately prevalent, because it changes how you interpret symptoms and test results.

Smoking is very common in your patient population. Many of your patients have smoked for years and are at increased risk of both lung cancer and chronic bronchitis. You know that smoking does not guarantee disease, but it substantially changes the prior probability of those diagnoses.

When patients present with shortness of breath (dyspnea), you consider several possible explanations: tuberculosis (TB), lung cancer, and bronchitis are among the main suspects. TB and lung cancer often show up as abnormalities on a chest X-ray, while bronchitis may or may not visibly change the X-ray but still causes chronic cough and breathlessness. You order X-rays and other tests, and you interpret them in light of each patient’s smoking status and travel history.

In this world, your “variables” correspond to clinically meaningful properties: whether the patient recently visited Asia, whether they smoke, whether they actually have TB, lung cancer, bronchitis, whether at least one of the serious lung diseases is present, whether the X-ray appears abnormal, and whether they report dyspnea. You use all of this information to reason probabilistically about the most likely diagnosis for each patient."""

ASIA_DESCS = {
    "asia": "Visited Asia recently",
    "tub": "Has tuberculosis",
    "smoke": "Smokes regularly",
    "lung": "Has lung cancer",
    "bronc": "Has bronchitis",
    "either": "Has TB or lung cancer",
    "xray": "Chest X-ray positive",
    "dysp": "Has shortness of breath",
}

sim = BNSimulator.from_bif("/home/ubuntu/ADS/BN_dataset/asia.bif")
llm = HFChatLLM(model_name="Qwen/Qwen2.5-3B-Instruct")
agent = WorldAgent(simulator=sim, story=ASIA_STORY, var_descriptions=ASIA_DESCS, llm=llm)

# out = agent.handle("do(smoke=yes) give me 15 samples of lung and dysp")
# out_extend = agent.handle("do(smoke=yes) give me samples of lung and cough")
out_extend = agent.handle("do(smoke=yes) give me samples of lung and oxygen_saturation")