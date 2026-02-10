"""Synthetic benchmark dataset for memory system evaluation.

Generates experiences with known ground-truth clusters so we can
measure retrieval precision/recall objectively.
"""

from __future__ import annotations

import random
import time

from emms.core.models import Experience

# Topic templates — each has a domain, keywords, and sentence patterns
TOPICS = {
    "finance": {
        "domain": "finance",
        "sentences": [
            "The stock market {verb} {pct}% on {reason}",
            "Federal Reserve {action} interest rates to {rate}%",
            "Bitcoin {verb} past ${price} amid {sentiment} sentiment",
            "{company} reported {quality} quarterly earnings of ${amount}B",
            "US GDP {verb} at {pct}% annualized rate in Q{quarter}",
            "Bond yields {verb} to {rate}% as investors {action}",
            "Oil prices {verb} ${price} per barrel on {reason}",
        ],
        "fills": {
            "verb": ["rose", "fell", "surged", "dropped", "climbed", "declined"],
            "pct": ["1.2", "2.5", "3.8", "0.7", "5.1", "4.3"],
            "reason": ["strong earnings", "inflation fears", "rate cut hopes",
                       "geopolitical tension", "labor market data", "tech rally"],
            "action": ["raised", "cut", "held", "adjusted"],
            "rate": ["4.5", "5.0", "3.75", "5.25", "4.0"],
            "company": ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta"],
            "quality": ["strong", "weak", "record", "disappointing", "mixed"],
            "amount": ["2.3", "18.7", "4.1", "9.8", "12.5"],
            "price": ["45000", "62000", "38000", "71000", "85"],
            "sentiment": ["bullish", "bearish", "neutral", "optimistic"],
            "quarter": ["1", "2", "3", "4"],
        },
    },
    "tech": {
        "domain": "tech",
        "sentences": [
            "{company} announced {product} with {feature} capabilities",
            "New AI model {name} achieves {metric} on {benchmark}",
            "Open source project {project} reaches {count} GitHub stars",
            "{language} {version} released with {feature} improvements",
            "Cybersecurity breach at {company} exposes {count} records",
            "{company} acquires {target} for ${amount} billion",
        ],
        "fills": {
            "company": ["OpenAI", "Google", "Meta", "Apple", "NVIDIA", "Anthropic"],
            "product": ["GPT-5", "Gemini Ultra", "Llama 4", "new chip", "AR glasses"],
            "feature": ["reasoning", "multimodal", "real-time", "autonomous", "edge"],
            "name": ["Claude 4", "GPT-5", "Gemini 2", "Llama 4"],
            "metric": ["state-of-the-art", "95% accuracy", "human parity"],
            "benchmark": ["MMLU", "HumanEval", "ARC-AGI", "GPQA"],
            "project": ["Llama.cpp", "Ollama", "vLLM", "LangChain"],
            "count": ["50000", "100000", "1M", "10M"],
            "language": ["Python", "Rust", "Go", "TypeScript"],
            "version": ["3.14", "2.0", "4.0", "1.0"],
            "target": ["a startup", "DeepMind spinoff", "a robotics firm"],
            "amount": ["2.1", "6.5", "14", "32"],
        },
    },
    "weather": {
        "domain": "weather",
        "sentences": [
            "Heavy {precip} expected in {region} with {intensity} winds",
            "Temperature {verb} to {temp}°F across the {region}",
            "Hurricane {name} strengthens to Category {cat} in the {ocean}",
            "{region} faces {condition} conditions through the {period}",
            "Record {extreme} temperatures reported in {region} at {temp}°F",
        ],
        "fills": {
            "precip": ["rain", "snow", "hail", "thunderstorms", "flooding"],
            "region": ["East Coast", "Midwest", "Pacific Northwest", "Gulf Coast", "Southwest"],
            "intensity": ["strong", "gusty", "severe", "moderate", "tropical"],
            "verb": ["rose", "dropped", "plummeted", "soared", "climbed"],
            "temp": ["105", "32", "45", "78", "-10", "92"],
            "name": ["Maria", "Felix", "Irene", "Oscar"],
            "cat": ["3", "4", "5"],
            "ocean": ["Atlantic", "Pacific", "Gulf of Mexico"],
            "condition": ["drought", "flooding", "blizzard", "heat wave"],
            "period": ["weekend", "week", "next three days"],
            "extreme": ["high", "low"],
        },
    },
    "science": {
        "domain": "science",
        "sentences": [
            "Researchers discover {finding} in {field} study",
            "New {instrument} reveals {discovery} about {subject}",
            "Clinical trial shows {drug} {result} for {condition}",
            "{institution} publishes breakthrough in {field}",
            "Study in {journal} reports {finding} across {count} patients",
        ],
        "fills": {
            "finding": ["novel mechanism", "genetic link", "quantum effect", "new species"],
            "field": ["neuroscience", "quantum physics", "genomics", "astrobiology"],
            "instrument": ["telescope", "particle accelerator", "gene sequencer", "MRI"],
            "discovery": ["unexpected structure", "new particles", "dark matter signal"],
            "subject": ["black holes", "DNA repair", "consciousness", "dark energy"],
            "drug": ["mRNA vaccine", "gene therapy", "immunotherapy", "CRISPR treatment"],
            "result": ["significant improvement", "promising results", "complete remission"],
            "condition": ["cancer", "Alzheimer's", "heart disease", "rare genetic disorder"],
            "institution": ["MIT", "Stanford", "CERN", "Nature team"],
            "journal": ["Nature", "Science", "Cell", "The Lancet"],
            "count": ["500", "1200", "10000", "3500"],
        },
    },
}


def _fill(template: str, fills: dict[str, list[str]]) -> str:
    result = template
    for key, values in fills.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_dataset(
    n_per_topic: int = 50,
    seed: int = 42,
) -> tuple[list[Experience], dict[str, list[str]]]:
    """Generate benchmark dataset with ground-truth topic labels.

    Returns (experiences, ground_truth) where ground_truth maps
    domain → list of experience IDs.
    """
    random.seed(seed)
    experiences: list[Experience] = []
    ground_truth: dict[str, list[str]] = {}
    base_time = time.time()

    for topic_name, topic in TOPICS.items():
        ground_truth[topic_name] = []

        for i in range(n_per_topic):
            template = random.choice(topic["sentences"])
            content = _fill(template, topic["fills"])
            importance = random.uniform(0.3, 1.0)
            novelty = random.uniform(0.2, 1.0)

            exp = Experience(
                content=content,
                domain=topic["domain"],
                timestamp=base_time + i * 0.1,
                importance=importance,
                novelty=novelty,
            )
            experiences.append(exp)
            ground_truth[topic_name].append(exp.id)

    random.shuffle(experiences)
    return experiences, ground_truth


# Pre-built queries with known relevant domains
QUERIES = {
    "stock market performance and earnings": "finance",
    "interest rates and Federal Reserve policy": "finance",
    "artificial intelligence model benchmarks": "tech",
    "open source programming language release": "tech",
    "hurricane and severe weather warnings": "weather",
    "temperature and climate conditions": "weather",
    "medical research clinical trial results": "science",
    "quantum physics particle discovery": "science",
}
