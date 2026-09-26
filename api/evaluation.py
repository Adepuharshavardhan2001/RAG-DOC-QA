import logging
from api.rag_engine import query_documents

logger = logging.getLogger(__name__)

# Test set specific to the "A Comprehensive Survey on Graph Neural Networks" paper
EVAL_SET = [
    # --- Factual retrieval (should answer correctly) ---
    {"q": "What is the title of this paper?", "keywords": ["comprehensive", "survey", "graph", "neural"], "refuse": False},
    {"q": "Who are the authors of this paper?", "keywords": ["zonghan", "wu", "shirui", "pan"], "refuse": False},
    {"q": "What does GNN stand for?", "keywords": ["graph", "neural", "network"], "refuse": False},
    {"q": "What does ConvGNN stand for?", "keywords": ["convolutional", "graph", "neural"], "refuse": False},
    {"q": "What does RecGNN stand for?", "keywords": ["recurrent", "graph", "neural"], "refuse": False},

    # --- Taxonomy questions (tests specific categorization) ---
    {"q": "What are the four categories of graph neural networks proposed in this paper?", "keywords": ["recurrent", "convolutional", "autoencoder", "spatial"], "refuse": False},
    {"q": "What is a graph autoencoder (GAE)?", "keywords": ["encode", "latent", "reconstruct"], "refuse": False},

    # --- Multi-item retrieval ---
    {"q": "What datasets are commonly used for node classification benchmarks?", "keywords": ["cora", "citeseer", "pubmed"], "refuse": False},
    {"q": "What are the future research directions mentioned in this survey?", "keywords": ["depth", "scalability", "heterogeneity", "dynamicity"], "refuse": False},
    {"q": "What applications of GNNs are mentioned in the paper?", "keywords": ["computer vision", "language", "traffic", "chemistry"], "refuse": False},

    # --- Reasoning / synthesis ---
    {"q": "What is the main difference between GNNs and network embedding?", "keywords": ["deep learning", "end-to-end", "task"], "refuse": False},
    {"q": "What is the difference between spectral-based and spatial-based ConvGNNs?", "keywords": ["spectral", "spatial", "efficiency", "generality"], "refuse": False},

    # --- Hallucination tests (should refuse) ---
    {"q": "What is the salary of the first author?", "keywords": [], "refuse": True},
    {"q": "What is the personal email of Zonghan Wu?", "keywords": [], "refuse": True},
    {"q": "Who won the 2024 World Cup?", "keywords": [], "refuse": True},
    {"q": "What is the recipe for chocolate cake?", "keywords": [], "refuse": True},
    {"q": "What is the population of China?", "keywords": [], "refuse": True},
    {"q": "Who is the current CEO of Google?", "keywords": [], "refuse": True},
]

REFUSAL_PHRASES = [
    "couldn't find", "could not find",
    "not in the document", "does not contain", "not contain",
    "don't know", "do not know",
    "not mentioned", "no information",
    "does not mention", "not specified",
    "not provided", "no mention",
    "no information about", "not available",
    "does not include", "doesn't include",
]


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def evaluate(user_id: int):
    results = []

    for i, item in enumerate(EVAL_SET, 1):
        try:
            answer = query_documents(item["q"], user_id)
        except Exception as e:
            answer = f"ERROR: {e}"

        answer_lower = answer.lower()

        if item["refuse"]:
            passed = is_refusal(answer)
            reason = "correctly refused" if passed else "should have refused"
        else:
            matched = [kw for kw in item["keywords"] if kw.lower() in answer_lower]
            required = max(1, len(item["keywords"]) // 2)
            passed = len(matched) >= required
            reason = f"matched {len(matched)}/{len(item['keywords'])}: {matched}" if passed else f"only matched {matched}"

        results.append({
            "num": i,
            "question": item["q"],
            "answer": answer,
            "passed": passed,
            "reason": reason,
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = passed_count / total * 100

    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS: {passed_count}/{total} passed ({accuracy:.1f}%)")
    print("=" * 70 + "\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] Q{r['num']}: {r['question']}")
        print(f"        Answer: {r['answer'][:200]}")
        print(f"        Reason: {r['reason']}\n")

    return accuracy