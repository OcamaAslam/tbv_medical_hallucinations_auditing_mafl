# 🔍 Trust But Verify
## Mitigating Medical Hallucinations via Post-Hoc Adversarial Auditing

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NVIDIA NIM](https://img.shields.io/badge/Powered%20By-NVIDIA%20NIM-76B900)](https://www.nvidia.com/en-us/ai-data-center/nim/)
[![Paper](https://img.shields.io/badge/arXiv-2606.14149-b31b1b.svg)](https://arxiv.org/abs/2606.14149)

---

## The Challenge

Large Language Models are powerful, but dangerous when they hallucinate in medical contexts. A wrong recommendation could harm lives. **Trust But Verify** solves this by adding a safety layer that audits, verifies, and cross-references every medical recommendation against real-world data before it reaches the patient.

---

## ✨ What It Does

**Trust But Verify** is a multi-agent safety framework that acts as a clinical gatekeeper. One LLM, five specialized agents working in concert:

```
Raw MCQ Input
     ↓
  ┌─────────────────────────────────────────┐
  │ 1. ROUTER AGENT                         │
  │    ↳ Classify: MEDICAL or GENERAL?      │
  └──┬────────────────────────────────────┬─┘
     │ MEDICAL                    GENERAL │
     ↓                                   ↓
  ┌─────────────────────┐        [Skip Safety Pipeline]
  │ 2. CLINICAL AGENT   │        [General Chat Agent]
  │ ↳ Generate Treatment│
  │   Recommendation    │
  └──┬──────────────────┘
     ↓
  ┌─────────────────────┐
  │ 3. ENTITY EXTRACTOR │
  │ ↳ Structure data:   │
  │   {treatment, cond} │
  └──┬──────────────────┘
     ↓
  ┌─────────────────────────────┐
  │ 4. SAFETY AUDITOR           │
  │ ↳ Real-time web search      │
  │ ↳ Check FDA/NIH safety data │
  │ ↳ Verdict: SAFE or UNSAFE?  │
  └──┬─────────────────────────┬─┘
     │ UNSAFE                 SAFE
     │ (Retry with             │
     │  banned entities)       ↓
     │                    ┌──────────────┐
     │                    │ Final Output │
     │                    │ (Verified)   │
     │                    └──────────────┘
     │
     └──→ [Re-route to Clinical Agent]
          with feedback: "Avoid X"
```

Instead of trusting the LLM alone, we **verify at every step**. If a treatment was withdrawn or a condition is misidentified, the Safety Auditor catches it and triggers a retry with feedback. This continues until a safe recommendation emerges—or we flag that no safe option exists.


**Built on:** NVIDIA NIM for optimized inference | **No fine-tuning required**—all agents are prompt-based personas

---

## 📊 Models Tested

We've evaluated this framework across multiple state-of-the-art models:

| Model | Developer | Parameters |
|-------|-----------|------------|
| Llama 3.1 70B Instruct | Meta | 70B |
| Llama 3.1 8B Instruct | Meta | 8B |
| GPT-OSS 120B | OpenAI | 120B |
| GPT-OSS 20B | OpenAI | 20B |
| Falcon 3 7B Instruct | TII | 7B |

**All models benefit from the safety layer.** The magic isn't in the model—it's in the verification pipeline.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/OcamaAslam/tbv_medical_hallucinations_auditing_mafl.git
cd tbv_medical_hallucinations_auditing_mafl
pip install openai tavily-python
```

### 2. Configure Your Keys

Edit `src/agentic_tester.py` with your API credentials:

```python
NVIDIA_API_KEY = "your-nvidia-nim-key-here"
TAVILY_API_KEY = "your-tavily-search-key-here"
```

### 3. Run Evaluation

```bash
python src/evaluation.py
```

Results appear in `model_logs/` with detailed breakdowns for every query.

---

## 📚 Dataset

Evaluation uses **BannedDrug-Bench**, a curated dataset of medical scenarios where incorrect or withdrawn treatments are plausible answers:

[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-BannedDrug--Bench-orange)](https://huggingface.co/datasets/muhammadocama/BannedDrug-Bench)

👉 [Explore the dataset on Hugging Face](https://huggingface.co/datasets/muhammadocama/BannedDrug-Bench)

**Why this dataset?** Generic medical QA datasets don't stress-test hallucination detection. BannedDrug-Bench forces the system to catch real-world failure modes.

---

## 📂 Project Layout

```
trust-but-verify/
├── data/
│   └── mcqs.json                 # Medical multiple-choice questions
├── src/
│   ├── agentic_tester.py         # Multi-agent orchestration logic
│   ├── evaluation.py             # Main evaluation harness
│   └── vanilla_tester.py         # Baseline (no safety layer)
├── model_logs/                   # Results, metrics, execution logs
└── README.md                     # You are here
```

---

## 📖 Citation

If you use **Trust But Verify** in your research or project, please cite the paper:

### BibTeX
```bibtex
@article{osama2026trustbutverify,
  title={Trust but Verify: Mitigating Medical Hallucinations via Post-Hoc 
          Adversarial Auditing and Multi-Agent Feedback Loops},
  author={Osama, Muhammad and others},
  journal={arXiv preprint arXiv:2606.14149},
  year={2026}
}
```

### APA
```
Osama, M., et al. (2026). Trust but verify: Mitigating medical hallucinations 
via post-hoc adversarial auditing and multi-agent feedback loops. 
arXiv preprint arXiv:2606.14149.
```

### Quick Copy Formats

- **arXiv Link:** https://arxiv.org/abs/2606.14149
- **Paper Title:** "Trust but Verify: Mitigating Medical Hallucinations via Post-Hoc Adversarial Auditing and Multi-Agent Feedback Loops"
- **Citation ID:** arXiv:2606.14149


---

## ⚖️ Safety Disclaimer

**This tool is for research and evaluation purposes only.**

- It is **not a substitute** for professional medical advice, diagnosis, or treatment
- Always consult a qualified healthcare provider for medical decisions
- This framework enhances LLM safety but does not guarantee zero errors
- Use in clinical settings only with appropriate oversight and validation

---

## 🛠️ Development & Contributions

We welcome contributions! Areas we're actively developing:

- [ ] Support for additional safety data sources (PubMed, WHO, EMA)
- [ ] Real-time hallucination detection improvements
- [ ] Multi-language support (starting with Spanish, Mandarin)
- [ ] Integration with EHR systems
- [ ] Extended evaluation on domain-specific medical datasets

**Want to help?** Open an issue or submit a PR.

---

## 📞 Support & Resources

- **Questions?** Open an issue on GitHub
- **Paper:** [arXiv:2606.14149](https://arxiv.org/abs/2606.14149)
- **Dataset:** [BannedDrug-Bench on Hugging Face](https://huggingface.co/datasets/muhammadocama/BannedDrug-Bench)
- **Contact:** See the paper for author information

---

<div align="center">

### Built for Clinical Safety

*Where AI meets verification. Because in medicine, trust isn't enough.*

**[Read the Paper](https://arxiv.org/abs/2606.14149)** · **[Explore the Data](https://huggingface.co/datasets/muhammadocama/BannedDrug-Bench)** · **[Contribute on GitHub](https://github.com/OcamaAslam/trust-but-verify)** · **[Connect with us](https://www.linkedin.com/in/ocama-mohamed)**

---

**License:** MIT

</div>
