# Privacy-LLM-Routing
Privacy-Aware Adaptive LLM Routing is a neural network framework that dynamically routes AI queries between local, hybrid, and cloud LLMs by jointly optimizing privacy protection, energy efficiency, and task quality, achieving 99.34% accuracy and reducing privacy risk by 45.77% compared to always-cloud deployment.

#Overview
When a user sends a query to an AI assistant, the system must decide:

Local — process entirely on-device (best privacy, limited quality)
Hybrid — split processing between device and cloud
Cloud — send to a remote LLM (best quality, highest privacy risk + energy cost)
This project trains a routing model that learns to make this decision based on:

The query text (does it contain PII like email, phone, location, or medical data?)
The device state (battery level, CPU load, RAM, network type)
A computed privacy risk score

#Key Results (9K held-out test set)
Metric	                                            Value
Overall Accuracy	                                  99.34%
Macro F1-Score                                    	0.9934
Privacy Risk Reduction vs Always-Cloud	            −45.77%
Energy Cost (vs Always-Cloud)	                      Balanced

#Project Structure

privacy-llm/
├── data/
│   ├── dataset.py             # Dataset generator (Gemini API or local Gemma 2B)
│   ├── adaptive_dataset.py    # PyTorch Dataset & DataLoader
│   ├── generation.py          # Utility generation helpers
│   ├── analyze.py             # Dataset analysis tools
│   ├── verify_improvements.py # Verification helpers
│   ├── local_dataset.jsonl    # Training dataset (generated)
│   └── held_out_test.jsonl    # Held-out evaluation dataset
├── models/
│   └── routing_model.py       # AdaptiveRoutingModel (BERT + MLP fusion)
├── training.py                # Model training script
├── evaluation.py              # Evaluation vs baselines + held-out data generation
├── generate_plots.py          # Research visualizations
├── research_plots/            # Plots from held-out test set
├── research_plots_training/   # Plots from training dataset
└── research_plots_evaluation/ # Full evaluation plots with baselines


#Model Architecture

Query Text  ──► BERT Encoder (bert-base-uncased, last 2 layers fine-tuned)
                      │
                      ▼  [CLS] token (768-dim)
                  ┌───┴────────────────────┐
                  │       Fusion Layer      │
Device Features ──► Device MLP (5→128→256→128)
(battery, CPU,   │       └─ concat ─┘      │
 RAM, network,   │  Linear(896→256) + BN   │
 privacy_risk)   └───────────┬─────────────┘
                             │
                      Routing Head
                      Linear(256→3)
                             │
                   [Local | Hybrid | Cloud]

                   
