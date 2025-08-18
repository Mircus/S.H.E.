
[![License: HNCL](https://img.shields.io/badge/license-HNCL-blue.svg)](/LICENCE.md)

<p align="center">
  <img src="she_logo.png" alt="SHE Logo" width="450"/>
</p>

# SHE: Simplicial Hyperstructure Engine

**SHE** is a modular Python framework for modeling, analyzing, and learning from **higher-order data** represented as weighted simplicial complexes.  
It integrates **topological data analysis (TDA)**, **Simplicial Neural Networks (SNNs)**, **Morse theory**, and **algebraic invariants** in a unified engine.

---

## 🔺 Motivation

Graphs are not enough. Many real-world systems involve **higher-order relationships**:  
teams, co-occurrences, cliques, multi-particle interactions, social triads.  

**SHE** generalizes network science to higher dimensions, providing:
- Simplicial complexes that evolve over time  
- Diffusion, persistence, and harmonic analysis  
- Simplicial convolutional operators for deep learning  

---

## 📦 Features

- Ingest raw data and build **time-evolving simplicial complexes**
- Add **weights** and **attributes** to nodes, edges, and higher-order simplices
- Apply **Simplicial Neural Networks (SNNs)** to structured data
- Compute **Hodge Laplacians** and run diffusion processes
- Extract **persistent homology** features (via Giotto-TDA / Gudhi)
- Rank simplices by **diffusion centrality**
- Visualize structures and diffusion dynamics interactively

---

## ⚡ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-org>/S.H.E..git
cd S.H.E.-main
pip install -r requirements.txt
```

Or install as a package (after adding `pyproject.toml`):

```bash
pip install .
```

---

## 🛠 Dependencies

Core libraries include:
- `torch`, `torch-geometric`
- `xgi`, `toponetx`
- `giotto-tda`, `gudhi`
- `networkx`, `scipy`, `numpy`, `pandas`
- `matplotlib`, `seaborn`, `plotly`
- `streamlit`

See [requirements.txt](requirements.txt) for details.

---

## 🚀 Quickstart

Run the demo script:

```bash
python examples/SHEDemo.py
```

Or explore the interactive notebook:

```bash
jupyter notebook examples/SHE_Comprehensive_Demo.ipynb
```

---

## 📂 Repository Structure

```
src/core/              # main engine modules
  ├── SHE.py           # central SHE engine
  ├── diffusion/       # Hodge Laplacians and diffusion
  ├── complex/         # simplicial complex representations
  ├── io/              # data loaders
  ├── visualize/       # visualization utilities
examples/              # demo scripts and notebooks
docs/                  # manuals and specifications
data/                  # synthetic datasets
```

---

---

## 📦 Features

- Ingest raw data and build **time-evolving simplicial complexes**
- Add **weights** and **attributes** to nodes, edges, and higher-order simplices
- Apply **Simplicial Neural Networks (SNNs)** via [TopoX](https://github.com/simplicial-topology/topox)
- Compute **persistence diagrams**, Laplacians, and other topological invariants
- Simulate and evaluate **diffusion dynamics**
- Visualize evolution and structure with an interactive **Streamlit dashboard**

---


---

## 🧠 Designed for Extensibility

SHE supports integration with:

- **Giotto-TDA**, **XGI**, **PyTorch**, and symbolic tools
- Any data type requiring higher-order modeling (biology, finance, linguistics, neuroscience)

---

## License
This project is licensed under the **Holomathics Non-Commercial License (HNCL)**.  
You are free to use, modify, and share this software for **personal, research, or educational** purposes.  
**Commercial use requires a separate license** from Holomathics.  
📧 Contact: [info@holomathics.com](mailto:info@holomathics.com)

---

## ✒️ Authors

**Mirco A. Mannucci**, with contributions from collaborators.
