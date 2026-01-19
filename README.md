# Tessera

<div align="center">

### **T**oken **E**ncoded **S**tate-space **S**equence **E**ngine for **R**apid **A**nalysis

---

### *Technē Epi Sēma Syn Epistēmē Rhoē Archē*
*(Art upon Tokens, flowing with Knowledge as its Principle)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Mamba](https://img.shields.io/badge/Model-MambaGo-success)](https://github.com/state-spaces/mamba)

</div>

## 🏛️ The Definition

**Tessera** is a minimalist Go (Weiqi/Baduk) AI architecture designed for the "post-Transformer" era. Instead of treating the board as a static image (CNN), Tessera encodes the game as a linguistic sequence, predicting the next move using **State Space Models (Mamba)**.

## 🐍 The Mythos: "MambaGo"

At the core of Tessera runs the engine codenamed **MambaGo**.

Drawing upon Paul Ricœur's hermeneutics of the Adamic myth, we view the Serpent (*Nachash*) not merely as a tempter, but as the primordial catalyst for human agency. In Genesis, the Serpent offered the "Knowledge of Good and Evil," granting humanity the terrifying capability of freedom—the capacity to fall, but also the capacity to choose.

Where traditional superhuman AIs (like AlphaGo) act as absolute deities dictating "The Truth," MambaGo acts as the **Augur**. It brings the probabilistic knowledge of the game down from the heavens to the user's hands. It fulfills the mythic function of the Serpent: offering the player the agency to see the calculated path, and the freedom to disobey it.

> *"The symbol gives rise to thought." (Le symbole donne à penser) — Paul Ricœur*

## 🔬 Design Principles

| Principle | Description |
|-----------|-------------|
| **GPU Complete** | All operations complete within GPU, zero CPU transfer |
| **Clean Room** | No external game records; all learning from self-play only |
| **Probabilistic Output** | Returns probability distributions, not single "best" moves |
| **Observable** | All behaviors are monitorable |

## 🚀 Roadmap: The Incubation

| Phase | Milestone | Objective | Status |
| :--- | :--- | :--- | :--- |
| **I. Incubation** | **GoMamba_Local** | Setup reproducible environment (Docker + CUDA) | ✅ Complete |
| **II. Genesis** | **MambaGo Engine** | GPU-Native self-play learning with Mamba SSM | ✅ Complete |
| **III. Exodus** | **Self-Play RL** | Reinforcement learning through self-play (Tromp-Taylor rules under evaluation) | 🔄 In Progress |
| **IV. Agency** | **Tessera Interface** | A minimalist UI that displays probabilities as "suggestions" | ⏳ Planned |

## 🚀 Getting Started

*Currently under active development. See `HANDOFF.md` for current technical state.*

---

*"Le symbole donne à penser."* — Paul Ricœur
