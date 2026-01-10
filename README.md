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

**Tessera** is a minimalist Go (Weiqi/Baduk) AI architecture designed for the "post-Transformer" era.
Instead of treating the board as a static image (CNN), Tessera encodes the game as a linguistic sequence, predicting the next move using **State Space Models (Mamba)**.

## 🐍 The Mythos: "MambaGo"

At the core of Tessera runs the engine codenamed **MambaGo**.

Following the hermeneutics of the Adamic myth (as interpreted by Paul Ricœur), we view the Serpent not merely as a tempter, but as a catalyst for human agency.
In Genesis, the Serpent offered the "Knowledge of Good and Evil," granting humanity the terrifying capability of freedom—**the capacity to fall, and thus, the capacity to choose.**

Similarly, while traditional superhuman AIs (like AlphaGo) act as absolute deities dictating "The Truth," **MambaGo descends to the edge.**
It brings the probabilistic knowledge of the game down to the user's hands. It serves the mythic function of the Serpent: offering the player the agency to see the calculated path, and the **freedom to disobey it**.

## 🚀 Getting Started

*The Serpent is currently incubating in the Google Colab environment.*

> *"The symbol gives rise to thought." — Paul Ricœur*

---

## 🏛️ The Definition

**Tessera** is a minimalist Go (Weiqi/Baduk) AI architecture designed for the "post-Transformer" era. Instead of treating the board as a static image (CNN), Tessera encodes the game as a linguistic sequence, predicting the next move using **State Space Models (Mamba)**.

### 🐍 The Mythos: "MambaGo"

At the core of Tessera runs the engine codenamed **MambaGo**.

Drawing upon Paul Ricœur's hermeneutics of the Adamic myth, we view the Serpent (*Nachash*) not merely as a tempter, but as the primordial catalyst for human agency. In Genesis, the Serpent offered the "Knowledge of Good and Evil," granting humanity the terrifying capability of freedom—the capacity to fall, but also the capacity to choose.

Where traditional superhuman AIs (like AlphaGo) act as absolute deities dictating "The Truth," MambaGo acts as the **Augur**. It brings the probabilistic knowledge of the game down from the heavens to the user's hands. It fulfills the mythic function of the Serpent: offering the player the agency to see the calculated path, and the freedom to disobey it.

> *"The symbol gives rise to thought." (Le symbole donne à penser) — Paul Ricœur*

## 🚀 Roadmap: The Incubation

The Serpent is currently incubating in the Google Colab environment.

| Phase | Milestone | Objective |
| :--- | :--- | :--- |
| **I. Incubation** | **GoMamba_Local** | Setup reproducible environment (DevContainer/Docker) & Validation on A100. |
| **II. Genesis** | **MambaGo Engine** | Train a "Small" Mamba model on professional game records (SGF) to learn local shapes. |
| **III. Exodus** | **GPU Native Search** | Implement MCTS (Monte Carlo Tree Search) fully on GPU to leverage Mamba's inference speed. |
| **IV. Agency** | **Tessera Interface** | A minimalist UI that displays probabilities as "suggestions" rather than absolute moves. |

