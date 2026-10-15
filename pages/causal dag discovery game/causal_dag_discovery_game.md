---
layout: default
title: "Causal DAG discovery game"
description: "Discover the arrows of a hidden causal DAG."
permalink: /causal_dag_discovery_game/
theme: jekyll-theme-architect
classes: wide
---

# Causal DAG discovery game

Try your hand at causal discovery! In this game, you can attempt to uncover the hidden structure of a causal DAG yourself using counterfactual data.

For a fair challenge (playing under the same conditions as the AI agent), play with "easy mode" turned off. The game begins with just two nodes and becomes more complex as you advance.

**How to play:**
- Node values depend on their parent nodes.
- **Draw an arrow:** Drag from one node to another to propose a directed edge (a causal link).
- **Remove an arrow:** Right-click an existing edge to delete it.
- **Freeze a node:** Click a node to set its value manually.
- The bar at the bottom of the screen shows the number of missing true edges and incorrect guesses.

Can you discover the true causal structure?

<iframe src="{{ '/assets/causal_dag_discovery_game/index.html' | relative_url }}?v=4" title="Causal DAG discovery game" width="640" height="360" style="border: 1px solid #ccc; max-width: 100%;"></iframe>
