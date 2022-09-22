---
layout: default
title: "Diffusion model"
permalink: /diffusion_model/
theme: jekyll-theme-architect
---

# Diffusion model

In this blog post we will develop from scratch an example of generative inverse diffusion model. The model aims to generate new realistic digit as seen in the MNIST dataset. 


The diffusion process consist in the progressive noising of a starting image.

During this procedur the information content of the starting image is progressivile reduced and in the end we are left with pure wihite noise.
The inverse diffusion is the process that takes an image from any step of the procedur and aim to reconstruct the preciding step. This can be done with a prorusly trained machine learninig model.

This model will be a vision transformer. For this blog post we skip the specific of this model and we concentarre of the diffusion procedure
We starts with some inputs, note how we also reshape the classic MNIST dataset to be made of (20, 20) shaped images. This is done purly for performance reason.
The hart of the diffusion procedure is the data generation. So let’s explore it in more details
The noise is generated from a gaussiion distribution with mean 0 and variance 1


[back](https://piantedosi.github.io/)
