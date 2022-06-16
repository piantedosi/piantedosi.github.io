---
layout: default
title: "kaggle 1"
permalink: /diffusion_model/
theme: jekyll-theme-architect
---

# usato https://tfhub.dev/google/universal-sentence-encoder/4 per generare i word embeddings

seguito da un modello di tipo ensamble Random forset/adaboost/altri per le previsoni, la log_loss alla meglio è sui 0.9

idea 1)  modelo word embadding + ensambele con anche gli embeddings delle altre parti di testo (escludendo la parte che sto vedendo in questo momento, includendola l'ho già fatto è 
non porta a nulla di nulla)

idea 2) isolation forest per cercare di identificare le parti di testo non adeguate! l'idea è apprendere informazioni sull'inadeguatezza anche dalle parti adeguate)

day 2:

- test con TDIDF ed embedding in random forest: 
    questo test è risultato impossibile per problemi di memoria 
    SOLUZIONE: tentare di salvare gli oggetti embeddings e TFIDF per ricaricarli senza buttare memoria per crearli
    
- test con la previsione unicamente di Ineffective:
  questo test è fallito parzialmente, roc, precison-recal e cumulative gain non sono nulli, ma non sono buoni
  NOTA: non è detto che la strada sia fallimentare può essere un pezzo di un modello più completo
