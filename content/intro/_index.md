---
title: 1. Introdução
weight: 1
aliases: 1
---

Modelos de linguagem baseados em _transformers_ são considerados estado-da-arte para tarefas relacionadas à linguagem natural, tendo desencadeado uma corrida tecnológica entre as maiores empresas de tecnologia do mundo para o treinamento e aperfeiçoamento desses modelos. No entanto, pouco sabemos sobre o funcionamento e raciocínio interno desses modelos. Isto se dá pela natureza de arquiteturas de aprendizado profundo como a de _tranformers_, que possuem milhões, até bilhões, de parâmetros. 

A partir disso surgem as técnicas de _mechanistic interpretability_, "interpretabilidade mecanicista" em tradução literal, que visam aplicar engenharia reversa aos modelos para entender seu funcionamento. As motivações para buscar esse entendimento mais profundo são muitas, mas esse _handbook_ tem como principal objetivo introduzir técnicas relevantes para pesquisas no campo de alinhamento de inteligência artificial.

O desenvolvimento desse _handbook_ é produto de um processo de _upskilling_ em _mechanistic interpretability_ de _transformers_ financiado pela
[Condor Initiative](https://condorinitiative.org/), e tem como objetivo introduzir o assunto a leitores que buscam um contato inicial com alinhamento de IA e estudos de interpretabilidade. Espera-se, para entendimento pleno deste _handbook_, que o leitor tenha conhecimentos de álgebra linear, estatística e técnicas básicas de aprendizado de máquina, como _multi-layer perceptrons_ e funções de ativação. 

Este texto tem base no [material de interpretabilidade do programa ARENA 3.0](https://arena3-chapter1-transformer-interp.streamlit.app/), mantendo maior foco em partes teóricas, e está estruturado em definições, observações e caixas de conteúdos e exemplos adicionais. Recomendamos ao leitor interessado em saber mais a exploração do material do ARENA, que contém exercícios práticos fundamentais ao entendimento mais aprofundado deste conteúdo.
