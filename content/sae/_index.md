---
title: 6. SAE
weight: 6
aliases: 6
---

Recentemente, a linha de pesquisa de utilizar Autoencoders Esparsos (ou Sparse Autoencoders) tem sido cada vez mais explorada na área de interpretabilidade de modelos como uma tentativa de contornar o problema da superposição. A proposta é utilizar esses modelos para gerar _features_ aprendidas que ofereçam uma unidade mais monossêmico de análise que os neurônios do modelo.

> **Definição (_autoencoder_)**.
> Tipo de rede neural cujo objetivo é aprender representações eficientes de maneira não supervisionada. É composto por um _encoder_, que gera a representação latente (que, em geral, possui dimensão menor que o _input_), e um _decoder_, responsável por realizar uma reconstrução da representação do espaço latente de volta para o espaço do _input_.

> ![](../img/classic_autoencoder.png)
> _Autoencoder_ clássico, contando com _encoder_ e _decoder_ para produzir uma representação latente mais simples das entradas originais que possam ser reconstruídas ao espaço de entrada.

> **Definição (_autoencoder_ esparso (SAE))**.
> _Autoencoder_ que aprende representações esparsas em um espaço de dimensionalidade maior do que as entradas.

Observações:
- A função de perda de um SAE normalmente é composto pela soma de uma perda de reconstrução (norma 2 entre a entrada e a saída) e uma penalidade de esparsidade (norma 1 da representação latente).
- A ideia trazida recentemente em artigos buscando contornar a superposição de _features_ [^cunningham2023sparseautoencoders] [^bricken2023monosemanticity] é utilizar SAEs para recuperar _features_ sobrepostas, mapeando o espaço de _features_ para um espaço latente esparso e de maior dimensão, permitindo a extração monossêmica de _features_ interpretáveis. 
- Esse _autoencoder_ é treinado em camadas internas de modelos de linguagem, decompondo as ativações em mais _features_ do que a quantidade de neurônios existentes.


{{% notice style="primary" title="Hipótese da superposição" %}}
  Teoria de que redes neurais pequenas exploram a esparsidade de _features_ e propriedade de espaços de alta dimensão para simular aproximadamente redes muito maiores e mais esparsas.

  ![](../img/superpositionhip.png)

  A proposta de utilizar SAEs para resolver a superposição compreende que o espaço latente obtido após treinamento em camadas internas da MLP de Transformers realiza a projeção inversa, nos levando de volta ao modelo hipotético em que não há superposição. 
{{% /notice %}}

{{% notice style="primary" title="SAE para interpretabilidade do Claude Sonnet 3" %}}
  O uso de SAEs para interpretabilidade tem crescido em tempos recentes, principalmente desde a publicação de "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" pela Anthropic.

  Esse artigo estabelece o primeiro sucesso em utilizar SAEs para criar um espaço de _features_ interpretáveis e monossêmicas em um modelo de linguagem grande, já que até então todos os avanços haviam sido feitos em _toy models_.

  Além de descobrir um espaço de milhões de _features_, o artigo também apresenta resultados qualitativos da alteração de _features_ especificas, aumentando e diminuindo valores de ativações correspondentes para observar comportamentos do modelo.


   ![](../img/anthropic.png)
{{% /notice %}}

{{% bibliography %}}
