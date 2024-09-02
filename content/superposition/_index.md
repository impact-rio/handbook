---
title: 5. Superposição
weight: 5
aliases: 5
---

A superposição acontece quando um modelo representa mais de $n$ _features_ em um espaço de ativações $n$-dimensional. Esse fenômeno pode acontecer dependendo da distribuição dos dados e suas _features_. Para conseguir estudar e replicar esse fenômeno, o artigo "_Toy Models for Superpostion_" [^elhage2022superposition] propõe modelos simples (chamados de modelos de brinquedo, ou _toy models_) que servem para testar vários modelos simultaneamente, variando aspectos dos dados e do próprio modelo.

> **Definição (_feature_)**
> Uma _feature_ é alguma característica dos dados de entrada que pode ser representada pelo modelo como direções no espaço latente.

Observações:
  - O entendimento do que é realmente uma _feature_ pode ser confuso. Uma _feature_ pode ser tanto uma característica simples do dado de entrada (ex.: se uma foto contém um gato ou não) ou algo que não é tão interpretável a primeira vista.

É natural pensar que, dado que o modelo representa _features_ como direções, só seria eficiente para o modelo representar uma quantidade de _features_ no máximo igual ao que ele possui de dimensões. Por exemplo, características de uma imagem como ter um gato ou um carro aparentam ser independentes, sendo mais eficiente dar direções ortogonais para essas duas _features_. No entanto, a importância ou probabilidade de certa _feature_ pode afetar como o modelo vai representá-la.

> **Definição (importância de uma _feature_)**
> A importância $I_i$ de uma _feature_ $i$ é o quão útil esta _feature_ é para atingir uma _loss_ mais baixa.

> **Definição (probabilidade de uma _feature_)**
> A probabilidade $p_i$ de uma _feature_ $i$ é a probabilidade dessa _feature_ ser diferente de zero no conjunto de dados.

Observações:
  - Podemos interpretar a probabilidade de uma _feature_ como uma medida complementar a uma medida de esparsidade (i.e., $S_i = 1 - p_i$): se uma _feature_ possui uma baixa probabilidade, isso implica que essa _feature_ é muito esparsa. Essa noção será útil ao analisar os gráficos do nosso modelo de brinquedo.
  - O modelo de brinquedo a ser utilizado é o _ReLU output model_, definido como $
      h = W x,
      x' = \text{ReLU}(W^T h + b),
    $ onde $W \in \mathbb{R}^(2 \times 5)$ é uma matriz de pesos e $b \in \mathbb{R}^2$ é um vetor de _bias_.
  - A _loss_ é dada por $L = \frac 1 {B F} \sum_x \sum_i I_i (x_i - x'_i)^2$, onde $I_i$ é a importância que damos para a _feature_ _i_, $B$ é o tamanho do batch e $F$ é a quantidade de _features_.
  - A partir da imagem abaixo, é possível observar que existem duas tendências conflituosas no processo de treinamento: 1) representar mais _features_, que é considerado desejável, e 2) reduzir interferência entre as _features_, já que também é interessante representar _features_ ortogonalmente.

> ![](../img/superposition.png)
> Gráfico dos pesos aprendidos pelo modelo de brinquedo de duas dimensões a depender da escassez das _features_ (que são 5 no total.) Se a esparsidade é baixa, o modelo dedica exclusivamente as duas dimensões para as _features_ mais importantes. Se a esparsidade aumenta, o modelo chega a comportar todas as 5 _features_ em um espaço de duas dimensões, causando um efeito de interferência entre as _features_.

Outra propriedade das _features_ que afeta como o modelo as representa é a correlação entre elas. Assim como uma baixa probabilidade das _features_ incentiva superposição, esperaríamos que a presença de _features_ anti-correlacionadas também incentive esse fenômeno.

> **Definição (correlação entre _features_)**
> Duas _features_ são correlacionadas se elas aparecem juntas em um conjunto de dados. Analogamente, duas _features_ são anti-correlacionadas se a aparição de uma delas está ligada a não aparição da outra.

> ![](../img/superposition_correlation.png)
> Gráfico dos pesos aprendidos pelo modelo de brinquedo a depender da correlação das _features_. Modelos tendem a representar _features_ correlacionadas em dimensões ortogonais (ou lado a lado se não for possível representá-las ortogonalmente) e tendem a representar _features_ anti-correlacionadas em direções opostas.

Observações:
 - A maioria das _features_ normalmente encontradas são anti-correlacionadas. Por exemplo, raramente um texto será classificado, ao mesmo tempo, como um código _Python_ e como uma ficção científica.

Assim, os fenômenos acima levam à ampla ocorrência de superposição em modelos de aprendizado de máquina. Não é possível estabelecer uma relação direta entre uma _feature_ e um neurônio (ou ativações), e não há uma base interpretável.


{{% notice style="primary" title="Superposição _versus_ polissemia" %}}
  Esses termos são bastante comuns quando tratamos de interpretabilidade de modelos, e normalmente observados em conjunto, mas definem conceitos diferentes:
  - *Polissemia:* ocorre quando um neurônio representa múltiplas _features_. Por si só não seria um problema, pois poderíamos encontrar uma base para _features_ tal que cada vetor correspondesse a uma única _feature_.

  - *Superposição:* ocorre quando temos mais _features_ que dimensões, e implica polissemia, já que torna necessário que uma única dimensão represente mais de uma _feature_ (a implicação contrária não é válida).
{{% /notice %}}

{{% bibliography %}}
