---
title: 3. Mechanistic Intepretability
weight: 3
aliases: 3
---

A proposta da _mechanistic interpretability_ é, a partir de um modelo pré-treinado, utilizar ténicas de engenharia reversa para descobrir algoritmos aprendidos pelo modelo a partir de seus pesos. Esse tipo de investigação é realizada com o objetivo de compreender melhor como o modelo se comporta em certas situações.

Modelos de aprendizado de máquina são descritos como _black box_, já que não conhecemos seu funcionamento. É importante que os modelos que treinamos tenham explicabilidade em diversos âmbitos. Em certas tarefas, como decisões de crédito, explicar as decisões do modelo a partes interessadas pode ser crucial para confiabilidade. Além disso, a interpretabilidade também desempenha um papel importante na segurança de modelos de inteligência artificial, já que compreender o funcionamento interno do modelo nos leva a aprimorar robustez de possíveis áreas de falha e compreender o impacto de decisões.

Observações:
  - Os conteúdos discutidos a partir de agora são frutos de aplicações bem práticas da teoria apresentada, e tem suporte para implementação na biblioteca [TransformerLens](https://github.com/neelnanda-io/TransformerLens"), desenvolvida por Neel Nanda para _mechanistic interpretability_ em modelos de linguagem similares ao GPT-2.
  O objetivo da biblioteca é facilitar o acesso e permitir modificações em partes internas do modelo, simplificando o processo de engenharia reversa.
  - O foco principal desse _handbook_ é discutir os conceitos teóricos de _mechanistic interpretability_ em _transformers_, mas encorajamos fortemente a exploração da biblioteca. Para isso, recomendamos o apoio da fonte inpiradora deste conteúdo (capítulo 1 do ARENA 3.0 [^ARENATransformerInterpretability]).

## Circuitos

Quando estudamos redes neurais profundas, nos deparamos quase certamente com grandes redes cheias de neurônios que se ativam sem um padrão muito claro. O estudo de circuitos visa entender a relação das camadas e os pesos com as ativações e descobrir os algoritmos que emergem do treinamento dessas redes neurais.

> **Definição (ativação)**.
  A ativação de um nó em um _forward pass_ é o valor associado a esse nó durante a propagação a partir de uma determinada entrada, i.e., o valor de $f_{\eta}(\mathbf{x})$, onde $f_{\eta}$ é a função que descreve o nó $\eta$ e $\mathbf{x}$ é uma entrada qualquer.

{{% notice style="primary" title="Ativação pra cá, ativação pra lá" %}}
  O termo ativação é bastante utilizado pela comunidade de interpretabilidade em múltiplos contextos. Façamos uma desambiguação do termo para tornar a leitura deste _handbook_ mais clara:

  - **Ativação como valor resultante de um _forward pass_**: trata-se de um valor associado à saída de um neurônio e repassado ao próximo, alterado a cada propagação.

  - **Função de ativação**: trata-se de uma função aplicada à combinação linear de entradas e pesos obtida pela saída de um neurônio. As funções de ativação introduzem não-linearidade no modelo, permitindo que aprendizado de padrões complexos. Podemos citar como exemplo as funções ReLU e sigmoide.

  - **Ativação de uma _head_ de atenção**: processo pelo qual uma _head_ de atenção identifica e foca em partes específicas do _input_ durante o cálculo da atenção. Quando uma _head_ de atenção "se ativa" (ou "dispara"), ela está destacando certas informações no contexto da sequência de entrada, permitindo que o modelo preste atenção a diferentes aspectos do dado em diferentes _heads_ e aprenda.
{{% /notice %}}

Observações:
  - Por exemplo, _scores_ de atenção (i.e., o produto escalado por $d_k$ entre as _queries_ e as _keys_) são ativações.
  - É importante distinguir ativações de parâmetros (que são os pesos e vieses aprendidos durante o treinamento e não mudam dependendo da entrada.).

> **Definição (Circuito [^olah2020zoom])**.
> Um circuito é um sub-grafo de uma rede neural que consiste em um conjunto de propriedades intimamente ligadas com seus pesos.

{{% notice style="primary" title="Matrizes de pesos de _transformers_" %}}
  O entendimento de circuitos em _transformers_ se baseia fortemente no conhecimento das matrizes de pesos que servem de base para seu funcionamento.
  
  Seja um modelo de _multi-head attention_, ou seja, que realiza várias operações de atenção em paralelo ao mapear _embeddings_ intermediários da dimensão interna do modelo para a dimensão de _head_. Identificaremos objetos específicos de cada _head_ com um sobrescrito ${(\cdot)}^h$. Seja, também,
  - $d_\text{model}$ a dimensão interna do modelo;
  - $d_\text{head}$ a dimensão de cada _head_ do modelo, normalmente definido como $d_\text{head} = d_\text{model} / n_\text{head}$, onde $n_\text{head}$ é o número de _heads_;
  - $d_\text{vocab}$ a dimensão do vocabulário;
  - $n_\text{ctx}$ o número máximo de _tokens_ que o modelo consegue processar (janela de contexto);
  - $W^h_K \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ a matriz de pesos para as _keys_;
  - $W^h_Q \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ a matriz de pesos para as _queries_;
  - $W^h_V \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$ a matriz de pesos para os _values_;
  - $W^h_O \in \mathbb{R}^{d_\text{head} \times d_\text{model}}$ a matriz de pesos para a saída;
  - $W^h_E \in \mathbb{R}^{d_\text{vocab} \times d_\text{model}}$ a matriz de pesos para _embedding_;
  - $W^h_U \in \mathbb{R}^{d_\text{model} \times d_\text{vocab}}$ a matriz de pesos para _unembedding_;
  - $W_\text{pos} \in \mathbb{R}^{n_\text{ctx} \times d_\text{model}}$ a matriz de pesos para _positional embedding_.

  Temos as seguintes matrizes:
  - $W^h_\text{OV} \in \mathbb{R}^{d_\text{model} \times d_\text{model}} = W^h_V W^h_O$, que descreve qual informação se move da fonte até o destino no fluxo residual. Chamaremos de *circuito OV*.
  - $W_E W^h_\text{OV} W_U \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$, que descreve qual informação se move da fonte até o destino do início ao fim. Chamaremos de *circuito OV completo*.
  - $W^h_\text{QK} \in \mathbb{R}^{d_\text{model} \times d_\text{model}} = W^h_Q W^h_K$, que descreve de onde e para onde as informações se movem no fluxo residual. Chamaremos de *circuito QK*. 
  - $W_E W^h_\text{QK} (W_E)^T \in \mathbb{R}^{d_\text{vocab} \times d_\text{vocab}}$, que descreve de onde e para onde as informações se movem entre os _tokens_ do vocabulário. Chamaremos de *circuito QK completo*.
  - $W_\text{pos} W^h_\text{QK} (W_\text{pos})^T \in \mathbb{R}^{n_\text{ctx} \times n_\text{ctx}}$, que descreve de onde e para onde as informações se movem entre os _tokens_ no contexto (i.e., entre as posíções) *circuito QK completo de posições*.
{{% /notice %}}

Observações:
  - Imagine que queremos analisar se uma determina _head_ $h$ está dando mais atenção para o _token_ imediatamente anterior a todo _token_. Conseguiríamos descobrir se isso é verdade ao constatar que o circuito QK completo de posições, $W_\text{pos} W^h_\text{QK} (W_\text{pos})^T$, é uma matriz que possui valores altos nas entradas logo abaixo da diagonal. Ou seja, dá um _score_ alto para o _token_ anterior.
  - Outro comportamento bem comum estudado pela ótica de circuitos é o de indução, no qual o modelo atribui um _score_ maior para sequência de tokens que já apareceram no texto. Ou seja, o modelo aprende a identificar aparições anteriores de um mesmo _token_ e a considerar o próximo _token_ da aparição anterior como um bom candidato a ser o próximo do atual _token_.

## _Heads_ e circuitos de indução

> **Definição (Attention patterns)**.
  Comportamentos que observamos em _heads_ de atenção que descrevem o tipo de relação entre _tokens_ capturadas por um _head_. Destacam-se os seguintes padrões:
>   - **_Head_ de _token_ anterior**: voltam a atenção ao _token_ anterior na sequência;
>   - **_Head_ de _token_ atual**: voltam a atenção ao próprio _token_ na sequência;
>   - **_Head_ de primeiro _token_**: voltam a atenção ao primeiro _token_ da sequência, em geral uma _flag_ de `<|endoftext|>`.

Observações:
  - Em interpretabilidade, temos interesse em detectar esse tipo de comportamento, e compreender que tipo de informação cada _head_ captura.
  - Os dois primeiros padrões podem parecer mais razoáveis. A intuição para as _heads_ de primeiro _token_ é que a primeira posição funciona como uma posição nula para _heads_ que não ativam com frequência.

{{% notice style="primary" title="_Attention patterns_" %}}
  Suponha que estamos computando a atenção em uma sequência:

  `<|endoftext|>` `Eu` `amo` `viajar` `para` `lugares` `legais` `.`

  Temos interesse em visualizar a atenção computada por cada _head_, e produzimos gráficos como os seguintes:

  ![](/handbook/img/attention_pattern.png)

  Cada gráfico representa um possível tipo de padrão de atenção. Para uma _head_ qualquer, poderíamos observar que o padrão obtido é de _head_ de _token_ anterior (I), _head_ de _token_ atual (II), _head_ de primeiro _token_ (III), ou mesmo não observar nenhum padrão, havendo uma distribuição de atenção de outras formas.

  Note que a biblioteca TransformerLens facilita a exploração e visualização dos padrões, e também sua detecção. Detectar automaticamente o padrão de _attention heads_ nos ajuda a quantificar nossas observações sobre processos internos do modelo.
{{% /notice %}}

> **Definição (_Heads_ de indução)**.
> _Heads_ de indução são _heads_ de atenção que realizam um padrão específico, procurando na janela de contexto por exemplos do _token_ atual. Quando o encontram, replicam o próximo _token_ do contexto.

Observações:
  - Na prática, _heads_ desse tipo fazem induções da forma `[a][b] … [a] → [b]`.
  - Não é possível ter _heads_ de indução em modelos de uma camada.
  - Em modelos com _heads_ de indução, dada uma sequência repetida de _tokens_, o modelo consegue prever a segunda parte da sequência. A Figura 1 mostra a melhora da habilidade de predição no momento em que a sequência de _tokens_ começa a se repetir:

> ![](/handbook/img/repeated_seq.png)
> **Figura 1**. Gráfico da log-probabilidade do _token_ correto por posição em uma sequência com repetição completa de _tokens_

Observações:
  - A observação de _heads_ de indução passa a ocorrer conforme aumentamos a escala do modelo. Para 2 bilhões de _tokens_ há ausência de _heads_ de indução e em 4 bilhões observamos essa capacidade muito bem desenvolvida. Chamamos desenvolvimentos repentinos como esse de capacidade emergente [^wei2022emergent-abilities].
  - Capacidades emergentes são bastante interessantes, mas trazem preocupações no ramo de alinhamento de IA, já que são características que não podem ser previstas treinando modelos de pequena escala. O assunto é pauta de muitos estudos, já que existe discordância se a observação dessas características é fruto da utilização de métricas descontínuas [^schaeffer2023emergent-habilities-mirage].

> **Definição (Circuito de indução)**.
> Um circuito de indução é (normalmente) composto pela composição das seguintes _heads_:
>   1. Uma _head_ de _token_ anterior, chamado de circuito QK de _token_ anterior.
>   2. Uma _head_ de indução que possui os dois seguintes mecanismos:
>     1. Atribui um _score_ alto para quando um _token_ $x_i$ é o mesmo token $x_j$ identificado pela _head_ de _token_ anterior, chamado de K-composição.
>     2. Copia o _token_ que vem logo depois de $x_i$ (i.e., atribui um _score_ alto para o _token_ de $x_{i+1}$ na posição $j+1$ quando $x_i = x_j$), chamado de circuito OV de cópia.

Observações:
  - É importante que, nesse ponto, a diferença entre _heads_ de indução e circuitos de indução esteja clara, já que trataremos de diversos circuitos. Um circuito de indução é um circuito composto por um _head_ de _token_ anterior em uma camada passada (responsável por gerar atenção entre a cópia do _token_ atual e seu _token_ seguinte) e uma _head_ de indução. Isto é, circuitos são conjuntos de _heads_ capazes de realizar uma tarefa.

{{% notice style="primary" title="Ferramentas de interpretabilidade" %}}
  Para compreendermos melhor o funcionamento geral do modelo, é importante compreender o impacto de cada componente. Isto é, quanto da performance do modelo em certa tarefa deve ser atribuído a cada componente?

  No contexto de _transformers_, temos interesse em saber o impacto de cada _head_ no resultado. Para isso, construímos ferramentas que auxiliam no processo, por exemplo:

  - *Atribuição de _logits_:* os _outputs_ finais de um modelo são _logits_ provenientes do fluxo residual, que são a soma da contribuição de cada camada. Podemos, portanto, decompor esses _logits_ em valores vindos de cada _head_ e compreender melhor o impacto de diferentes _heads_ no resultado. 
  - *Ablation:* investigação do desempenho do modelo removendo componentes para entender sua contribuição. No contexto de _transformers_, podemos suprimir certas _heads_ alterando seus valores para zero e compreender o impacto da mudança no resultado.
{{% /notice %}}

## Engenharia reversa do circuito de indução

O objetivo de realizar engenharia reversa é entender melhor o funcionamento dos modelos. Queremos saber não apenas que tarefa cada parte de um modelo realiza, mas também o porquê.

Vejamos um exemplo de engenharia reversa para o circuito *OV*. Sabemos que esse circuito é dado por $W_E W^h_\text{OV} W_U$, mas os únicos fatores interpretáveis do circuito são os _tokens_ de _input_ e os _logits_ de _output_. Desejamos analisar a própria matriz de pesos. 

Como estamos tratando de indução, imagine que temos um _input_ `A` `B` ... `A` `B`. Nesse caso, sendo $b$ o _one-hot-encoding_ de `B`, $b^T W_E W^h_\text{OV} W_U$ é o vetor de _logits_ movido do primeiro _token_ `B` para o segundo `A`, utilizado por este como predição. Assim, esperamos ter uma predição com alta probabilidade de `B`.

Vamos quebrar em partes para clarificação:
- $b^T W_E$ é o _embedding_ de `B`
- $b^T W_E W^h_\text{OV}$ é o vetor movido do primeiro _token_ `B` para o segundo `A`
- $b^T W_E W^h_\text{OV} W_U$ é o vetor de _logits_ representando o impacto da _head_ de atenção $h$ na predição do _token_ após o segundo `A`. Representa a cópia de `B` para o segundo `A`.

Como `B` é copiado para o _token_ atual, $b^T W_E W^h_\text{OV} W_U$ resulta em valores altos na diagonal (o elemento $($`B`$, X)$ da matriz deve ser mais alto para $X=$`B`, e portanto temos uma _head_ de _token_ atual).

Observe como a análise detalhada das matrizes de pesos e a engenharia reversa do circuito de indução nos permitem identificar como os modelos de aprendizado profundo utilizam representações internas para realizar previsões. Ao decompor as operações, podemos entender que o _embedding_ de $b$ é transformado e movido pelo circuito e, finalmente, traduzido em _logits_ pela matriz $W_U$.

​O entendimento de como o processo se dá é crucial para melhorar a interpretabilidade e a confiança nos modelos, permitindo ajustes mais precisos nas arquiteturas e pesos para melhorar seu desempenho em tarefas específicas.

{{% bibliography %}}
