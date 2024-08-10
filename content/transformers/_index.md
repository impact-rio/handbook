---
title: 2. Transformers e atenção
weight: 2
aliases: 2
---

A arquitetura de _transformer_ [^VaswaniEtAlAttentionAll2017] revolucionou o campo de processamento de linguagem natural (PLN) e de aprendizado profundo e, atualmente, é o estado-da-arte em tarefas de linguagem. Essa arquitetura é baseada no conceito de atenção, que permite ao modelo descobrir relações entre as diversas partes de um determinado texto.

## Arquitetura

A arquitetura de _transformer_ pode variar dependendo da tarefa a ser desempenhada. Nessas anotações, vamos escolher focar na arquitetura que serve de base para os modelos GPT [^RadfordEtAlImprovingLanguage2018] [^RadfordEtAlLanguageModels2019]. Nesse tipo de modelo, o objetivo central é prever qual seria o próximo _token_ mais provável dada uma sequência de _tokens_, característica de modelos autoregressivos.

> **Definição (Tokenização [^BishopBishopDeepLearning2024])**.
A tokenização é o processo de transformar um texto em uma representação semântica compreendido como uma sequência de _tokens_. Os _tokens_ podem ser palavras, fragmentos de palavras, pontuações ou outros grupos de caracteres. O conjunto de todos os _tokens_ $\mathcal{X}$ de um modelo é chamado de vocabulário.

{{% notice style="primary" title="Byte-pair encoding" %}}
  O algoritmo de _byte-pair encoding_ é a técnica de _tokenização_ utilizada pelo modelo GPT-2. O processo do algoritmo compreende em realizar agrupamentos de _tokens_ dois a dois até atingir um critério de parada 

  Podemos, por exemplo, partir de um conjunto inicial de _tokens_ composto pelos caracteres do texto "banana":
  `b`, `a`, `n`, `a`, `n`, `a`. 

  Para cada iteração do algoritmo, contamos a frequência de todos os pares de _tokens_ adjacentes do texto, e substituímos o par mais frequente por um novo _token_ composto pela concatenação deles. No nosso caso, teríamos um texto tokenizado como `b` `a` `na` `na`.

  Repetimos o passo anterior até atingir o critério de parada (um critério plausível é atingir determinado número de _tokens_ no vocabulário).
{{% /notice %}}

> **Definição (modelos autoregressivos [^BishopBishopDeepLearning2024])**.
> Seja $\mathcal{X}$ um conjunto de _tokens_ e $\mathcal{X}^N$ um espaço de sequências de _tokens_ de tamanho $N$. Um modelo autogressivo $p : \mathcal{X}^N 
\to [0, 1]$ induz uma distribuição de probabilidade sobre $\mathcal{X}^N$ atribuindo probabilidade a toda sequência $\mathbf{x} = (x_1, ..., x_N) \in \mathcal{X}^N$ da seguinte maneira:
> $$
  p(x_1, ..., x_N)
  = p(x_1) \prod_{n = 2}^N p(x_n | x_1, ..., x_{n-1}).

> **Definição (_transformer_ [^VaswaniEtAlAttentionAll2017])**.
> A arquitetura original de _transformer_ possui uma estrutura _encoder-decoder_. O _encoder_ mapeia uma sequência de tokens $\mathbf{x} = (x_1, ..., x_n)$ em uma sequência de representações contínuas $\mathbf{z} = (z_1, ..., z_n)$, vetores de alta dimensão que capturam informações semânticas e contextuais dos _tokens_ de entrada em um espaço contínuo, que chamamos de _embedding_. A sequência $\mathbf{z}$ posteriormente é utilizada pelo _decoder_ para gerar uma sequência de _tokens_ de saída $\mathbf{y} = (y_1, ..., y_m)$ em conjunto a um novo _input_, geralmente em tarefas _seq2seq_. Essas operações são realizadas utilizando o mecanismo de atenção.

Observações:
  - Tarefas _seq2seq_ são aquelas em que realizamos traduções de um contexto para outro, e dependemos da estrutura completa _encoder-decoder_. Exemplos de tarefas _seq2seq_ são a tradução de texto de uma linguagem para outra e resumo de textos, em que inserimos uma entrada no modelo e esperamos como saída uma versão resumida.

> ![Testando](/handbook/img/transformer_architecture.jpeg "Testando")
> {#fig:transformer-architecture}
> **Figura 1**. Representação gráfica da arquitetura de transformer do artigo _original_ [^VaswaniEtAlAttentionAll2017]

> ![](/handbook/img/gpt_1_architecture.png)
> {#fig:gpt-1-architecture}
> **Figura 2**. Representação gráfica da arquitetura baseada em _transformer_ do GPT-1 [^RadfordEtAlImprovingLanguage2018]

Observações:
  - Uma intuição inicial do modelo _encoder-decoder_ é a ideia de utilizar um _encoder_ para capturar o contexto entre sequências de tokens de entrada utilizando atenção, produzindo _hidden states_ (representações intermediárias). Mais tarde, o _decoder_ é usado para gerar sequências de _tokens_ com base no último _hidden state_ produzido pelo _encoder_, e recursivamente adicionando o _output token_ produzido pelo _decoder_ a cada _time step_.
  - A **figura 1** ilustra a **definição de transformer**, mostrando os blocos de _encoder_ e _decoder_ da arquitetura de _transformer_. No entanto, o GPT (ilustrado na **figura 2**) se baseia em uma arquitetura que utiliza apenas o _decoder_ da arquitetura original. Isto porque o GPT mapeia os _tokens_ de uma sentença em um vetor de probabilidade para cada _token_ do vocabulário.
  - Nem todos os _transformers_ são _encoder-decoder_, já que essa arquitetura foi desenvolvida para situações em que desejamos mudar de um contexto para outro (isto é, temos uma tarefa Seq2Seq), como traduções. Para tarefas de classificação, por exemplo, utilizar apenas um _encoder_, enquanto em tarefas de geração de texto no mesmo um contexto, um _decoder_ pode ser suficiente.


## Componentes e técnicas

> **Definição (embedding)**.
> _Embedding_ é a representação vetorial dos _tokens_ de um texto em um espaço latente de alta dimensão $\mathbb{R}^{d_\text{embed}}$, sendo $d_\text{embed}$ a dimensão escolhida para o espaço de _embedding_.

Observações:
  - Modelos como o GPT aprendem uma matriz de pesos $W_E$ (também chamada de tabela de consulta de _embeddings_), que transforma a versão vetorial _one-hot-encoded_ de um _token_ $x_i \in [0, 1]^{d_\text{vocab}}$ (onde somente a $i$-ésima entrada de $x_i$ é igual a 1) em um vetor de dimensão $1 \times d_\text{embed}$:
  - Tabelas de consulta de _embeddings_ são obtidas a partir de modelos pré-treinados de _word embedding_, como o Word2Vec [^MikolovEtAlWord2Vec2013].
  - É esperado que o modelo consiga aprender a representar cada característica semântica de um _token_ em uma dimensão do espaço latente, como será discutido na @sec:superposition. No entanto, é possível que o modelo acabe representando um número maior de propriedades do que o número de dimensões do espaço latente, fenômeno conhecido como superposição [^elhage2022superposition]. 
]

> **Definição (positional embedding)**.
>  _Positional embedding_ é um _embedding_ adicional com informações posicionais de cada _token_ adicionado ao _embedding_ do texto.

Observações:
 - A proposta do _Transformer_ é reconhecer que as posições não são o único determinante de relevância entre _tokens_, como ocorre nas convoluções. No entanto, a linguagem natural conta com localidade (adjetivos, por exemplo, tendem a se referir ao substantivo mais próximo na sentença), e essa informação é útil ao modelo.

> **Definição (fluxo residual)**.
  Componente de modelos de _deep learning_ em que a entrada $\mathbf{x}_{i-1}$ de uma camada intermediária $i$ é somada com sua saída $f_i (\mathbf{x}_{i-1})$, permitindo a preservação de informações anteriores:
  $$
  \mathbf{x}_{i} = f_i (\mathbf{x}_{i-1}) + \mathbf{x}_{i-1}.
  $$

Observações:
  - Se a função ideal a ser aprendida por essa camada for a função identidade, essa implementação tira a necessidade de $f_i$ aprender como reconstruir totalmente a entrada, já que só seria necessário garantir que, para todo $\mathbf{x}_i$,  $f_i (\mathbf{x}_{i-1}) = 0$.
  - Essa técnica também ajudar a mitigar problemas de _vanishing gradient_ durante o treinamento, isto é, ao realizarmos sempre a soma de saídas de cada camada, evitamos um problema comum em redes neurais profundas em que gradientes das camadas iniciais se tornam extremamente pequenos durante o _backpropagation_ (conforme os gradientes são propagados para trás por várias camadas, eles podem diminuir exponencialmente, interrompend).
  - A arquitetura do _Transformer_ utiliza um fluxo residual, essencial para sua memória e fluxo de informações, mantendo informações das camadas anteriores. Assim, os valores no fluxo representam o acúmulo de todas as inferências feitas até aquele ponto.
]


{{% notice style="primary" title="_Keys_, _queries_ e _values_" %}}
  As _keys_ ($K$), _queries_ ($Q$) e _values_ ($V$) são elementos fundamentais em mecanismos de atenção. Eles são obtidas através de projeções lineares, onde os _embeddings_ dos _tokens_ são multiplicados por matrizes de projeção $W^K$, $W^Q$ e $W^V$ aprendidas durante o treinamento.
  As *keys* representam o _token_ de origem, as *queries* representam os _tokens_ de destino e os *values* representam a semântica e contexto dos _tokens_.

  $W^K$, $W^Q$ e $W^V$ são matrizes de pesos que são ajustadas durante o treinamento. Cada uma dessas matrizes tem dimensões específicas que dependem da dimensão do _embedding_ de entrada ($d_\text{embed}$) e da dimensão desejada para keys, queries e values ($d_k$, $d_q$, $d_v$).
{{% /notice %}}

{{% notice style="primary" title="Uma analogia" %}}
  Façamos uma analogia para o funcionamento de um _transformer_ para clarificarmos o conceito de _keys_, _queries_ e _values_. Imagine uma fila de pessoas, em que cada um possui um _token_, e seu objetivo é descobrir o _token_ da pessoa a sua frente. Cada pessoa pode passar perguntas para quem está atrás de si na fila (jamais para frente), e qualquer um atrás pode escolher responder, passando informação para quem fez a pergunta. 

  Portanto, para a frase `O João entregou um pão para Maria.`, a primeira pessoa possui o _token_ `O`, a segunda possui o _token_ `João`, e assim por diante

  ![](/handbook/img/fila.png)

  -  Cada pessoa na fila representa um vetor no fluxo residual. Inicialmente, só possuem informações do seu próprio _token_, mas conforme perguntam e recebem respostas passam a armazenar mais informações.
  - A operação executada por um _head_ de atenção é representada por um par pergunta-resposta:
    - A pessoa que pergunta é o _token_ de destino, as pessoas que respondem são os _tokens_ de origem.
    - A pergunta é a _query_
    - A informação que determina quem responde à pergunta é a _key_
    -  A informação que é passada de volta para quem fez a pergunta é o _value_.

  Nesse contexto, um processo razoável seria o seguinte:
  - A pessoa com o terceiro _token_ (`entregou`) pergunta "Alguém tem um _token_ com um sujeito?", realizando uma _query_.
  - A primeira e a segunda pessoas da fila (únicos _tokens_ anteriores a `entregou`) acordam entre si que entre seus dois _tokens_, `João` é o único que representa um sujeito, decisão que representa a influência da _key_.
  - A pessoa com o _token_ `João` informa o _token_ atual que ele próprio é o sujeito, que portanto está na segunda posição. Essa informação repassada representa o _value_.
{{% /notice %}}


> **Definição (atenção [^VaswaniEtAlAttentionAll2017])**.
  Atenção é uma função que mapeia uma _query_ e pares de _key-value_ em uma saída, calculada como a soma ponderada dos valores por uma medida de compatibilidade da _query_ com a _key_ correspondente.
  A função de atenção mais utilizada em _transformers_ é a atenção do produto escalonado, definida como:
  $$
  \text{Attention}(Q, K, V) = \text{softmax}((Q K^T) / \sqrt(d_k)) V.
  $$

Observações:
  - O produto $Q K^T$ é escalado pela raiz quadrada de $d_k$ (dimensão dos vetores _keys_ e _values_) para evitar que o resultado cresça muito para valores altos de $d_k$, o que poderia levar a gradientes muito próximos de 0.
  - O objetivo da atenção é realizar o fluxo de informação entre os _tokens_ do _input_. Isto é, para um _token_ de origem (_key_) calculamos a atenção para todos os _tokens_ de destino (_queries_).
  - Na arquitetura original do Transformer e em diversas aplicações até hoje, utiliza-se a atenção causal (atenção com _masking_ para _tokens_ futuros), impedindo que informação de _tokens_ futuros influenciem _tokens_ passados. Em outras aplicações, como o BERT [^DevlinEtAlBERT2018], utiliza-se atenção bidirecional, que permite o fluxo de informação nos dois sentidos.
  - Normalmente, ao invés de realizarmos uma única aplicação da função de atenção, computamos várias atenções em paralelo, concatenando os resultados e projetando o resultado de volta para a dimensão do modelo. Essa técnica é conhecida como _multi-head attention_ e visa capturar diferentes dependências entre os _tokens_.

> **Definição (multi-Layer Perceptron (MLP))**.
  Arquitetura de rede neural que consiste em múltiplas camadas totalmente conectadas de nós com ativações não-lineares. Obtemos o _output_ do nós $k$ através do _input_ $z$, do peso $w_k$ e de uma função de ativação $g$:
  $$
  z_k = g \left( \sum_j w_{j k} z_j \right)
  $$

Observações:
  -  No contexto de _transformers_, tratamos de rede neural padrão com uma única camada oculta e uma função de ativação não linear. Durante a MLP, não há mais fluxo de informação entre os tokens, e opera-se em cada posição do fluxo residual de forma independente, com os mesmos parâmetros.
  - A intuição é que a MLP é responsável por armazenar o conhecimento adquirido pela camada de atenção, liberar espaço no fluxo residual e introduzir não-linearidade, permitindo que o modelo capture padrões complexos nos dados.

> **Definição (amostragem)**.
  Processo de seleção de um subconjunto representativo de dados de um conjunto de dados maior. No contexto de Machine Learning, tratamos da obtenção dos _outputs_ esperados para a tarefa a partir dos resultados obtidos em modelos.

Observações:
  - Em _transformers_ autoregressivos, quando obtemos os _logits_, podemos amostrar simplesmente utilizando o _token_ com maior probabilidade como predição. Essa forma de amostragem, no entanto, gera _outputs_ repetitivos. Amostrar a partir de uma distribuição de probabilidade dos _logits_, por outro lado, leva à perda de coerência das frases geradas.
  - Assim, utiliza-se técnicas como o _Top-K Sampling_, em que amostramos a partir das _k_ probabilidades mais altas.

{{% notice style="primary" title="Arquitetura do _transformer decoder-only_" %}}
  A arquitetura do _transformer_ autoregressivo utilizado tem a seguite abstração de alto nível:

  ![](/handbook/img/transformer_decoder.png)

  Assim, o processo de treinamento do modelo se dá reunindo os conceitos definidos até então. A partir da tokenização dos _inputs_ e do subsequente _embedding_ dos _tokens_, temos os _inputs_ em forma de vetores de informação. Em seguida passamos pelos blocos residuais, compostos por:

  - Um _attention block_, em que o resultado de cada _attention head_ adicionado ao fluxo residual:

  $$
  x_{i+1} = x_i + \sum_{h \in H_i}h(x_i)
  $$

  - Um _feed-forward block_, isto é, um MLP, em que o resultado também é adicionado ao fluxo residual:

  $$
  x_{i+2} = x_ei+1e + m{x_{i+1}}
  $$

  Por fim, realizamos um _unembedding_, que converte nosso fluxo residual em _logits_. Um _logit_ é um valor real que representa a propensão relativa de cada token do vocabulário ser o próximo na sequência. Para transformar esses _logits_ em probabilidades, aplicamos a função _softmax_, que normaliza os _logits_ para que a soma das probabilidades seja 1. Dessa forma, cada probabilidade resultante indica a chance de um _token_ específico ser o próximo na sequência, e escolhemos a forma de amostragem de modo que haja variabilidade.
{{% /notice %}}

{{% bibliography %}}
