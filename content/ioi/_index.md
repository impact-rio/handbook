---
title: 4. Identificação de Objetos Indiretos (IOI)
weight: 4
aliases: 4
---

Um circuito bastante conhecido no GPT-2 é o de identificação de objetos indiretos. Como o próprio nome indica, a tarefa performada por esse circuito consiste em completar objetos indiretos em sentenças. Por exemplo, na sentença `João e Maria fugiram para a floresta. João confiava em`, é esperado que o modelo complete-a com _token_ `Maria`.

É bastante razoável questionar a importância de estudar esse circuito. Por que essa tarefa seria relevante? As razões do laboratório de Redwood Research para o desenvolvimento do artigo "_Interpretability in the Wild_" [^wang2022ioi] se baseiam no fato de que o circuito é um dos primeiros a serem desenvolvidos (pela frequência dessa estrutura gramatical) e também na facilidade de mensurar essa capacidade.

Esperamos que a sequência da sentença `João confiava em` seja `Maria`, e não `João`, e podemos medir a habilidade de resolução do problema calculando a diferença de probabilidade entre esses dois _tokens_. Além disso, a diferença de _logits_ é bastante explicativa, pois evidencia a capacidade do modelo de identificar o objeto indireto da sentença, não apenas a repetição de nomes.

Nesse capítulo, trataremos simultaneamente do circuito de Identificação de Objetos Indiretos e de ferramentas importantes de _Mechanistic Interpretability_ utilizadas para investigar e identificar esse circuito.

## Técnicas para identificações de circuitos

> **Definição (atribuição direta de _logits_)**.
> Aplicação de um _unembedding_ e _LayerNorm_ (normalização da saída de uma camada) diretamente ao _output_ de um nó, realizada com objetivo de compreender a contribuição de cada nó para a predição do próximo _token_.]

Observações:
  - No contexto de IOI, utilizamos a atribuição direta à diferença de _logits_.
  - O mesmo processo pode ser realizado para obter a decomposição de importância em diferentes níveis: para cada bloco de atenção, para cada _head_ de atenção.
  - O DLA não leva em conta que nós posteriores dependem dos anteriores, e _outputs_ de nós iniciais podem ser desconsiderados por nós posteriores, e portanto ter pouca ou nenhuma significância para o resultado final. 
  - É considerado um processo de _denoising_, pois removemos ruído ao decompor as saídas em contribuições de cada nó e analisar quais são mais relevantes.

> **Definição (_patching_ de ativações [^meng2022activationpatching])**.
> O _Patching_ de ativações é o processo de submeter o modelo a uma execução correta, em que obtemos uma predição de _token_ certa, e uma execução corrompida, em que não obtemos o _token_ certo. Posteriormente, há uma intervenção na execução corrompida em alguma ativação para inserção (_patching_) da ativação equivalente da execução correta. Ao final, mede-se o impacto da mudança na predição de _tokens_.

Observações:
- É considerado um processo de _noising_, pois adicionamos ruído ao experimentar diferentes _patchings_ e avaliar o efeito na probabilidade do _token_ correto.
- Podemos realizar _patching_ em _heads_ de atenção, em uma MLP, ou mesmo em valores do fluxo residual.
- No contexto de IOI, podemos considerar uma execução correta `João e Maria foram à floresta. João deu um pão a`, e uma execução corrompida `João e Maria foram à floresta. Maria deu um pão a`. Realizar um _patching_ de ativação nos permite compreender que partes do modelo estão corretamente identificando objetos indiretos

> ![](../img/activation-patching.png)
> Processo corrompido inicial e _patching_ de ativação do processo correto no processo corrompido.

> **Definição (_patching_ de conexões)**.
> Processo de submeter o modelo a uma execução correta, em que obtemos uma predição de _token_ certa, e uma execução corrompida, em que não obtemos o _token_ certo. Posteriormente, há uma intervenção na execução corrompida em alguma interação entre ativações (uma aresta) para inserção (_patching_) da conexão equivalente da execução correta. Ao final, mede-se o impacto da mudança na predição de _tokens_.

Observações:
  - Procura-se entender o que acontece quando o _input_ direto da _head_ `[A]` para a _head_ `[B]` é trocado pelo _input_ de `[A']`, o valor que a _head_ teria sob outra distribuição, mantendo igual todo o resto.
  - O objetivo desse tipo de _patching_ é mais específica: busca-se compreender a importância do circuito formado pela conexão entre duas _heads_ de atenção.
  - A implementação é complexa, já que a atualização desse valor ocorre dentro de processos maiores do fluxo residual, e não podemos alterar nenhuma outra interação.

> ![](../img/path-patching.png)
> Processo corrompido inicial e _patching_ de conexões do processo correto no processo corrompido.

## Funcionamento do circuito de IOI

Através daas técnicas apresentadas anteriormente, foi possível compreender o funcionamento do circuito de IOI [^wang2022ioi]. Agora, traremos uma visão intuitiva do processo.

Vamos retomar a analogia da [seção de Transformers de atenção](../transformers). Imagine, novamente, a fila de pessoas em que cada um possui um _token_, e seu objetivo é descobrir o _token_ da pessoa a sua frente. Retomamos as mesmas regras para realizar perguntas: cada pessoa pode passar perguntas para quem está atrás de si na fila e nunca para frente, e qualquer um atrás pode escolher responder, passando informação para quem fez a pergunta. 

Teremos agora a  frase `João e Maria foram a floresta. João deu um pão para Maria`. Nesse caso, como tratamos de Identificação de Objeto Indireto, a pessoa que possui o _token_ `para` precisa concluir que o _token_ a sua frente é `Maria`. 

-  Cada pessoa na fila representa um vetor no fluxo residual. Inicialmente, só possuem informações do seu próprio _token_, mas conforme perguntam e recebem respostas passam a armazenar mais informações.
- A operação executada por um _head_ de atenção é representada por um par pergunta-resposta, onde a pergunta representa a query, quem responderá é determinado pela _key_ e a resposta será o _value_

Agora podemos partir para o circuito IOI. Cada _bullet_ representa uma classe de _heads_ de atenção que faz parte da identificação do circuito:
 - **_Heads_ de _tokens_ duplicados**: a pessoa com o segundo _token_ `João` pergunta "Alguém mais tem um _token_ `João`?" e recebe a resposta do primeiro _token_ `João`, assim como sua posição. Agora ele sabe que o seu _token_ é repetido e também a localização da primeira ocorrência.
-  **_Heads_ de inibição-S**: o _token_ atual (`para`) pergunta "Que nomes são repetidos?", e recebe resposta do segundo _token_ `João`, informando que esse _token_ é repetido e que o primeiro `João` está na 1ª posição.
-  **_Heads_ de mover nomes**: o _token_ atual (`para`) pergunta "Alguém tem um nome que não seja `João` e não esteja na 1ª posição?" e recebe resposta do _token_ `Maria`. Utilizamos essa resposta como previsão.
]

> **Definição (circuito de IOI)**.
> Circuito composto por _heads_ de _tokens_ duplicados (DTH), _heads_ de inibição-S (SIH) e _heads_ de mover nomes (NMH) que realiza a predição de objetos indiretos.

Observações:
  - O circuito pode ter variações. Por exemplo, o _token_ seguinte ao primeiro `João` (e) poderia já ser ter informações do _token_ atrás de si (_head_ de _token_ anterior), e portanto responder à _query_ do DTH.
  - Existem _heads_ de mover nomes de _backup_. Isto é, quando realizamos _ablation_ nas NMH, estas _heads_ passam a realizar o trabalho. Uma possível explicação é que essa capacidade tenha surgido para lidar com o _dropout_ (regularização que remove aleatoriamente neurônios da rede a cada iteração), garantindo assim que o modelo ainda possa funcionar adequadamente mesmo quando algumas partes são "desativadas".

{{% bibliography %}}
