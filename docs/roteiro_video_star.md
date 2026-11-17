# Roteiro do vídeo — método STAR (≤ 5 min)

Apresentação: [`apresentacao_video_star.pptx`](apresentacao_video_star.pptx) — 14 slides, **duração total 4:50**. A fala de cada slide também está nas **notas do apresentador** do PPT (696 palavras ≈ 140 palavras/min).

## Como gravar

1. Abra o PPT e inicie em **Apresentação de Slides → Modo do Apresentador** (as notas ficam só na sua tela).
2. Grave a tela com OBS, Loom ou o próprio PowerPoint (**Gravar → Do início**), que já sincroniza áudio e slides.
3. Avance o slide quando o tempo do selo ⏱ (canto superior direito) terminar. Se sobrar fôlego, respire; se atrasar, pule a última frase do slide.
4. Exporte em 1080p, publique (ex.: YouTube não listado) e cole o link no topo do README.

## Roteiro por slide

| # | Tempo | Etapa | Slide | Fala |
|---|---|---|---|---|
| 1 | 0:00–0:10 | Abertura | Degradação silenciosa em Credit Scoring | Olá! Neste vídeo eu apresento o Tech Challenge da Fase 4: como detectar e explicar a degradação silenciosa de um modelo de crédito em produção. |
| 2 | 0:10–0:45 | S — Situation | O modelo não cai — ele erra em silêncio | Imagine uma fintech com um modelo de credit scoring em produção. Ele não dá erro, não cai e responde em milissegundos. Mesmo assim, pode estar errado. Com inflação e novos perfis de clientes, o mundo mudou, mas o modelo continua decidindo como no dia em que foi treinado. No nosso experimento, o sistema ficou cem por cento disponível enquanto a AUC caiu de 0,77 para 0,71. Isso é degradação silenciosa: o prejuízo só aparece no balanço, ou numa reclamação de discriminação. |
| 3 | 0:45–1:15 | T — Task | Missão: a camada de confiabilidade | A tarefa foi construir a camada de sustentação e confiabilidade com três objetivos. Primeiro, interceptar dados ruins antes que cheguem ao modelo. Segundo, detectar drift de dados e de conceito com testes estatísticos e relatórios. Terceiro, garantir compliance com a LGPD, incluindo uma análise causal que explique por que o modelo degradou. Usamos o dataset público Default of Credit Card Clients, com trinta mil clientes e vinte e dois por cento de inadimplência. |
| 4 | 1:15–1:30 | A — Action | Arquitetura: contrato antes do modelo | Esta é a arquitetura. Todo lote passa pelo contrato de dados antes do modelo. Depois vêm a inferência, a detecção de drift, o monitoramento de performance e os alertas, tudo registrado em logs, no MLflow e no dashboard. |
| 5 | 1:30–1:55 | A — Action | Data Contract: lixo não entra no modelo | Treinamos um XGBoost com AUC de 0,77. O contrato, feito em Pandera, tem onze regras rígidas: idade mínima de dezoito anos, limite não nulo e dentro de faixa, pagamentos não negativos, sem duplicatas e modo strict, que rejeita colunas não previstas. Forçamos um lote corrompido, inclusive com uma coluna de CPF vazando: cento e cinquenta e seis violações e ingestão bloqueada. |
| 6 | 1:55–2:15 | A — Action | Simulação: 6 meses de crise econômica | Separamos sessenta por cento dos dados como referência e quarenta por cento como produção, em seis meses. A cada mês aumentamos a inflação nas faturas, a queda nos pagamentos, os atrasos e a entrada de clientes jovens com limite menor. E injetamos concept drift: utilização alta do limite passa a gerar default. |
| 7 | 2:15–2:45 | A — Action | Detecção: 81% das variáveis com drift | O Evidently compara a referência com junho usando Kolmogorov-Smirnov e PSI. Resultado: dezessete de vinte e uma variáveis com drift, oitenta e um por cento. As maiores mudanças estão nos atrasos de pagamento, com PSI acima de dois, e na idade, com PSI de 0,6. O limite de alerta é 0,2. Um detalhe importante: todos esses dados passaram no contrato. Drift é dado válido, porém diferente. |
| 8 | 2:45–3:15 | A — Action | Observabilidade: um job por mês | Cada mês é processado por uma execução agendada, uma janela a cada dez minutos na demonstração. O job valida o contrato, faz a inferência, roda os testes de drift e de performance e dispara alertas. Tudo vai para logs JSON centralizados com identificador de execução, para o MLflow, com o modelo registrado e um run por mês, e para métricas no formato Prometheus. O painel mostra latência, volume e status de cada tarefa. |
| 9 | 3:15–3:37 | A — Action | Governança LGPD por design | Na governança: o identificador é pseudonimizado com HMAC, sexo e estado civil ficam fora do modelo e só servem para auditoria de viés. A base legal é a proteção do crédito, artigo sétimo, inciso dez, com direito à revisão do artigo vinte, e há um plano de retenção por tipo de dado. |
| 10 | 3:37–3:55 | R — Result | Resultado: 27 alertas desde fevereiro | E o resultado: seis meses processados, um lote bloqueado e vinte e sete alertas automáticos. No mapa de calor, o drift aparece já em fevereiro nos atrasos de pagamento e esquenta mês a mês. |
| 11 | 3:55–4:10 | R — Result | Impacto no negócio: aprovação despenca | No modelo, a AUC cai de 0,77 para 0,71, a aprovação despenca de setenta e cinco para cinquenta e seis por cento e a inadimplência real sobe de vinte e dois para trinta e cinco por cento. |
| 12 | 4:10–4:28 | R — Result | Causa raiz: o concept drift | Para achar a causa, desligamos cada mecanismo e medimos de novo. A causa raiz é o concept drift: explica oitenta por cento da queda de AUC e é invisível ao monitor de variáveis, porque o PSI do score fica em 0,01. O covariate shift muda quanto aprovamos, não o ranking. |
| 13 | 4:28–4:42 | R — Result | Lições aprendidas e prescrição | Três lições: contrato e drift são camadas diferentes; KS com amostra grande gera falso positivo, por isso o PSI é a métrica principal; e sem monitorar rótulos o concept drift passa despercebido. A prescrição: retreinar com dados recentes e recalibrar o limiar. |
| 14 | 4:42–4:50 | Encerramento | Obrigado! | O código, os relatórios e toda a documentação estão no GitHub. Obrigado! |

## Critérios de avaliação cobertos

| Critério (peso) | Slides |
|---|---|
| Validação de dados (25%) | 4, 5 |
| Detecção de drift (25%) | 6, 7, 10 |
| Observabilidade e logs (20%) | 8, 10 |
| Governança LGPD (15%) | 9 |
| Vídeo STAR, impacto (15%) | 2–3 (S/T), 11–13 (R) |

## Demonstração ao vivo (opcional)

Se quiser mostrar o sistema rodando em vez das imagens: no slide 5 rode `credit-risk validate` no terminal; no slide 7 abra `reports/drift/data_drift_2026-06.html`; no slide 10 abra `reports/monitoring_dashboard.html`. Isso adiciona cerca de 20 s — compense encurtando a fala do slide 8.
