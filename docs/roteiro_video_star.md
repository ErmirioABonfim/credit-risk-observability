# Roteiro do vídeo — método STAR (≤ 5 min)

**Duração-alvo: 4:45.** Gravar tela + webcam (OBS ou Teams/Loom), 1080p. Fale em ritmo calmo (~140 palavras/min).

## Preparação (antes de apertar REC)

1. Rodar uma vez `credit-risk run-all` (≈ 6 min) para ter todos os artefatos prontos.
2. Abrir as abas/janelas nesta ordem:
   - **A1** README no GitHub (seção 2 — Arquitetura, com o diagrama)
   - **A2** Terminal no diretório do projeto, com fonte grande (≥ 18 pt), limpo
   - **A3** `src/credit_risk/contracts.py` no VS Code (regras R1–R11 visíveis)
   - **A4** `reports/drift/data_drift_2026-06.html` (Evidently)
   - **A5** `reports/monitoring_dashboard.html`
   - **A6** MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000
   - **A7** README seção 7 (Governança/LGPD e Causalidade)
3. Desativar notificações; zoom do navegador em 110–125%.

---

## S — Situation (0:00 – 0:40) · *tela: A1, webcam grande*

> "Uma fintech tem um modelo de credit scoring em produção. Ele não dá erro, não cai, responde em milissegundos…
> e mesmo assim pode estar errado. Com inflação e novos perfis de clientes, o mundo mudou, mas o modelo continua
> decidindo como se fosse o ano em que foi treinado. Isso é **degradação silenciosa**: o modelo reprova bons
> clientes, aprova clientes de risco, e ninguém percebe até o prejuízo aparecer no balanço — ou numa
> reclamação de discriminação.
> Usamos o dataset público *Default of Credit Card Clients*, com 30 mil clientes e 22% de inadimplência."

## T — Task (0:40 – 1:15) · *tela: A1 rolando para a tabela da seção 1*

> "Nossa missão foi construir a **camada de sustentação e confiabilidade** com três objetivos:
> primeiro, **interceptar dados ruins** antes que cheguem ao modelo;
> segundo, **detectar drift** — de dados e de conceito — com testes estatísticos e relatórios;
> terceiro, **garantir compliance** com a LGPD, incluindo base legal, retenção, viés e uma análise causal
> que explique *por que* o modelo degradou."

## A — Action (1:15 – 3:30)

### A.1 Data Contract (1:15 – 1:55) · *tela: A3 → A2*

> "Treinamos um XGBoost baseline com AUC de 0,77 no holdout. Antes de qualquer treino ou inferência, todo lote
> passa por um **contrato Pandera** com 11 regras rígidas: idade ≥ 18, limite não nulo e dentro de faixa,
> pagamentos não negativos, sem duplicatas — e `strict=True`, que rejeita qualquer coluna não prevista."

**Ação na tela:** no terminal, rodar:
```bash
credit-risk validate
```
> "Aqui eu forço um lote corrompido: nulos, menor de idade, duplicatas e até uma coluna de **CPF** vazando.
> Resultado: **156 violações, ingestão bloqueada**, lote em quarentena. Lixo não entra no modelo."

### A.2 Simulação e detecção de drift (1:55 – 2:45) · *tela: README 5.2 → A4*

> "Separamos 60% como **Referência** e 40% como **Produção**, fatiada em seis meses. De janeiro a junho
> injetamos, com intensidade crescente: inflação nas faturas, queda nos pagamentos, mais atrasos, clientes mais
> jovens com limite menor — isso é data drift — e um **concept drift**: clientes com utilização alta do limite
> passam a dar default. Janeiro é o controle, sem drift."

**Ação na tela:** abrir A4 (Evidently junho), mostrar o topo "Dataset Drift detected" e a tabela de colunas.
> "O Evidently, com KS e PSI, aponta **81% das features com drift** em junho. As piores: atrasos de pagamento,
> com PSI acima de 2, e idade, com PSI de 0,6 — o limite de alerta é 0,2."

### A.3 Observabilidade (2:45 – 3:15) · *tela: A6 → A2*

> "Cada mês é processado por um job agendado — uma execução por janela, via Task Scheduler ou GitHub Actions.
> Ele valida o contrato, faz a inferência, roda os testes de drift e de performance e gera alertas.
> Tudo vai para **logs JSON centralizados**, para o **MLflow** — com o modelo registrado e um run por mês —
> e para métricas no formato **Prometheus**."

**Ação na tela:** no MLflow, abrir o experimento, selecionar os runs `monitor-*` e mostrar o gráfico de `perf_auc`
e `share_drifted_evidently` por step. Opcional (5 s): `Get-Content logs/pipeline.jsonl -Tail 3`.

### A.4 Governança (3:15 – 3:30) · *tela: A7*

> "Na governança: o identificador é **pseudonimizado com HMAC**, sexo e estado civil **ficam fora do modelo**
> e só são usados para auditoria de viés. A base legal é a **proteção do crédito**, art. 7º, inciso X, com direito
> à revisão da decisão automatizada do art. 20, e um plano de retenção por tipo de dado."

## R — Result (3:30 – 4:45) · *tela: A5 (dashboard)*

> "O resultado está neste dashboard." *(rolar devagar)*

1. **KPIs (3:35):** "6 janelas processadas, 1 lote bloqueado, **27 alertas** e a AUC caindo de 0,77 para 0,71."
2. **Heatmap PSI (3:50):** "O drift aparece **já em fevereiro** em `pay_2` e `pay_3` e vai esquentando mês a mês."
3. **Saúde do modelo (4:00):** "A aprovação despenca de 75% para 56% e a inadimplência real sobe de 22% para 35%."
4. **Análise causal (4:10):** "Com intervenções contrafactuais, desligando cada mecanismo, descobrimos que a
   **causa raiz é o concept drift**: ele explica 80% da queda de AUC — e é **invisível** ao monitor de features,
   o PSI do score fica em 0,01. O covariate shift muda quanto aprovamos, mas quase não muda o ranking."
5. **Lições (4:25):** "Três lições: contrato e drift são camadas diferentes — os dados de junho eram válidos
   e o modelo piorou mesmo assim; KS com amostra grande gera falso positivo, por isso usamos PSI como métrica
   principal; e sem monitorar rótulos, mesmo atrasados, o concept drift passaria despercebido.
   A prescrição é retreinar com dados recentes e recalibrar o limiar."

**Encerramento (4:40 – 4:45):** "Código, relatórios e documentação estão no repositório. Obrigado!"

---

## Checklist de critérios cobertos no vídeo

| Critério (peso) | Momento |
|---|---|
| Validação de dados (25%) | A.1 — lote corrompido bloqueado ao vivo |
| Detecção de drift (25%) | A.2 — relatório Evidently de junho |
| Observabilidade e logs (20%) | A.3 + Result — MLflow, logs, dashboard, alertas |
| Governança LGPD (15%) | A.4 — pseudonimização, base legal, retenção, viés |
| Clareza STAR / impacto (15%) | estrutura S-T-A-R + análise causal e lições |

**Dicas:** grave cada bloco separado e junte na edição; se passar de 5:00, corte primeiro o trecho de logs
do terminal (A.3) e depois o detalhamento das lições.
