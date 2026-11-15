# Análise causal — janela 2026-06

AUC controle (sem drift): **0.7874** → AUC observada: **0.7067** (queda 0.0807). Causa raiz principal: **concept_default**.

| Cenário | AUC | Default real | Default previsto | PSI do score |
|---|---|---|---|---|
| sem_drift (controle) | 0.7874 | 22.5% | 25.1% | 0.009 |
| todos os mecanismos (observado) | 0.7067 | 35.4% | 44.5% | 0.6989 |
| apenas covariate shift (P(X)) | 0.7713 | 22.5% | 44.5% | 0.6989 |
| apenas concept drift (P(y|X)) | 0.7186 | 32.0% | 25.1% | 0.009 |
| + inflation_bills | 0.7877 | 22.5% | 25.0% | 0.0416 |
| + new_young_low_limit | 0.7844 | 22.5% | 25.7% | 0.0437 |
| + lower_payments | 0.7883 | 22.5% | 26.0% | 0.0684 |
| + payment_delays | 0.7784 | 22.5% | 34.8% | 0.21 |
| + concept_default | 0.7186 | 32.0% | 25.1% | 0.009 |
| todos - inflation_bills | 0.6998 | 34.0% | 42.0% | 0.5683 |
| todos - new_young_low_limit | 0.7136 | 33.9% | 41.5% | 0.5655 |
| todos - lower_payments | 0.6977 | 35.4% | 37.9% | 0.4152 |
| todos - payment_delays | 0.7206 | 35.4% | 27.5% | 0.2341 |
| todos - concept_default | 0.7713 | 22.5% | 44.5% | 0.6989 |

| Mecanismo | Tipo | Queda AUC isolada | AUC recuperada se removido | Δ default previsto | Δ default real |
|---|---|---|---|---|---|
| concept_default | concept | 0.0688 | 0.0646 | +0.0% | +9.6% |
| payment_delays | covariate | 0.009 | 0.0139 | +9.7% | +0.0% |
| new_young_low_limit | covariate | 0.003 | 0.0069 | +0.6% | +0.0% |
| inflation_bills | covariate | -0.0003 | -0.0069 | -0.1% | +0.0% |
| lower_payments | covariate | -0.0009 | -0.009 | +0.9% | +0.0% |
