---
marp: true
theme: uncover
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  section {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fafafa;
    color: #333;
    padding: 30px 50px 20px;
    font-size: 28px;
  }
  h1 {
    color: #0066cc;
    font-weight: 700;
    font-size: 1.5em;
    border-bottom: 3px solid #0066cc;
    padding-bottom: 8px;
    margin-bottom: 15px;
  }
  h2 { color: #0066cc; font-weight: 600; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.7em;
    margin: 10px 0;
  }
  th {
    background: #0066cc;
    color: white;
    padding: 8px 12px;
    text-align: left;
  }
  td {
    padding: 6px 12px;
    border-bottom: 1px solid #ddd;
  }
  tr:nth-child(even) { background: #f5f5f5; }
  strong { color: #0066cc; }
  code {
    background: #e8f4fc;
    color: #0066cc;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  ul, ol { font-size: 0.95em; line-height: 1.6; }
  li { margin: 8px 0; }
  img { border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  section.title {
    background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 { color: white; border: none; font-size: 2.4em; }
  section.title h2 { color: rgba(255,255,255,0.9); font-weight: 400; }
  section.title p { color: rgba(255,255,255,0.8); }
  section.title strong { color: #ffcc00; }
  footer { color: #999; font-size: 0.6em; }
---

<!-- _class: title -->

# MLOps — Credit Scoring
## Deploiement, Monitoring & Cycle de vie du modele

**Projet 8 — Data Scientist**

---

# Contexte & Objectif

**Modele** : LightGBM (Gradient Boosting) entraine dans le Projet 6 sur le dataset Home Credit
**Mission** : Deployer ce modele en production avec un cycle MLOps complet

| Element | Detail |
|---------|--------|
| Algorithme | LightGBM — rapide, gere les NaN nativement |
| Features | 795 features d'ingenierie |
| Seuil | 0.11 — optimise pour minimiser le cout metier |
| Output | Probabilite de defaut de paiement (0 a 1) |

**Problematique** : Comment maintenir un modele fiable dans le temps, quand les donnees de production evoluent ?

---

# Architecture MLOps — Vue d'ensemble

```
                    ┌─────────────────────────────────┐
                    │        Donnees versionnees       │
                    │  reference (200K) + test (50K)   │
                    │  + pool de drift (57K)           │
                    │  dataset_registry.json (SHA-256) │
                    └──────────────┬──────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  ┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
  │  FastAPI      │     │  Streamlit        │     │  Pipeline        │
  │  Serving      │────▶│  Dashboard        │     │  Reentrainement  │
  │  port 8000    │     │  port 7860        │     │  retrain.py      │
  └──────────────┘     └───────────────────┘     └────────┬─────────┘
         │                       │                        │
         │              predictions.jsonl          ┌──────▼───────┐
         │                       │                 │ Gate humaine  │
         │                       ▼                 │ approve.py    │
         │              ┌────────────────┐         └──────┬───────┘
         │              │ Evidently AI   │                │
         │              │ Drift Detection│───drift───────▶│
         │              └────────────────┘         Nouveau modele
         └──────────────────────────────────────────promu──┘
```

---

# Stack Technique

| Composant | Technologie | Role |
|-----------|-------------|------|
| API de scoring | **FastAPI** + Pydantic | Servir les predictions (single + batch) |
| Modele | **LightGBM** Booster | Prediction de defaut de paiement |
| Monitoring | **Streamlit** + Plotly | Dashboard temps reel (scores, latence) |
| Drift detection | **Evidently AI** 0.5.0 | Tests statistiques sur les distributions |
| Conteneurisation | **Docker** + Compose | 2 services : API + Dashboard |
| CI/CD | **GitHub Actions** | Tests (pytest) → Build → Health check |
| Pipeline | **Python** scripts | Reentrainement automatise |

---

# API de Scoring — FastAPI

## 3 endpoints

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/health` | GET | Verification de sante (liveness probe) |
| `/predict` | POST | Prediction unique — 1 client |
| `/predict/batch` | POST | Predictions par lot — N clients |

**Performance mesuree** :

| Metrique | Valeur |
|----------|--------|
| Latence moyenne | **1.13 ms** par prediction |
| P99 | 1.64 ms |
| Batch 1000 | **0.133 ms/record** (20x plus efficace) |

18 tests unitaires (pytest) couvrent l'API, la validation, et le modele.

---

# Donnees — Splits & Versioning

## 307 511 lignes du dataset Home Credit

| Dataset | Lignes | Role |
|---------|--------|------|
| `reference_data.parquet` | 200 000 | Entrainement + baseline pour drift |
| `test_data.parquet` | 50 000 | Validation (comparaison AUC) |
| `drift_pool.parquet` | 57 511 | Pool pour simuler des scenarios de drift |

**Dataset Registry** (`dataset_registry.json`) :
- Version, date de creation, source
- **SHA-256** de chaque fichier — garantit l'integrite
- Nombre de lignes/colonnes

En production, ce registre serait remplace par **DVC** (Data Version Control) avec stockage sur S3/GCS.

---

# Drift Detection — Evidently AI

## Comment detecter que les donnees de production ont change ?

**Niveau 1 — Par feature** (104 features numeriques) :
- Test statistique : **Wasserstein distance** (numeriques) ou **Jensen-Shannon** (categorielles)
- Si p-value < 0.05 → la feature est consideree driftee

**Niveau 2 — Dataset** :
- Si **>50% des features** driftent → `dataset_drift = True` (seuil par defaut Evidently)
- Ce flag declenche le pipeline de reentrainement

---

# 4 Scenarios de Drift Simules

## Sous-ensembles reels biaises du pool (57K lignes)

| Scenario | Filtre | Features driftees | Drift ? |
|----------|--------|-------------------|---------|
| Controle (aleatoire) | Random | 0/104 (0.0%) | **Non** |
| Demographique | Clients < 35 ans | 14/104 (13.5%) | **Non** |
| Economique | Revenus > 300K | 52/104 (50.0%) | **Oui** |
| Credit | Montant > 1M | 17/104 (16.3%) | **Non** |

- **Controle** : prouve que le systeme ne declenche pas de fausse alerte
- **Economique** : AMT_ANNUITY (1.10), AMT_GOODS_PRICE (0.95), AMT_CREDIT (0.92) — les features financieres driftent massivement
- Seul le scenario economique depasse le seuil de 50% → declenche le reentrainement

---

# Pipeline de Reentrainement

## `retrain.py` — 4 etapes automatisees

| Etape | Action |
|-------|--------|
| 1. Detection | Evidently compare production vs reference → `dataset_drift = True` |
| 2. Reentrainement | LightGBM entraine sur **reference + donnees de production** combinees |
| 3. Validation | Candidat vs Champion sur le jeu de test (comparaison AUC) |
| 4. Gate humaine | Ecrit `pending_approval.json` — **arrete et attend** |

Le reentrainement combine les donnees historiques (200K) avec les nouvelles donnees de production (4 358 clients hauts revenus) → le modele s'adapte a la nouvelle population.

---

# Human-in-the-loop — Approval Gate

## Pourquoi ne pas deployer automatiquement ?

**Credit scoring = secteur reglemente** :
- Un modele qui refuse des credits doit etre **explicable et auditable**
- Un reentrainement peut introduire un **biais** contre un groupe demographique
- L'AUC sur le test set ne garantit pas le comportement en production

**Resultats de notre pipeline** :

| Modele | AUC |
|--------|-----|
| Champion (avant) | 0.7478 |
| Candidat (apres) | **0.7492** |
| Difference | **+0.0015** |

L'expert revoit les metriques → execute `approve.py` → le candidat est promu champion.

---

# Model Registry — Champion / Candidat / Rollback

| Fichier | Role | Analogie DevOps |
|---------|------|-----------------|
| `model.lgb` | Champion (production) | Slot production |
| `model_candidate.lgb` | Candidat (en attente) | Slot staging |
| `model_previous.lgb` | Ancien champion | Rollback |

`approve.py` :
1. Archive le champion → `model_previous.lgb`
2. Promeut le candidat → `model.lgb`
3. Met a jour `dataset_registry.json`

Si le nouveau modele pose probleme → rollback vers `model_previous.lgb`.

---

# Monitoring — Streamlit Dashboard

## Suivi en temps reel des predictions

| Widget | Metrique |
|--------|----------|
| KPI row | Nombre de predictions, score moyen, latence moyenne |
| Distribution des scores | Histogramme des probabilites |
| Latence dans le temps | Evolution de la latence par prediction |
| Outcomes | Repartition Default vs Approved (seuil 0.11) |
| Probabilite dans le temps | Scores avec ligne de seuil |

Le dashboard lit `predictions.jsonl` — chaque prediction est loguee en JSON structuré avec timestamp, probabilite, latence, IP client, et nombre de features manquantes.

---

# La difference cle : MLOps ≠ DevOps

## En DevOps, le systeme se degrade quand **on change le code**.
## En MLOps, le modele se degrade quand **le monde change** — sans toucher au code.

C'est pour cela qu'il faut :
- **Monitorer les donnees**, pas seulement l'uptime
- **Automatiser le reentrainement**, pas seulement le deploiement
- **Garder un humain dans la boucle**, parce qu'un modele n'est pas deterministe comme du code

---

# Synthese

| Critere | Implementation |
|---------|---------------|
| Serving | FastAPI + Docker — latence 1.13 ms |
| Donnees | 307K lignes, 3 splits, registre SHA-256 |
| Drift detection | Evidently AI — 4 scenarios reels simules |
| Reentrainement | Automatise — reference + production combinees |
| Validation | Candidat vs Champion (AUC sur 50K lignes) |
| Gate humaine | `pending_approval.json` → revue expert → `approve.py` |
| Monitoring | Streamlit dashboard temps reel |
| CI/CD | GitHub Actions — 18 tests + Docker build |

---

<!-- _class: title -->

# Merci

## Questions ?
