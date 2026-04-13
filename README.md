---
title: Credit Scoring MLOps
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Project 8 — MLOps Partie 2 : Déploiement et Monitoring

API de scoring crédit basée sur un modèle LightGBM, avec monitoring en temps réel, détection de drift, pipeline de réentraînement automatisé et validation humaine.

## Architecture MLOps

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CYCLE MLOps COMPLET                         │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Données │───▶│  Modèle  │───▶│   API    │───▶│  Monitoring  │  │
│  │ versionnées│  │ champion │    │ FastAPI  │    │  Streamlit   │  │
│  │ (parquet) │  │  (.lgb)  │    │ (8000)   │    │   (7860)     │  │
│  └──────────┘    └──────────┘    └─────┬────┘    └──────┬───────┘  │
│       ▲                                │                 │          │
│       │                                ▼                 ▼          │
│       │                         predictions.jsonl   Dashboard      │
│       │                                                  │          │
│       │              ┌──────────────────────────────────┐│          │
│       │              │      Détection de Drift          ││          │
│       │              │      (Evidently AI)              ││          │
│       │              └──────────────┬───────────────────┘│          │
│       │                             │ drift détecté       │          │
│       │                             ▼                     │          │
│       │              ┌──────────────────────────────────┐│          │
│       │              │   Pipeline de Réentraînement     ││          │
│       │              │   (pipeline/retrain.py)          ││          │
│       │              └──────────────┬───────────────────┘│          │
│       │                             │                     │          │
│       │                             ▼                     │          │
│       │              ┌──────────────────────────────────┐│          │
│       │              │   Validation Humaine (Gate)      ││          │
│       │              │   (pipeline/approve.py)          ││          │
│       │              └──────────────┬───────────────────┘│          │
│       │                             │ approuvé            │          │
│       └─────────────────────────────┘                     │          │
│                                                           │          │
│  ┌─────────────────┐    ┌──────────────────────────────┐ │          │
│  │ Dataset Registry │    │    CI/CD GitHub Actions      │ │          │
│  │ (versionnement) │    │  test → build → deploy       │ │          │
│  └─────────────────┘    └──────────────────────────────┘ │          │
└─────────────────────────────────────────────────────────────────────┘
```

## Stack technique

| Composant | Technologie | Rôle |
|-----------|------------|------|
| Modèle | LightGBM | Prédiction de défaut de crédit (795 features, seuil 0.11) |
| API | FastAPI | Endpoints REST de scoring |
| Validation | Pydantic | Schémas requête/réponse |
| Dashboard | Streamlit + Plotly | Monitoring temps réel |
| Drift | Evidently AI | Détection de dérive des données |
| Réentraînement | Pipeline Python | Automatisé avec gate humaine |
| Versionnement | Dataset Registry | SHA-256, versions, métadonnées |
| Conteneurisation | Docker + Compose | 2 services (API + dashboard) |
| CI/CD | GitHub Actions | Test → build → deploy |
| Tests | Pytest | 18 tests (API, validation, modèle) |

## Démarrage rapide

### Docker Compose

```bash
docker compose up --build -d
```

- API : http://localhost:8000 (docs : http://localhost:8000/docs)
- Dashboard : http://localhost:7860

### Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000            # Terminal 1
streamlit run dashboard/app.py --server.port 7860     # Terminal 2
```

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérification de santé |
| POST | `/predict` | Prédiction client unique |
| POST | `/predict/batch` | Prédiction batch (format MLflow) |
| GET | `/docs` | Documentation Swagger |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"AMT_INCOME_TOTAL": 202500, "AMT_CREDIT": 406597.5, "EXT_SOURCE_2": 0.263}}'
```

## Pipeline de Réentraînement

### Cycle complet

```bash
# 1. Détection de drift + réentraînement + validation
python pipeline/retrain.py

# 2. Revue humaine des résultats (AUC, comparaison champion/candidat)

# 3. Approbation et déploiement
python pipeline/approve.py
```

### Fonctionnement

1. **Détection de drift** : compare les données de production vs la référence (Evidently AI, tests KS/Chi²)
2. **Réentraînement** : si drift détecté, entraîne un modèle candidat LightGBM
3. **Validation** : compare candidat vs champion sur le jeu de test (AUC, distributions)
4. **Gate humaine** : écrit `pending_approval.json` — un humain doit valider avant déploiement
5. **Déploiement** : archive l'ancien champion, promeut le candidat, met à jour le registre

## Détection de Drift

### Données

| Dataset | Lignes | Description |
|---------|--------|-------------|
| `reference_data.parquet` | 200 000 | Échantillon de référence (entraînement) |
| `test_data.parquet` | 50 000 | Jeu de test (validation) |
| `drift_pool.parquet` | 57 511 | Pool pour simuler des scénarios de drift |

### Scénarios simulés

| Scénario | Filtre | But |
|----------|--------|-----|
| Contrôle | Aléatoire | Vérifier l'absence de drift sur données i.i.d. |
| Démographique | Clients < 35 ans | Changement de population |
| Économique | Revenus > 300K | Changement de profil financier |
| Crédit | Montant > 1M | Afflux de gros prêts |

## Versionnement

Le fichier `data/dataset_registry.json` trace :
- Version, date, nombre de lignes/colonnes
- Hash SHA-256 de chaque fichier de données
- Version du modèle et métriques de validation

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v   # 18 tests
```

## Structure du projet

```
project-8/
├── app/                          # Code API
│   ├── main.py                   # Endpoints FastAPI
│   ├── model.py                  # Chargement modèle et prédiction
│   ├── schemas.py                # Schémas Pydantic
│   ├── features.py               # 795 noms de features
│   ├── config.py                 # Configuration
│   └── logging_config.py         # Logging structuré JSON
├── dashboard/app.py              # Dashboard Streamlit + Plotly
├── pipeline/                     # Pipeline de réentraînement
│   ├── retrain.py                # Drift → retrain → validate → gate
│   └── approve.py                # Approbation humaine → deploy
├── model/
│   ├── model.lgb                 # Modèle champion
│   ├── model_candidate.lgb       # Modèle candidat (après retrain)
│   └── model_previous.lgb        # Modèle archivé (après approve)
├── data/
│   ├── reference_data.parquet    # 30K lignes — référence drift
│   ├── test_data.parquet         # 10K lignes — validation
│   ├── drift_pool.parquet        # 267K lignes — pool de simulation
│   └── dataset_registry.json     # Registre versionné
├── tests/                        # 18 tests pytest
├── notebooks/
│   ├── data_drift_analysis.ipynb # 4 scénarios de drift (Evidently)
│   └── performance_profiling.ipynb
├── .github/workflows/ci-cd.yml  # CI/CD
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── requirements.txt
└── requirements-dev.txt
```

## CI/CD

Le pipeline GitHub Actions :
1. **Test** : `pytest` sur chaque push/PR
2. **Build** : image Docker + health check
