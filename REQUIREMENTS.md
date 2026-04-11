# Projet 8 — Confirmez vos competences en MLOps (Partie 2/2)

**Duree** : 45 heures
**Derniere mise a jour** : 18 fevrier 2026

---

## Contexte

Vous etes Data Scientist dans l'entreprise "Pret a Depenser". Apres avoir developpe et versionne un modele de scoring (Projet 7 — Initiez-vous au MLOps), vous devez maintenant deployer ce modele en production et mettre en place un suivi.

Ce projet fait suite au projet precedent "Initiez-vous au MLOps (Partie 1/2)" et reutilise le modele de scoring que vous avez developpe, versionne et evalue avec MLflow.

---

## Message Slack de Chloe Dubois (Lead Data Scientist)

> "Salut ! Excellents resultats sur la derniere version du modele de scoring ! Le departement 'Credit Express' est tres impatient de l'utiliser pour traiter les nouvelles demandes en quasi temps reel. Il nous faut absolument une API fonctionnelle et deployable (Docker Ready!) d'ici la fin de la semaine prochaine. Peux-tu prioriser ca ? On a aussi besoin d'un dashboard ou rapport de suivi pour verifier que tout se passe bien une fois en prod (distribution des scores, temps de reponse, ce genre de choses). Tiens-moi au courant de ton plan d'action ! Merci !"

---

## Objectifs

- Concevoir un systeme de suivi du cycle de vie du modele d'apprentissage
- Prendre ses premiers pas en ML Engineering
- Deployer le modele via une API
- Conteneuriser avec Docker
- Mettre en place un monitoring proactif
- Automatiser le pipeline CI/CD

---

## Livrables a produire

1. **Historique des versions** — liste des commits sur GitHub
2. **Scripts API** — API fonctionnelle (Gradio ou FastAPI) qui recoit les donnees d'un client et retourne un score de prediction
3. **Tests unitaires automatises**
4. **Dockerfile** — conteneurisation du code
5. **Analyse du Data Drift** — notebook montrant distribution des scores predits, latence API, temps d'inference, etc. (via Streamlit, Dash, ou notebook)
6. **Screenshots de la solution de stockage des donnees de production**
7. **Pipeline CI/CD** — fichier YAML (GitHub Actions) automatisant tests, build Docker, deploiement lors d'un push sur main
8. **Documentation README** — comment lancer l'API et interpreter le monitoring

---

## Etapes de la mission

### Etape 1 — Controle de version et depot

- Initialiser un depot Git structure (code source, tests, notebooks, Dockerfile, requirements, etc.)
- Commits explicites, historique clair
- Depot public sur GitHub
- `.gitignore` pour exclure donnees sensibles

**Resultats attendus** :
- Lien vers depot Git public
- Historique de commits clair et pertinent

### Etape 2 — API + Docker + CI/CD

- Developper une API (Gradio ou FastAPI) : recoit donnees client, retourne prediction
- Conteneuriser avec Docker
- Pipeline CI/CD (GitHub Actions) :
  - Executer les tests (unitaires, integration)
  - Construire l'image Docker si tests OK
  - Deployer l'image (simule ou reel)
- Charger le modele UNE SEULE FOIS au demarrage de l'API (pas a chaque requete)
- Tests : donnees manquantes, valeurs hors plage, types incorrects
- Securiser API et pipeline (secrets, validation d'entree)
- Deploiement suggere : Hugging Face Spaces

**Resultats attendus** :
- Code source API fonctionnel
- Dockerfile
- Pipeline CI/CD visible sur la plateforme
- Tests automatises integres au pipeline

### Etape 3 — Stockage et analyse des donnees de production

- Stocker : logs d'appels, inputs, outputs, temps d'execution (minimum)
- Analyse automatique : detection data drift, anomalies, taux d'erreur, latence
- PoC local acceptable si pas de cloud
- Logging structure (JSON)
- Utiliser Evidently AI ou NannyML pour detection de drift
- Dashboard de visualisation (Streamlit, Dash, Grafana)

**Resultats attendus** :
- Solution de stockage decrite et/ou implementee
- Script/notebook d'analyse automatique (drift, anomalies)
- Presentation de l'etude sur la derive des donnees

### Etape 4 — Optimisation des performances

- Analyser performances reelles ou simulees (temps d'inference, latence, CPU/GPU)
- Profiling pour identifier goulots d'etranglement
- Tester strategies d'optimisation (quantification, optimisation de code, hardware)
- Integrer version optimisee dans le depot
- Documenter les optimisations et resultats

**Resultats attendus** :
- Rapport detaillant tests d'optimisation, resultats, goulots d'etranglement
- Version optimisee deployee via CI/CD
- Justification de la configuration finale (librairies, software, hardware)
- Amelioration du temps d'inference/reponse demontree

---

## Soutenance (30 minutes)

L'evaluateur joue le role de **Chloe, la Lead Data Scientist** chez "Pret a Depenser".

### 1. Presentation des livrables (15 min)

Presenter :
- Le contexte de la mission
- Les resultats du monitoring : analyse de la derive des donnees (graphiques, metriques), collecte/analyse des logs
- Les optimisations de performance : tests, goulots d'etranglement, amelioration obtenue
- La structure du depot GitHub : organisation du code, Dockerfile, CI/CD, README

Demonstrations :
- **Fonctionnement de l'API** : envoyer une requete et montrer la reponse (score predit)
- **Fonctionnement du pipeline CI/CD** : montrer qu'un commit declenche automatiquement tests → build Docker → deploiement

### 2. Discussion (10 min)

Chloe challengera sur :
- **Robustesse et fiabilite** : gestion des erreurs a tous les niveaux
- **Monitoring et maintenance** : gestion de la derive, maintenance long terme
- **Optimisation et scalabilite** : choix software/hardware

### 3. Debrief (5 min)

Points forts, axes d'amelioration, validation des competences.

---

## Competences validees

1. Prendre ses premiers pas en ML Engineering
2. Concevoir un systeme de suivi du cycle de vie du modele d'apprentissage

---

## Format de depot des livrables

Dossier ZIP nomme : `Confirmez_vos_competences_en_MLOps_Kieu_Tuan`

Fichiers nommes :
- `Kieu_Tuan_1_historique_032026`
- `Kieu_Tuan_2_scripts_032026`
- `Kieu_Tuan_3_dockerfile_032026`
- `Kieu_Tuan_4_tests_032026`
- `Kieu_Tuan_5_pipeline_cicd_032026`
- `Kieu_Tuan_6_analyse_drift_032026`
- `Kieu_Tuan_7_screenshots_stockage_032026`

---

## Outils suggeres

- **API** : Gradio, FastAPI
- **Conteneurisation** : Docker
- **CI/CD** : GitHub Actions
- **Tests** : Pytest
- **Deploiement** : Hugging Face Spaces, Heroku, Google Cloud Run
- **Monitoring/Drift** : Evidently AI, NannyML
- **Dashboard** : Streamlit, Dash, Grafana
- **Profiling** : cProfile
- **Optimisation** : ONNX Runtime
- **Logging** : logging Python, Fluentd, Logstash
- **Stockage** : Elasticsearch, PostgreSQL
