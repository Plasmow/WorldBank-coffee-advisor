# Coffee Advisor

Hackathon Hack-Nation × Banque mondiale, Challenge 04 « Small AI for Development », track Agriculture.
Rendu : dimanche 4 octobre 2026, 12 h (Paris). Live Demo en ligne obligatoire + vidéo de 2 à 5 min.

## Produit

Conseiller agricole **par SMS** pour Noor, petite productrice de café au mont Elgon (Ouganda), qui n'a qu'un **téléphone basique** et parle luganda.

- SMS libre → traduction (NLLB-200) → classifieur (`leaf_rust`, `phoma`, `healthy`, `other`) + second avis d'un petit LLM → réponse **fixe** tirée de `data/templates.json`.
- Message flou : une seule question de précision. Doute persistant ou `other` : « pas sûr » à Noor + **SMS automatique à l'agent** (garde-fou humain, critère éliminatoire).
- Suivi automatique 3 jours après un diagnostic, entre 18 h et 20 h (UTC+3). Réponse « 3 = pire » → alerte agent.
- Mots-clés : `PRIX` (prix UCDA, sans IA), `AIDE`, `STOP` / `START`. Bienvenue + consentement au premier message.

Hors périmètre : app mobile, vision, tableau de bord agent, grand LLM (Claude, GPT…) dans les échanges avec Noor.

## Architecture

- Backend FastAPI + SQLite sur **Replit** (1 worker, 1 instance). Page de démo sur **Lovable** ; `web/index.html` = secours (Vercel).
- SMS via **Africa's Talking (sandbox)** : webhook `POST /sms` (formulaire `from`, `text`…).
- Démo : chaque visiteur a un numéro `+256799XXXXXX` (jamais envoyé à Africa's Talking) et sa propre horloge simulée (`app/clock.py`).
  `run_due()` tourne à chaque requête de démo et dans une boucle de fond (le serveur peut se mettre en veille).

```
app/   main.py (routes, CORS, /health) · router.py (handle_incoming) · db.py · clock.py · scheduler.py · at_client.py · demo_routes.py
ai/    analyze.py (provisoire, mots-clés, à remplacer par la vraie chaîne)
data/  templates.json (en, lg) · prices.json
web/   index.html          scripts/ model/ docs/   données et classifieur
```

## Contrats (prévenir l'équipe avant de les changer)

- `analyze(text, clarify_answer=None) -> {lang, text_en, label, proba, llm_label, decision: answer|clarify|escalate, template_id}`
- `model/labels.json` : `{"labels": [...], "threshold": 0.8}`
- `GET /api/demo/state?phone=` → `{now, messages: [{id, direction: in|out, text, ts}], events: [{id, kind: agent_alert|followup, text, ts}]}`
- `POST /api/demo/send {phone, text}` · `/clock {phone, day 0-6, slot morning|day|evening|night}` · `/reset {phone}`
- Env : `AT_USERNAME`, `AT_API_KEY`, `AT_SHORTCODE`, `AGENT_PHONE`, `DEMO_MODE`, `FRONTEND_ORIGIN`, `USE_LLM`, `DB_PATH`

## Équipe

- **P1** : données (dataset Mendeley, SMS générés par Claude), classifieur, évaluation, datasheet → `scripts/`, `model/`, `docs/`
- **P2** : backend, Africa's Talking, scheduler, routes de démo, déploiement Replit, USSD → `app/`
- **P3** : chaîne IA (`ai/`), traductions luganda, prix, page Lovable, vidéo → `ai/`, `data/`, `web/`

Jalons : 3 h backend en ligne et SMS du simulateur qui reçoit une réponse · 7 h flux complet avec le vrai modèle · 9 h gel total.

## Règles

- `main` doit toujours tourner (Replit la déploie) : tests `pytest` avant chaque push, petits commits.
- Tout message vers Noor vient de `templates.json`. Tout horodatage passe par `clock.now(phone)`.
- Code en anglais, commentaires en anglais. Pas de dépendance lourde dans `app/`. Jamais de poids de modèle ni de `.env` dans git.
