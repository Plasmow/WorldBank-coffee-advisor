# État du projet — passation

Rédigé le 4 octobre 2026 au matin, rendu à 12 h. Lire `CLAUDE.md` d'abord
pour le produit et les contrats d'équipe ; ce fichier dit où on en est.

## En ligne, maintenant

**https://worldbank-coffee-advisor.onrender.com** — Render, plan gratuit,
512 Mo, déployé depuis `main`.

- `/demo` : la démo complète, et **c'est elle qu'il faut montrer au jury**
- `/health` : `{"ok":true,"rss_mb":60,"ai":{"enabled":false,"loaded":false}}`
- `/sms`, `/ussd` : prêts, callbacks Africa's Talking à brancher
- `/api/demo/*` : pilotage de la démo, restreint aux numéros `+256799…`

L'instance dort après 15 min (réveil 30-60 s). Pendant la démo, garder :
`while true; do curl -s -o /dev/null https://worldbank-coffee-advisor.onrender.com/health; sleep 600; done`

104 tests verts (`uv run pytest -q`). Branche `main`, tout est poussé.

## Ce qui fonctionne, vérifié de bout en bout

Parcours complet joué contre la production : bienvenue → diagnostic rouille →
suivi J+3 via l'horloge simulée → réponse « 3 » → alerte agent → escalade
hors-périmètre → prix UCDA réels. Plus : STOP/START, PRICE, HELP, AGENT,
isolation de deux visiteurs, robustesse du webhook (champs manquants, texte
vide, 2000 caractères, message rejoué).

Menu USSD : 1 prix, 2 signaler un symptôme, 3 demander un agent.

## Les trois blocages

### 1. Le vrai classifieur n'est pas branché

`model/classifier.joblib` (7,2 Ko) et `model/labels.json` existent — P1 a
livré. Mais `ai/classifier.py` est **toujours le stub à mots-clés**.

Et le brancher a un coût que personne n'a mesuré : l'inférence passe par des
embeddings **e5-small-v2**, donc `sentence-transformers` + **torch** (~800 Mo
installés) + le modèle e5 (~130 Mo). **Ça ne rentre pas dans les 512 Mo de
Render.** Il faudrait une VM, ou renoncer.

### 2. Désaccord de libellé : `unknown` vs `other`

`model/labels.json` annonce `["healthy", "leaf_rust", "phoma", "unknown"]`.
`ai/analyze.py` teste `if label == "other"` ; `app/analysis.py` ne connaît
que healthy/leaf_rust/phoma. Le jour où le vrai classifieur est branché, il
renverra `unknown`, la branche `other` ne se déclenchera jamais et les
messages hors périmètre n'escaladeront plus. **À trancher avant de brancher.**

### 3. Le luganda ne passe pas

NLLB-200-600M traduit mal cette langue. Mesuré :

| luganda | NLLB | attendu |
|---|---|---|
| Ebikoola by'emmwanyi zange birina obutuli obwa kyenvu | *My coffee beans are yellowish* | feuilles, taches jaunes |
| Emmwanyi zange nnungi | *My oil is good* | mon café va bien |

Aucun classifieur ne rattrape « My oil is good ». Soit on assume une démo en
anglais, soit on passe au modèle 1.3B (RAM × 2).

## Africa's Talking

Compte sandbox **à découvert : EUR -0,1785**. Les envois échouent désormais.

Ce qui reste gratuit et prouve quand même le vrai canal télécom :
**USSD entrée 1 (prix)** — réponse à l'écran, aucun SMS émis. Rejouable à
volonté. Les entrées 2 et 3 coûtent un SMS, le canal SMS est mort.

## État de l'IA

`USE_REAL_AI=0` en production, **et c'est le bon réglage**. Sur l'anglais, la
chaîne de P3 et le repli à mots-clés donnent les mêmes diagnostics
aujourd'hui, pour 60 Mo au lieu de 1,5 Go.

`USE_REAL_AI=1` fonctionne pourtant de bout en bout en local : anglais sans
aucun modèle, luganda via le traducteur CTranslate2 int8 (621 Mo sur disque,
1,4 Go en RAM, 1,4 s au chargement). Ollama n'est pas installé : `llm_predict`
renvoie `label=None` et `analyze()` répond sur le classifieur seul
(`reason="llm_unavailable"`), ce qui est le plan B assumé de P3.

## Pièges déjà payés — ne pas les refaire

- **`at_client`** : AT refuse un message avec **201 Created** et le motif dans
  le corps. Vérifier `Recipients`, pas le code HTTP.
- **Expéditeur** : en sandbox il faut **omettre** le champ `from`. Vide ou
  arbitraire → `InvalidSenderId`, livré à personne, facturé quand même.
- **`.env`** : lu au démarrage avec `override=False` (les secrets de
  l'hébergeur gagnent), et **ignoré sous pytest** via `SKIP_DOTENV` — une
  suite qui lit le `.env` du poste passe ici et échoue ailleurs.
- **Mots entiers** : `ant` matchait dans `plants`, chaque message partait à
  l'agent. Tout est en `\b…s?\b` maintenant.
- **Bienvenue** : conditionnée à un drapeau `welcomed` explicite, pas à
  « la ligne n'existait pas » — `/api/demo/clock` crée la ligne.
- **`render.yaml`** n'est lu que si le service est créé en **Blueprint**. Un
  Web Service manuel utilise les champs du tableau de bord.
- **Pas de `uv` ni de `--reload`** sur un hébergeur :
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

## Architecture en une phrase

Quatre entrées (`/sms`, `/api/demo/send`, `/ussd`, le scheduler) convergent
vers `router.handle_incoming()`, et **toute** sortie vers Noor passe par
`db.send()` — c'est là qu'est vérifié le STOP, une seule fois pour tous les
appelants. `app/` n'importe jamais `ai/` au niveau module : la couture est
`app/analysis.py`, import paresseux sous `USE_REAL_AI=1`, et toute panne de
la chaîne se traduit par une escalade vers un humain, jamais par un silence.

## Commandes

```bash
uv run pytest -q                        # 104 tests
uv run uvicorn app.main:app --reload    # local, puis /demo
curl https://worldbank-coffee-advisor.onrender.com/health
```

## À faire, par ordre

1. **Brancher les callbacks AT** sur `/sms` et `/ussd` — gratuit, et l'USSD
   entrée 1 donne une preuve du vrai réseau sans dépenser un centime.
2. **Décider `unknown` vs `other`** avec P1 et P3.
3. **Décider du classifieur réel** : il impose torch et une VM. Sans décision,
   rester sur le stub, qui marche.
4. Accorder la page de P3 (`?api=` + `FRONTEND_ORIGIN`) — **ou s'en passer** :
   `/demo` est servi par le backend, même origine, zéro configuration.
5. Vidéo 2-5 min : montrer `/demo` (illimité, seul à savoir faire le J+3),
   puis l'USSD entrée 1 pour le vrai canal.
