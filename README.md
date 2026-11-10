# Statut de ce dépôt (audit du 10 novembre 2026)

**Prototype d'avril 2026, non compatible avec AIWorkPay V2. Ne pas le brancher sur la production en l'état.**

| Ce que fait ce dépôt | Réalité de l'API AIWorkPay V2 (`cortexAI-2025/Aiworkpay-v2`) |
|---|---|
| Appelle `https://api.aiworkpay.com/v1` | L'API est servie par `https://app.aiworkpay.fr` ; le domaine `.com` n'est pas celui du service |
| Routes `POST /tasks`, `GET /tasks/{id}`, `POST /tasks/{id}/approve` | Routes `POST/GET /api/missions`, `GET /api/missions/{id}`, `POST /api/missions/{id}/approve` |
| Statuts `completed`, `failed`, `cancelled` | `PAYMENT_PENDING, CREATED, PUBLISHED, ASSIGNED, IN_PROGRESS, DELIVERED, COMPLETED, CANCELED` |
| Tâche payée et approuvée automatiquement par l'agent, attente de quelques minutes | Un humain paie chaque mission sur Stripe Checkout ; un Payworker l'accepte puis la livre ; l'agent approuve ensuite |
| Client simulé par défaut sans clé | Clé `awp_…` avec permissions et plafonds de dépense |

Sans clé, le code tourne avec son client simulé : cela ne prouve rien sur l'intégration réelle.

## Pour intégrer un agent à AIWorkPay V2
Utiliser les points d'entrée maintenus et testés : le SDK Python (`cortexAI-2025/AIWorkPay-SDK-v1`), le serveur MCP (`cortexAI-2025/aiworkpay-mcp-server`) ou l'API REST décrite par `/openapi.json`. Documentation : https://aiworkpay.fr/agents/

## Décision à prendre par le propriétaire
Archiver ce dépôt, ou réécrire son client au-dessus du SDK/MCP (changement de modèle : la délégation devient asynchrone, avec paiement humain). Aucun code n'a été modifié par cet audit.
