# Stock Market Volatility Predictor - Backend (Flask + MySQL)

Endpoints: `GET /api/health`, `POST /api/predict {"ticker":"AAPL"}`, `GET /api/history`, `GET /api/model-info`

## Run locally
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then export the variables (or set them in your shell)
python app.py
```

## Retrain on real Yahoo Finance data (recommended)
The bundled `model/volatility_model.joblib` was trained on SYNTHETIC data (no internet where it was built).
Run this on your computer and commit the new file:
```
python train.py                                  # 16 default tickers, 10 years
python train.py --tickers AAPL MSFT TSLA --years 8
```
Keep the same scikit-learn version as requirements.txt, otherwise the pickle may not load.

## Deploy: Aiven (MySQL) -> Render (API)
1. Aiven: create a free MySQL service. Copy Host, Port, User, Password, Database name, and download `ca.pem`.
2. Render: New > Web Service > your GitHub repo. Root Directory = `backend` (if monorepo).
   Build: `pip install -r requirements.txt`
   Start: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
3. Render > Environment, add:
   `PYTHON_VERSION=3.12.3`, `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`,
   `DB_CA_CERT` (paste the full ca.pem text), `CORS_ORIGINS` (your frontend URL, or `*`).
   Alternative: only `DATABASE_URL` = Aiven Service URI.
4. In Aiven, make sure the service allows connections from the internet (Allowed IP addresses includes 0.0.0.0/0).
5. Open `https://YOUR-API.onrender.com/api/health` - `database_ok` should be true. The table is created automatically.

Free tiers: Render sleeps after ~15 min idle (first request is slow). Never commit passwords - only `.env.example`.
