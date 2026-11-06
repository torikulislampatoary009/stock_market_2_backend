"""
MySQL connection helper (PyMySQL) - built for Render (app) + Aiven (MySQL).

Set the credentials as ENVIRONMENT VARIABLES in the Render dashboard.
Use ONE of these two styles:

  A) Separate variables (recommended):
       DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
  B) Single Aiven "Service URI":
       DATABASE_URL = mysql://avnadmin:PASSWORD@HOST:PORT/defaultdb?ssl-mode=REQUIRED

SSL (Aiven requires it):
  DB_SSL        true (default) | false
  DB_CA_CERT    paste the full text of Aiven's ca.pem here (Render env var, multi-line OK)
  DB_CA_PATH    ...or a path to a ca.pem file in the container
  If neither CA is given, the connection is still encrypted but the server
  certificate is not verified (fine for a student project; add the CA for strict mode).
"""
import os
import tempfile
from urllib.parse import urlparse, unquote

import pymysql
from pymysql.cursors import DictCursor

_ca_file_cache = None


def _config():
    url = os.getenv("DATABASE_URL")
    if url:
        u = urlparse(url)
        return {
            "host": u.hostname,
            "port": u.port or 3306,
            "user": unquote(u.username or ""),
            "password": unquote(u.password or ""),
            "database": (u.path or "/").lstrip("/") or "defaultdb",
        }
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "defaultdb"),
    }


def _ca_path():
    """Return a path to the CA file (writes DB_CA_CERT to a temp file if provided)."""
    global _ca_file_cache
    if _ca_file_cache:
        return _ca_file_cache
    pem = os.getenv("DB_CA_CERT")
    if pem:
        pem = pem.replace("\\n", "\n").strip() + "\n"
        f = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
        f.write(pem)
        f.close()
        _ca_file_cache = f.name
        return _ca_file_cache
    path = os.getenv("DB_CA_PATH")
    if path and os.path.exists(path):
        _ca_file_cache = path
    return _ca_file_cache


def get_connection():
    cfg = _config()
    kwargs = dict(
        host=cfg["host"], port=cfg["port"], user=cfg["user"], password=cfg["password"],
        database=cfg["database"], charset="utf8mb4", cursorclass=DictCursor,
        connect_timeout=10, read_timeout=15, write_timeout=15, autocommit=True,
    )
    if os.getenv("DB_SSL", "true").lower() not in ("false", "0", "no"):
        ca = _ca_path()
        if ca:
            kwargs.update(ssl_ca=ca, ssl_verify_cert=True, ssl_verify_identity=False)
        else:
            kwargs.update(ssl={"fake_flag_to_enable_tls": True})  # encrypted, certificate not verified
    return pymysql.connect(**kwargs)


def init_db():
    """Create the table if it does not exist. Safe to call on every start."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(16) NOT NULL,
                    as_of_date DATE NOT NULL,
                    last_close DOUBLE NOT NULL,
                    predicted_close DOUBLE NOT NULL,
                    predicted_change_pct DOUBLE NOT NULL,
                    predicted_volatility DOUBLE NOT NULL,
                    annualized_volatility DOUBLE NOT NULL,
                    volatility_level VARCHAR(16) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ticker (ticker),
                    INDEX idx_created (created_at)
                ) CHARACTER SET utf8mb4
            """)
    finally:
        conn.close()


def save_prediction(p):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions (ticker, as_of_date, last_close, predicted_close,
                   predicted_change_pct, predicted_volatility, annualized_volatility, volatility_level)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (p["ticker"], p["as_of_date"], p["last_close"], p["predicted_close"],
                 p["predicted_change_pct"], p["predicted_volatility"],
                 p["annualized_volatility"], p["volatility_level"]),
            )
    finally:
        conn.close()


def get_history(limit=20, ticker=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if ticker:
                cur.execute("SELECT * FROM predictions WHERE ticker=%s ORDER BY id DESC LIMIT %s", (ticker, limit))
            else:
                cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        for r in rows:
            r["as_of_date"] = str(r["as_of_date"])
            r["created_at"] = str(r["created_at"])
        return rows
    finally:
        conn.close()


def ping():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    finally:
        conn.close()
