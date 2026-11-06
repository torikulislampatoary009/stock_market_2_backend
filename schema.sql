-- Optional: the app creates this table automatically on start-up.
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
) CHARACTER SET utf8mb4;
