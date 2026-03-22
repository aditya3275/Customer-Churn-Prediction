-- ─────────────────────────────────────────────────────────────────────────────
-- analytics.sql — ChurnGuard AI  (PostgreSQL)
-- Sample queries for business reporting and model monitoring.
-- Replace schema names / views as needed for your reporting tool.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. Overall churn risk distribution ───────────────────────────────────────
-- How many customers fall into each risk bucket?
SELECT
    risk_category,
    COUNT(*)                        AS customer_count,
    ROUND(AVG(churn_probability)::NUMERIC, 4) AS avg_churn_prob,
    ROUND(SUM(lifetime_value)::NUMERIC, 2)    AS total_clv_at_stake
FROM predictions
GROUP BY risk_category
ORDER BY avg_churn_prob DESC;


-- ── 2. High-risk customers with full profile ─────────────────────────────────
-- For CRM / retention team: who needs immediate attention?
SELECT
    c.customer_id,
    c.gender,
    c.tenure,
    c.contract_type,
    c.monthly_charges,
    p.churn_probability,
    p.risk_category,
    p.lifetime_value,
    p.predicted_at
FROM predictions p
JOIN customers c USING (customer_id)
WHERE p.risk_category = 'High Risk'
ORDER BY p.churn_probability DESC, p.lifetime_value DESC
LIMIT 100;


-- ── 3. Revenue at risk by contract type ──────────────────────────────────────
SELECT
    c.contract_type,
    COUNT(DISTINCT c.customer_id)           AS customers,
    ROUND(AVG(p.churn_probability)::NUMERIC, 4)  AS avg_churn_prob,
    ROUND(SUM(c.monthly_charges)::NUMERIC, 2)    AS monthly_revenue,
    ROUND(
        SUM(c.monthly_charges * p.churn_probability)::NUMERIC, 2
    )                                            AS revenue_at_risk
FROM predictions p
JOIN customers c USING (customer_id)
GROUP BY c.contract_type
ORDER BY revenue_at_risk DESC;


-- ── 4. Churn trend over time (daily) ─────────────────────────────────────────
-- Monitor model output drift day-by-day.
SELECT
    DATE_TRUNC('day', predicted_at)         AS prediction_date,
    COUNT(*)                                AS total_predictions,
    ROUND(AVG(churn_probability)::NUMERIC, 4) AS avg_churn_prob,
    SUM(CASE WHEN churn_prediction = 'Yes' THEN 1 ELSE 0 END) AS churners
FROM predictions
GROUP BY DATE_TRUNC('day', predicted_at)
ORDER BY prediction_date DESC;


-- ── 5. Retention action effectiveness ────────────────────────────────────────
-- Which recommended actions were acted upon and led to retention?
SELECT
    recommended_action,
    COUNT(*)                                              AS times_recommended,
    SUM(CASE WHEN action_taken IS NOT NULL THEN 1 ELSE 0 END) AS times_acted,
    SUM(CASE WHEN outcome = 'retained' THEN 1 ELSE 0 END)     AS retained_count,
    ROUND(
        100.0 * SUM(CASE WHEN outcome = 'retained' THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN action_taken IS NOT NULL THEN 1 ELSE 0 END), 0),
        2
    )                                                     AS retention_rate_pct
FROM retention_actions
GROUP BY recommended_action
ORDER BY retention_rate_pct DESC NULLS LAST;


-- ── 6. API performance report ─────────────────────────────────────────────────
-- Identify slow or failing endpoints.
SELECT
    endpoint,
    method,
    COUNT(*)                                        AS request_count,
    ROUND(AVG(response_time_ms)::NUMERIC, 1)        AS avg_response_ms,
    MAX(response_time_ms)                           AS max_response_ms,
    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_count,
    ROUND(
        100.0 * SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                               AS error_rate_pct
FROM audit_logs
GROUP BY endpoint, method
ORDER BY avg_response_ms DESC;


-- ── 7. Top customers by CLV at risk ──────────────────────────────────────────
-- Prioritise retention spend on the highest-value accounts.
SELECT
    c.customer_id,
    c.contract_type,
    c.monthly_charges,
    p.churn_probability,
    p.lifetime_value,
    p.risk_category,
    p.predicted_at
FROM predictions p
JOIN customers c USING (customer_id)
WHERE p.predicted_at = (
    -- latest prediction only per customer
    SELECT MAX(p2.predicted_at)
    FROM predictions p2
    WHERE p2.customer_id = p.customer_id
)
ORDER BY p.lifetime_value DESC
LIMIT 50;


-- ── 8. Segment churn by internet service & payment method ────────────────────
SELECT
    c.internet_service,
    c.payment_method,
    COUNT(*)                                         AS customers,
    ROUND(AVG(p.churn_probability)::NUMERIC, 4)      AS avg_churn_prob
FROM predictions p
JOIN customers c USING (customer_id)
GROUP BY c.internet_service, c.payment_method
ORDER BY avg_churn_prob DESC;


-- ── 9. Dashboard snapshot history ────────────────────────────────────────────
-- See how aggregate risk metrics evolved over time.
SELECT
    snapshot_id,
    total_high_risk_customers,
    avg_churn_probability,
    total_revenue_at_risk,
    snapshot_timestamp
FROM dashboard_snapshots
ORDER BY snapshot_timestamp DESC
LIMIT 30;
