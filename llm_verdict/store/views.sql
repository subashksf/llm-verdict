-- Convenience views for llm-verdict analytics

CREATE OR REPLACE VIEW model_summary AS
SELECT
    r.run_id,
    r.model_id,
    r.model_version,
    r.suite_hash,
    r.created_at,
    COUNT(DISTINCT s.task_id) AS tasks_graded,
    AVG(CASE WHEN s.passed THEN 1.0 ELSE 0.0 END) AS pass_rate,
    SUM(t.cost_usd) AS total_cost_usd,
    MEDIAN(t.latency_ms_total) AS p50_latency_ms
FROM runs r
LEFT JOIN trials t ON r.run_id = t.run_id
LEFT JOIN scores s ON t.run_id = s.run_id AND t.task_id = s.task_id AND t.trial_index = s.trial_index
GROUP BY r.run_id, r.model_id, r.model_version, r.suite_hash, r.created_at;

CREATE OR REPLACE VIEW category_summary AS
SELECT
    r.run_id,
    r.model_id,
    s.task_id,
    s.grader_name,
    AVG(CASE WHEN s.passed THEN 1.0 ELSE 0.0 END) AS pass_rate,
    AVG(s.score) AS mean_score,
    SUM(t.cost_usd) AS category_cost_usd
FROM runs r
JOIN scores s ON r.run_id = s.run_id
JOIN trials t ON s.run_id = t.run_id AND s.task_id = t.task_id AND s.trial_index = t.trial_index
GROUP BY r.run_id, r.model_id, s.task_id, s.grader_name;

CREATE OR REPLACE VIEW head_to_head AS
SELECT
    a.run_id AS run_a,
    b.run_id AS run_b,
    a.task_id,
    a.passed AS passed_a,
    b.passed AS passed_b,
    a.score AS score_a,
    b.score AS score_b
FROM scores a
JOIN scores b ON a.task_id = b.task_id AND a.trial_index = b.trial_index
WHERE a.run_id != b.run_id;

CREATE OR REPLACE VIEW longitudinal AS
SELECT
    r.model_id,
    r.model_version,
    r.created_at,
    r.suite_hash,
    AVG(CASE WHEN s.passed THEN 1.0 ELSE 0.0 END) AS pass_rate,
    AVG(s.score) AS mean_score,
    SUM(t.cost_usd) AS total_cost_usd
FROM runs r
JOIN scores s ON r.run_id = s.run_id
JOIN trials t ON s.run_id = t.run_id AND s.task_id = t.task_id AND s.trial_index = t.trial_index
GROUP BY r.model_id, r.model_version, r.created_at, r.suite_hash
ORDER BY r.model_id, r.created_at;
