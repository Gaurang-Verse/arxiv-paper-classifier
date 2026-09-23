# Monitoring

This document describes what monitoring is implemented, what was measured,
and what is honestly *not* covered.

## What is implemented

### 1. Structured request logging
Every `/predict` call writes one JSON line to stdout:

    {"event": "prediction", "latency_ms": 41.2, "threshold_used": 0.2,
     "num_predicted_categories": 2, "top_category": "math.NA"}

Stdout is the standard place for container logs: Docker, Kubernetes, and log
aggregators all collect it without extra setup. Request text is deliberately
**not** logged, since abstracts could be unpublished work.

### 2. Prometheus metrics (`GET /metrics`)
| Metric | Type | Why |
|---|---|---|
| `prediction_requests_total` | Counter | Traffic volume |
| `prediction_latency_seconds` | Histogram | Latency percentiles (p50/p95/p99) |
| `predicted_category_count` | Histogram | Labels returned per request. A sudden move toward 0 or toward many labels is a cheap early sign of input or model problems |

Covered by `tests/test_api.py::test_metrics_endpoint_exposes_prometheus_format`.

### 3. Prediction drift check (`scripts/check_prediction_drift.py`)
Uses the Population Stability Index (PSI) to compare the distribution of
**predicted** categories between a reference batch and a current batch.
Bins are the top 15 categories from the reference batch plus an "other" bucket.

## Measured results (validation set, batch size 300)

| Check | PSI |
|---|---|
| Null test: two disjoint validation batches (same distribution) | 0.0536 |
| Positive control: batch of papers with a true `math.*` label | 1.1549 |

Rule of thumb: <0.1 no significant shift, 0.1–0.25 moderate, >0.25 significant.
The check stays quiet on same-distribution data and fires on a real shift.

## A mistake worth recording

The first version of the drift script reported **PSI = 1.3094 on
same-distribution validation data**, which is a false alarm. There were two causes:

1. **It compared the wrong distributions.** It compared model *predictions*
   against *true* label frequencies from the EDA. The model under-predicts rare
   categories (validation macro-F1 is only 0.33), so the gap between true and
   predicted labels looked like drift even though there was none.
2. **The bins were too sparse.** 172 categories across 200 papers left most
   bins empty. Clipping empty bins to a tiny epsilon inflates the log term.

Fix: compare predictions to predictions, and pool rare categories into
"other". Validating a drift metric with both a null test and a positive
control is what exposed the problem. Without them, the broken version
would have looked reasonable.

## What is NOT covered

- **No production traffic has been observed.** The API has not been deployed
  for real users. All drift numbers above come from the validation set used as
  a stand-in. They show that the check behaves sensibly. They do not show that
  real drift has been detected.
- **No Prometheus server, Grafana dashboard, or alerting is set up.** The
  `/metrics` endpoint exposes data in the standard format, but nothing
  scrapes it.
- **No scheduled drift job.** In production this script would run on a
  schedule over a window of logged requests. Here it is run by hand.
- **No ground-truth performance monitoring.** Live accuracy would need true
  labels for incoming papers (e.g. the categories arXiv eventually assigns),
  which this project does not collect.