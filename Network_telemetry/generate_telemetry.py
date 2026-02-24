import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# 7 days of per-minute data
minutes = 7 * 24 * 60
regions = ["us-east", "us-west", "eu-west", "ap-south"]
isps = ["ISP_A", "ISP_B", "ISP_C"]

rows = []
base_time = pd.Timestamp("2025-01-01")

for m in range(minutes):
    ts = base_time + pd.Timedelta(minutes=m)
    hour = ts.hour

    # Peak hours roughly 9–10 and 18–19 local time
    peak = hour in [9, 10, 18, 19]

    for region in regions:
        for isp in isps:
            # Traffic: higher during daytime / peak
            traffic = np.random.lognormal(
                mean=4 + 0.3 * (8 <= hour <= 22),
                sigma=0.5
            )

            # Latency: spikes in peak hours
            latency = np.random.normal(
                loc=40 + (20 if peak else 0),
                scale=8
            )

            # Jitter: more variable in peak
            jitter = np.random.exponential(
                scale=4 + (3 if peak else 0)
            )

            # Packet loss: small but higher in peak
            loss = abs(np.random.normal(
                loc=0.3 + (0.4 if peak else 0),
                scale=0.25
            ))

            # Random “incident” anomalies
            anomaly = np.random.rand() < 0.002
            if anomaly:
                latency *= np.random.uniform(2.0, 4.0)
                jitter *= np.random.uniform(2.0, 4.0)
                loss += np.random.uniform(1.0, 3.0)

            rows.append([
                ts, region, isp,
                traffic, latency, jitter, loss, anomaly
            ])

cols = [
    "timestamp", "region", "isp",
    "traffic_mbps", "latency_ms", "jitter_ms",
    "packet_loss_pct", "is_anomaly"
]
df = pd.DataFrame(rows, columns=cols)

# Isolation Forest anomaly detection on metrics
features = ["traffic_mbps", "latency_ms", "jitter_ms", "packet_loss_pct"]
X = df[features]

model = IsolationForest(
    contamination=0.01,
    random_state=42,
    n_estimators=100,
    max_samples=0.8,   # use 80% of rows for faster training
    n_jobs=-1 
)
model.fit(X)

preds = model.predict(X)   # -1 = anomaly, 1 = normal
df["anomaly_model"] = (preds == -1)

# Save to CSV for loading to SQL / Power BI
df.to_csv("network_telemetry_simulated.csv", index=False)
print("Saved network_telemetry_simulated.csv with", len(df), "rows")
print(df.head())
