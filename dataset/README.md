# Dataset directory

The dataset is not included in this repository because redistribution permission belongs to the original data providers.

After obtaining and preprocessing the H-BRP data, place the four requester-peer QoS matrices and two context files in this directory:

```text
SuccessRate{suffix}.csv
rightBlock{suffix}.csv
recentHeight{suffix}.csv
roundtripTime{suffix}.csv
ClientWithCTX.csv
PeerWithCTX.csv
```

Replace `{suffix}` with the experimental setting suffix, such as `_12_1000` or `_100_5000`.
