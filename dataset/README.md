# Dataset directory

The dataset is not included in this repository. For dataset availability, access, and preprocessing instructions, please refer to the H-BRP source paper and its accompanying official repository:

- P. Zheng, Z. Zheng, and L. Chen, "Selecting reliable blockchain peers via hybrid blockchain reliability prediction," *IET Software*, vol. 17, no. 4, pp. 362--377, 2023. [https://doi.org/10.1049/sfw2.12118](https://doi.org/10.1049/sfw2.12118)
- Original H-BRP data and implementation repository: [https://github.com/InPlusLab/BlockchainReliabilityPrediction](https://github.com/InPlusLab/BlockchainReliabilityPrediction)

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
