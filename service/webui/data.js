// baked snapshot of results/*.json — the Overview falls back to this when no backend is reachable
window.RECOUP_RESULTS = {
  "headline": {
    "per_case": 2681,
    "ci": [
      1427,
      4162
    ],
    "n": 400
  },
  "recoup_vs_fixed": {
    "net_per_case": 2681,
    "ci": [
      1427.3,
      4161.52
    ]
  },
  "scoreboard": [
    {
      "name": "never act",
      "recovered": 0,
      "rate": 0.0,
      "attempts": 0,
      "msg": 0.0,
      "on_time": 0.0,
      "preserved": 0.88,
      "net": 9549130
    },
    {
      "name": "fixed schedule",
      "recovered": 302303,
      "rate": 0.3275,
      "attempts": 994,
      "msg": 0.73,
      "on_time": 0.3,
      "preserved": 0.88,
      "net": 9560513
    },
    {
      "name": "always nudge",
      "recovered": 290712,
      "rate": 0.31,
      "attempts": 966,
      "msg": 1.812,
      "on_time": 0.255,
      "preserved": 0.855,
      "net": 9565256
    },
    {
      "name": "+ cause classifier",
      "recovered": 386543,
      "rate": 0.425,
      "attempts": 804,
      "msg": 0.745,
      "on_time": 0.3975,
      "preserved": 0.8925,
      "net": 9800417
    },
    {
      "name": "+ funding-window timing",
      "recovered": 724340,
      "rate": 0.735,
      "attempts": 553,
      "msg": 0.168,
      "on_time": 0.26,
      "preserved": 0.9425,
      "net": 10588594
    },
    {
      "name": "Recoup",
      "recovered": 717314,
      "rate": 0.7175,
      "attempts": 544,
      "msg": 0.168,
      "on_time": 0.3,
      "preserved": 0.9425,
      "net": 10632992
    }
  ],
  "ablation": [
    {
      "name": "no timing",
      "net": 2438,
      "gap": 83.8,
      "recovery": 0.69,
      "on_time": 0.3375
    },
    {
      "name": "no cause",
      "net": 1958,
      "gap": 70.5,
      "recovery": 0.6325,
      "on_time": 0.15
    },
    {
      "name": "liquidity aware",
      "net": 2570,
      "gap": 94.2,
      "recovery": 0.735,
      "on_time": 0.26
    },
    {
      "name": "recoup",
      "net": 2681,
      "gap": 90.2,
      "recovery": 0.7175,
      "on_time": 0.3
    },
    {
      "name": "oracle",
      "net": 2443,
      "gap": 100.0,
      "recovery": 0.76,
      "on_time": 0.33
    }
  ],
  "oracle_gap": 43.2,
  "recoup_minus_oracle": 95354,
  "sensitivity": [
    {
      "name": "baseline priors",
      "v": 2491,
      "lo": 811,
      "hi": 4413
    },
    {
      "name": "fewer cashflow more dead",
      "v": 2768,
      "lo": 1435,
      "hi": 4415
    },
    {
      "name": "churn hazard \u00d71.8",
      "v": 2127,
      "lo": 645,
      "hi": 3839
    },
    {
      "name": "low ltv",
      "v": 1269,
      "lo": 605,
      "hi": 2023
    },
    {
      "name": "long outages",
      "v": 2535,
      "lo": 855,
      "hi": 4461
    },
    {
      "name": "deeper shortfalls",
      "v": 2131,
      "lo": 458,
      "hi": 4067
    }
  ],
  "fairness": [
    {
      "group": "Business",
      "n": 69,
      "recovery": 0.652,
      "vs_fixed": 0.435,
      "on_time": 0.391,
      "escalated": 0.072,
      "net": 3862,
      "ci": [
        1335,
        7473
      ]
    },
    {
      "group": "Gig",
      "n": 129,
      "recovery": 0.667,
      "vs_fixed": 0.217,
      "on_time": 0.434,
      "escalated": 0.047,
      "net": 2381,
      "ci": [
        438,
        4976
      ]
    },
    {
      "group": "Salaried",
      "n": 202,
      "recovery": 0.772,
      "vs_fixed": 0.485,
      "on_time": 0.183,
      "escalated": 0.079,
      "net": 2470,
      "ci": [
        407,
        4562
      ]
    }
  ],
  "classifier": {
    "acc": 0.9,
    "ece": 0.047,
    "ece_raw": 0.085
  },
  "liquidity": {
    "mae": 6.1,
    "naive": 11.8,
    "p85_cov": 0.83
  }
};
