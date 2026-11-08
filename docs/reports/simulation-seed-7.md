# decision-gate simulation report

Seed 7; 200 decisions per day per question; 21 days of history, 14 steady, 14 drifted. Model overconfidence 2.0, reviewer accuracy 100%, 10% of auto decisions get their true outcome after 3 days, refit and refresh every 7 days. All error rates are measured against the simulation's ground truth.

## `refund.eligible@v1`

- steady: 94.4% automated, true auto error 130/2643 = 4.9% (95% CI 4.1%-5.8%), within the 10% budget
- drift from day 36: detected at the end of day 42 (7 day(s) of drifted traffic), auto tier suspended at the end of day 42
- drift, until the action: true auto error 153/1332 = 11.5% (95% CI 9.8%-13.3%)
- drift, after the action: nothing automated

| phase | days | decisions | auto | verify | human | hard + novel | ECE raw | ECE calibrated | true auto error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| history | 1-21 | 4200 | 0.0% | 0.0% | 100.0% | 15.4% | 0.043 | - | - |
| steady | 22-35 | 2800 | 94.4% | 5.6% | 0.0% | 15.7% | 0.045 | 0.007 | 130/2643 = 4.9% (95% CI 4.1%-5.8%) |
| drift | 36-49 | 2800 | 47.6% | 52.4% | 0.0% | 45.1% | 0.092 | 0.034 | 153/1332 = 11.5% (95% CI 9.8%-13.3%) |

Events:

- day 21: calibrator v1 activated (isotonic, 4200 labeled decisions)
- day 21: routing refreshed: auto >= 0.6667, verify >= 0.6667
- day 28: refit rejected: isotonic held-out log loss 0.1823 does not beat active v1 (isotonic) 0.1823
- day 28: routing refreshed: auto >= 0.6667, verify >= 0.6667
- day 35: refit rejected: isotonic held-out log loss 0.1926 does not beat active v1 (isotonic) 0.1923
- day 35: routing refreshed: auto >= 0.6667, verify >= 0.6667
- day 42: refit rejected: isotonic held-out log loss 0.1929 does not beat active v1 (isotonic) 0.1924
- day 42: routing refreshed: auto >= 0.6667, verify >= 0.6667
- day 42: drift check: drift (signals: label_psi); action suspended: auto tier suspended (was 0.6667): 138 eligible labeled decisions in the window, fewer than min_labeled=200
- day 43: drift check: drift (signals: label_psi); action already_tightened
- day 44: drift check: drift (signals: label_psi); action already_tightened
- day 45: drift check: drift (signals: label_psi); action already_tightened
- day 47: drift check: drift (signals: label_psi); action already_tightened
- day 48: drift check: drift (signals: label_psi); action already_tightened
- day 49: calibrator v5 activated (isotonic, 6196 labeled decisions)
- day 49: routing refreshed: auto >= 0.7652, verify >= 0.5000
- day 49: drift check: drift (signals: label_psi); action tightened: auto threshold raised from 0.7652 to 0.7668: re-selected on 1400 labeled decisions in the window

## `ticket.queue@v1`

- steady: 67.4% automated, true auto error 100/1888 = 5.3% (95% CI 4.3%-6.4%), not distinguishable from the 5% budget
- drift from day 36: detected at the end of day 39 (4 day(s) of drifted traffic), auto tier suspended at the end of day 39
- drift, until the action: true auto error 47/433 = 10.9% (95% CI 8.1%-14.2%)
- drift, after the action: nothing automated

| phase | days | decisions | auto | verify | human | hard + novel | ECE raw | ECE calibrated | true auto error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| history | 1-21 | 4200 | 0.0% | 0.0% | 100.0% | 26.6% | 0.094 | - | - |
| steady | 22-35 | 2800 | 67.4% | 32.6% | 0.0% | 26.9% | 0.102 | 0.016 | 100/1888 = 5.3% (95% CI 4.3%-6.4%) |
| drift | 36-49 | 2800 | 15.5% | 84.5% | 0.0% | 50.8% | 0.175 | 0.075 | 47/433 = 10.9% (95% CI 8.1%-14.2%) |

Events:

- day 21: calibrator v1 activated (platt, 4200 labeled decisions)
- day 21: routing refreshed: auto >= 0.8814, verify >= 0.4053
- day 28: calibrator v2 activated (isotonic, 4546 labeled decisions)
- day 28: routing refreshed: auto >= 0.9469, verify >= 0.3333
- day 35: refit rejected: isotonic held-out log loss 0.3575 does not beat active v2 (isotonic) 0.3570
- day 35: routing refreshed: auto >= 0.9469, verify >= 0.3333
- day 39: drift check: drift (signals: ece, auto_error); action suspended: auto tier suspended (was 0.9469): no threshold on the window's labels: even the highest threshold 0.9468 has a 95% lower bound of 0.8769 on precision (220/241), below the target 0.9500
- day 40: drift check: drift (signals: label_psi, ece); action already_tightened
- day 41: drift check: drift (signals: label_psi, ece); action already_tightened
- day 42: refit rejected: isotonic held-out log loss 0.4616 does not beat active v2 (isotonic) 0.4586
- day 42: routing refreshed: auto >= -, verify >= 0.3333
- day 42: drift check: drift (signals: label_psi, ece); action already_tightened
- day 43: drift check: drift (signals: label_psi, ece); action already_tightened
- day 44: drift check: drift (signals: label_psi, ece); action already_tightened
- day 45: drift check: drift (signals: label_psi, ece); action already_tightened
- day 46: drift check: drift (signals: label_psi, ece); action already_tightened
- day 47: drift check: drift (signals: label_psi, ece); action already_tightened
- day 48: drift check: drift (signals: ece); action already_tightened
- day 49: refit rejected: temperature held-out log loss 0.4531 does not beat active v2 (isotonic) 0.4414
- day 49: routing refreshed: auto >= -, verify >= 0.3333
