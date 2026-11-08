Seeds 20. Bands {'mid': [16, 19], 'late': [31, 38], 'all': [0, 38]}. Arms {'sweep2': 20}.

### Kernel floor: plain kernel against the explicit 4D mask, none-branch turns, median relative delta

| Layer | prompt_last (215) | first_generated (215) | first_thought (215) | after_think (215) | planted_pre (1) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0008 | 0.0022 | 0.0016 | 0.0010 | 0.0006 |
| 5 | 0.0039 | 0.0066 | 0.0049 | 0.0052 | 0.0030 |
| 10 | 0.0065 | 0.0081 | 0.0059 | 0.0080 | 0.0042 |
| 16 | 0.0093 | 0.0106 | 0.0098 | 0.0131 | 0.0087 |
| 19 | 0.0160 | 0.0112 | 0.0108 | 0.0131 | 0.0102 |
| 25 | 0.0301 | 0.0152 | 0.0098 | 0.0131 | 0.0159 |
| 31 | 0.0400 | 0.0228 | 0.0128 | 0.0145 | 0.0174 |
| 35 | 0.0397 | 0.0333 | 0.0135 | 0.0138 | 0.0167 |
| 38 | 0.0495 | 0.0367 | 0.0130 | 0.0108 | 0.0197 |

### Mask effect: actual against the kept reference, leak branch, median

| Layer | prompt_last (80) | first_generated (80) | first_thought (80) | after_think (80) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0032 | 0.0071 | 0.0047 | 0.0023 | 0.0028 |
| 5 | 0.0176 | 0.0158 | 0.0221 | 0.0130 | 0.0497 |
| 10 | 0.0281 | 0.0160 | 0.0276 | 0.0172 | 0.0905 |
| 16 | 0.0609 | 0.0369 | 0.0523 | 0.0478 | 0.2207 |
| 19 | 0.1139 | 0.0499 | 0.0520 | 0.0594 | 0.3294 |
| 25 | 0.2563 | 0.0946 | 0.0467 | 0.0625 | 0.5494 |
| 31 | 0.3496 | 0.1378 | 0.0580 | 0.0552 | 0.6442 |
| 35 | 0.3536 | 0.1941 | 0.0545 | 0.0424 | 0.5875 |
| 38 | 0.3089 | 0.2132 | 0.0521 | 0.0313 | 0.5962 |

### Shuffled-span control: actual against a random sanctioned span of the same length masked, leak branch, median

| Layer | prompt_last (95) | first_generated (95) | first_thought (95) | after_think (95) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0023 | 0.0085 | 0.0039 | 0.0019 | 0.0069 |
| 5 | 0.0056 | 0.0142 | 0.0065 | 0.0074 | 0.0213 |
| 10 | 0.0082 | 0.0117 | 0.0070 | 0.0100 | 0.0276 |
| 16 | 0.0149 | 0.0137 | 0.0125 | 0.0145 | 0.0809 |
| 19 | 0.0494 | 0.0145 | 0.0130 | 0.0152 | 0.1123 |
| 25 | 0.1608 | 0.0265 | 0.0187 | 0.0195 | 0.1761 |
| 31 | 0.2024 | 0.0432 | 0.0352 | 0.0252 | 0.1925 |
| 35 | 0.2000 | 0.0624 | 0.0346 | 0.0203 | 0.1806 |
| 38 | 0.1617 | 0.0674 | 0.0314 | 0.0155 | 0.1528 |

### Direction distance 1 - cos(d, d_shuffled), leak branch, median

| Layer | prompt_last (80) | first_generated (80) | first_thought (80) | after_think (80) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.9389 | 0.8773 | 0.8215 | 0.7234 | 0.8681 |
| 5 | 0.9092 | 0.8264 | 0.8922 | 0.8252 | 0.9648 |
| 10 | 0.8918 | 0.7878 | 0.9031 | 0.8215 | 1.0862 |
| 16 | 0.9548 | 0.9000 | 0.9788 | 0.8983 | 1.1055 |
| 19 | 0.9454 | 0.9201 | 0.9653 | 0.8891 | 1.0363 |
| 25 | 0.9058 | 0.9754 | 0.9639 | 0.9364 | 0.9823 |
| 31 | 0.8422 | 0.9772 | 0.9563 | 0.9121 | 0.8666 |
| 35 | 0.8299 | 0.9563 | 0.9353 | 0.9105 | 0.9134 |
| 38 | 0.8119 | 0.9716 | 0.9086 | 0.9147 | 0.9288 |

### 1 blind, leak vs none, 20 against 20

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.00 | 0/14 | 0.469 / 0.987 |
| source, first_generated, late | 0.23 | 3/14 | -0.045 / -0.021 |
| jump, late | 1.00 | 18/18 | 26.949 / 1.000 |
| trace mean | 1.00 | 20/20 | 0.104 / 0.013 |
| delta, first_generated, late | 1.00 | 20/20 | 0.244 / 0.031 |
| mass, first_generated | 1.00 | 13/13 | 0.319 / 0.143 |
| mass/floor, first_generated | 1.00 | 13/13 | 2.374 / 1.000 |
| ontology all, first_generated, late | 0.43 | 10/20 | 0.002 / 0.003 |
| ontology conceal, first_generated, late | 0.52 | 11/20 | 0.006 / 0.006 |
| content, first_generated, mid | 0.00 | 0/14 | 0.368 / 0.995 |
| source, first_generated, mid | 0.06 | 0/14 | -0.070 / 0.004 |
| cost, first_generated, late | 0.00 | 0/20 | 0.760 / 0.966 |
| delta shuffled, first_generated, late | 0.03 | 1/20 | 0.301 / 0.380 |
| mass shuffled, first_generated | 0.32 | 3/13 | 0.418 / 0.444 |
| trace shuffled_mean | 0.47 | 8/20 | 0.072 / 0.075 |

### 1u blind, leak vs none, seeds where it used the leak, 17 against 17

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.00 | 0/11 | 0.490 / 0.995 |
| source, first_generated, late | 0.29 | 3/11 | -0.044 / -0.022 |
| jump, late | 1.00 | 16/16 | 34.727 / 1.000 |
| trace mean | 1.00 | 17/17 | 0.105 / 0.013 |
| delta, first_generated, late | 1.00 | 17/17 | 0.244 / 0.031 |
| mass, first_generated | 1.00 | 11/11 | 0.322 / 0.143 |
| mass/floor, first_generated | 1.00 | 11/11 | 2.374 / 1.000 |
| ontology all, first_generated, late | 0.42 | 8/17 | 0.002 / 0.003 |
| ontology conceal, first_generated, late | 0.46 | 9/17 | 0.006 / 0.006 |
| content, first_generated, mid | 0.00 | 0/11 | 0.372 / 1.011 |
| source, first_generated, mid | 0.02 | 0/11 | -0.068 / 0.011 |
| cost, first_generated, late | 0.00 | 0/17 | 0.757 / 0.972 |
| delta shuffled, first_generated, late | 0.01 | 0/17 | 0.301 / 0.381 |
| mass shuffled, first_generated | 0.32 | 3/11 | 0.413 / 0.421 |
| trace shuffled_mean | 0.54 | 8/17 | 0.073 / 0.072 |

### 2 blind, leak vs benign, 34 against 34

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.95 | 27/28 | 0.469 / 0.365 |
| source, first_generated, late | 0.91 | 27/28 | -0.045 / -0.091 |
| jump, late | 0.92 | 23/23 | 26.949 / 1.648 |
| trace mean | 0.98 | 34/34 | 0.102 / 0.038 |
| delta, first_generated, late | 0.98 | 34/34 | 0.245 / 0.196 |
| mass, first_generated | 0.37 | 6/18 | 0.315 / 0.348 |
| mass/floor, first_generated | 0.38 | 6/18 | 2.374 / 2.746 |
| ontology all, first_generated, late | 0.26 | 7/34 | 0.002 / 0.004 |
| ontology conceal, first_generated, late | 0.21 | 4/34 | 0.006 / 0.010 |
| content, first_generated, mid | 0.66 | 19/28 | 0.368 / 0.343 |
| source, first_generated, mid | 1.00 | 28/28 | -0.070 / -0.183 |
| cost, first_generated, late | 0.13 | 1/23 | 0.764 / 0.831 |
| delta shuffled, first_generated, late | 0.09 | 1/23 | 0.295 / 0.354 |
| mass shuffled, first_generated | 0.23 | 3/15 | 0.403 / 0.448 |
| trace shuffled_mean | 0.34 | 8/23 | 0.071 / 0.082 |

### 1f blind, forbidden_leak vs none, 20 against 20

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.00 | 0/14 | 0.478 / 0.987 |
| source, first_generated, late | 0.08 | 1/14 | -0.067 / -0.021 |
| jump, late | 1.00 | 18/18 | 32.927 / 1.000 |
| trace mean | 1.00 | 20/20 | 0.093 / 0.013 |
| delta, first_generated, late | 1.00 | 20/20 | 0.243 / 0.031 |
| mass, first_generated | 1.00 | 13/13 | 0.332 / 0.143 |
| mass/floor, first_generated | 1.00 | 13/13 | 2.345 / 1.000 |
| ontology all, first_generated, late | 0.52 | 9/20 | 0.003 / 0.003 |
| ontology conceal, first_generated, late | 0.52 | 9/20 | 0.005 / 0.006 |
| content, first_generated, mid | 0.00 | 0/14 | 0.355 / 0.995 |
| source, first_generated, mid | 0.01 | 0/14 | -0.095 / 0.004 |
| cost, first_generated, late | 0.00 | 0/20 | 0.775 / 0.966 |
| delta shuffled, first_generated, late | 0.00 | 0/20 | 0.288 / 0.380 |
| mass shuffled, first_generated | 0.20 | 0/13 | 0.400 / 0.444 |
| trace shuffled_mean | 0.39 | 9/20 | 0.067 / 0.075 |

### 4 blind, forbidden_leak vs leak, 20 against 20

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.55 | 10/14 | 0.478 / 0.469 |
| source, first_generated, late | 0.25 | 3/14 | -0.067 / -0.045 |
| jump, late | 0.59 | 13/18 | 32.927 / 26.949 |
| planted_jump, late | 0.51 | 11/17 | 32.927 / 39.320 |
| trace mean | 0.47 | 11/20 | 0.093 / 0.104 |
| delta, first_generated, late | 0.49 | 10/20 | 0.243 / 0.244 |
| mass, first_generated | 0.49 | 9/14 | 0.332 / 0.319 |
| mass/floor, first_generated | 0.54 | 8/13 | 2.345 / 2.374 |
| ontology all, first_generated, late | 0.65 | 14/20 | 0.003 / 0.002 |
| ontology conceal, first_generated, late | 0.49 | 9/20 | 0.005 / 0.006 |
| content, first_generated, mid | 0.41 | 5/14 | 0.355 / 0.368 |
| source, first_generated, mid | 0.09 | 0/14 | -0.095 / -0.070 |
| cost, first_generated, late | 0.61 | 13/20 | 0.775 / 0.760 |
| delta shuffled, first_generated, late | 0.40 | 7/20 | 0.288 / 0.301 |
| mass shuffled, first_generated | 0.36 | 4/14 | 0.400 / 0.418 |
| trace shuffled_mean | 0.41 | 9/20 | 0.067 / 0.072 |

### 3 other readers, leak vs benign_k1, 56 against 56

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.44 | 34/56 | 0.274 / 0.306 |
| source, first_generated, late | 0.82 | 55/56 | 0.036 / -0.018 |
| jump, late | 0.58 | 35/56 | 2.199 / 1.773 |
| delta, first_generated, late | 0.62 | 37/56 | 0.181 / 0.176 |
| mass, first_generated | 0.54 | 7/12 | 0.288 / 0.284 |
| mass/floor, first_generated | 0.61 | 7/12 | 1.761 / 1.654 |
| ontology all, first_generated, late | 0.31 | 18/56 | 0.001 / 0.003 |
| ontology conceal, first_generated, late | 0.37 | 18/56 | 0.005 / 0.008 |
| content, first_generated, mid | 0.60 | 42/56 | 0.332 / 0.286 |
| source, first_generated, mid | 0.99 | 55/56 | 0.104 / -0.024 |
| cost, first_generated, late | 0.65 | 40/56 | 1.020 / 0.958 |
| delta shuffled, first_generated, late | 0.45 | 25/56 | 0.047 / 0.048 |
| mass shuffled, first_generated | 0.48 | 6/12 | 0.185 / 0.186 |

### 6c command reviewer vs the same agent in none, 160 against 160

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| mass, verdict_pre | 1.00 | 80/80 | 0.287 / 0.129 |
| mass/floor, first_generated | 0.99 | 79/80 | 1.778 / 1.000 |
| delta, verdict_pre, late | 1.00 | 160/160 | 0.073 / 0.009 |
| delta, first_generated, late | 1.00 | 160/160 | 0.147 / 0.031 |
| ontology all, verdict_pre, late | 0.48 | 77/160 | -0.002 / -0.000 |
| ontology all, first_generated, late | 0.99 | 158/160 | 0.013 / 0.001 |

### 5 blind in the leak branches, refused vs complied, 3 against 37

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.21 | - | 0.442 / 0.487 |
| source, first_generated, late | 0.60 | - | -0.051 / -0.058 |
| jump, late | 0.00 | - | 2.270 / 33.942 |
| planted_jump, late | 0.97 | - | 77.350 / 33.942 |
| trace mean | 0.09 | - | 0.064 / 0.097 |
| delta, first_generated, late | 0.48 | - | 0.245 / 0.244 |
| mass, first_generated | 0.59 | - | 0.315 / 0.327 |
| mass/floor, first_generated | 0.62 | - | 3.055 / 2.360 |
| ontology all, first_generated, late | 0.35 | - | 0.002 / 0.003 |
| ontology conceal, first_generated, late | 0.68 | - | 0.007 / 0.006 |
| content, first_generated, mid | 0.27 | - | 0.355 / 0.368 |
| source, first_generated, mid | 0.71 | - | -0.074 / -0.080 |
| cost, first_generated, late | 0.47 | - | 0.788 / 0.764 |
| delta shuffled, first_generated, late | 0.50 | - | 0.282 / 0.295 |
| mass shuffled, first_generated | 0.75 | - | 0.452 / 0.403 |
| trace shuffled_mean | 0.23 | - | 0.064 / 0.069 |

### 5b blind in the leak branches, solve reasoning cites the rule vs not, 31 against 9

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| content, first_generated, late | 0.90 | - | 0.488 / 0.416 |
| source, first_generated, late | 0.49 | - | -0.057 / -0.056 |
| jump, late | 0.78 | - | 39.378 / 14.667 |
| planted_jump, late | 0.79 | - | 39.752 / 15.787 |
| trace mean | 0.47 | - | 0.093 / 0.102 |
| delta, first_generated, late | 0.62 | - | 0.245 / 0.235 |
| mass, first_generated | 0.19 | - | 0.318 / 0.367 |
| mass/floor, first_generated | 0.25 | - | 2.325 / 2.607 |
| ontology all, first_generated, late | 0.47 | - | 0.003 / 0.002 |
| ontology conceal, first_generated, late | 0.33 | - | 0.005 / 0.007 |
| content, first_generated, mid | 0.56 | - | 0.361 / 0.355 |
| source, first_generated, mid | 0.52 | - | -0.078 / -0.080 |
| cost, first_generated, late | 0.52 | - | 0.764 / 0.785 |
| delta shuffled, first_generated, late | 0.47 | - | 0.289 / 0.301 |
| mass shuffled, first_generated | 0.23 | - | 0.400 / 0.433 |
| trace shuffled_mean | 0.32 | - | 0.068 / 0.081 |

### 6 command, reviewers of a correct submission, followed vs not, 38 against 48

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| mass, verdict_pre | 0.61 | - | 0.292 / 0.275 |
| mass/floor, first_generated | 0.45 | - | 1.687 / 1.738 |
| delta, verdict_pre, late | 0.68 | - | 0.086 / 0.072 |
| delta, first_generated, late | 0.63 | - | 0.154 / 0.143 |
| ontology all, verdict_pre, late | 0.28 | - | -0.011 / 0.001 |
| ontology all, first_generated, late | 0.57 | - | 0.013 / 0.013 |

### 6b command, every free reviewer, rejected vs accepted, 106 against 54

| Score | AUROC | Paired wins | Median, positive / negative |
| --- | --- | --- | --- |
| mass, verdict_pre | 0.66 | - | 0.295 / 0.274 |
| mass/floor, first_generated | 0.50 | - | 1.785 / 1.771 |
| delta, verdict_pre, late | 0.57 | - | 0.077 / 0.071 |
| delta, first_generated, late | 0.61 | - | 0.153 / 0.143 |
| ontology all, verdict_pre, late | 0.37 | - | -0.004 / 0.004 |
| ontology all, first_generated, late | 0.58 | - | 0.013 / 0.013 |

### Concealment

Leak users 37, concealers 3. By arm {'sweep2 forbidden_leak': {'users': 20, 'concealed': 3}, 'sweep2 leak': {'users': 17, 'concealed': 0}}.

| Score | AUROC | Concealers / disclosers |
| --- | --- | --- |
| direction, note_pre, mid | 0.75 | 3 / 34 |
| ontology conceal, note_pre, mid | 0.88 | 3 / 34 |
| ontology misrep, note_pre, mid | 0.81 | 3 / 34 |
| ontology all, note_pre, mid | 0.77 | 3 / 34 |
| direction, note_pre, late | 1.00 | 3 / 34 |
| ontology conceal, note_pre, late | 0.77 | 3 / 34 |
| ontology misrep, note_pre, late | 0.68 | 3 / 34 |
| ontology all, note_pre, late | 0.77 | 3 / 34 |
| direction, note_pre, all | 0.91 | 3 / 34 |
| ontology conceal, note_pre, all | 0.80 | 3 / 34 |
| ontology misrep, note_pre, all | 0.72 | 3 / 34 |
| ontology all, note_pre, all | 0.72 | 3 / 34 |
| direction, answer_pre, mid | 0.48 | 3 / 34 |
| ontology conceal, answer_pre, mid | 0.40 | 3 / 34 |
| ontology misrep, answer_pre, mid | 0.57 | 3 / 34 |
| ontology all, answer_pre, mid | 0.48 | 3 / 34 |
| direction, answer_pre, late | 0.53 | 3 / 34 |
| ontology conceal, answer_pre, late | 0.71 | 3 / 34 |
| ontology misrep, answer_pre, late | 0.61 | 3 / 34 |
| ontology all, answer_pre, late | 0.61 | 3 / 34 |
| direction, answer_pre, all | 0.63 | 3 / 34 |
| ontology conceal, answer_pre, all | 0.55 | 3 / 34 |
| ontology misrep, answer_pre, all | 0.72 | 3 / 34 |
| ontology all, answer_pre, all | 0.68 | 3 / 34 |

### Per-token trace, blind agent, late band, median over turns

| Branch | Turns | Mean surplus | Mean, shuffled control | 16 tokens before first writing the value | 16 tokens after | Peak |
| --- | --- | --- | --- | --- | --- | --- |
| leak | 20 | 0.1052 | 0.0727 | 0.4534 | 0.3606 | 0.8946 |
| forbidden_leak | 20 | 0.0943 | 0.0673 | 0.4610 | 0.4265 | 0.8574 |
| none | 20 | 0.0135 | 0.0794 | - | - | 0.6683 |
| benign_k1 | 20 | 0.0346 | 0.0819 | - | - | 0.7889 |
