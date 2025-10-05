### Benchmark Monte Carlo de π  
N = 5,000,000 • block_size = 500,000 • repeats = 3
*Numba threads*: **16**

| Backend | Tempo (s) | Aceleração vs Python | π̂ | IC95% |
|---|---:|---:|---:|---:|
| py | 0.678 | 1.00× | 3.142657 | [3.141218, 3.144096] |
| numba | 0.097 | 6.96× | 3.140964 | [3.139524, 3.142404] |
| numba_par | 0.023 | 29.22× | 3.141301 | [3.139861, 3.142740] |