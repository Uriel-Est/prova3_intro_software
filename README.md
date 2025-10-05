# Aceleração Python com Numba e Visualização com Matplotlib/Plotnine

## Introdução

Este projeto demonstra a aplicação de ferramentas essenciais para ciência de dados em Python: **Numba** para aceleração de código numérico e **matplotlib/plotnine** para visualização de dados. Através da implementação do método de Monte Carlo para estimação do valor de π, exploramos como essas ferramentas podem ser combinadas para criar soluções eficientes e visualmente informativas.

O **Numba** é um compilador JIT (Just-In-Time) que acelera significativamente funções Python que envolvem operações matemáticas intensivas e loops, convertendo código Python em instruções de máquina otimizadas em tempo de execução . Já as bibliotecas de visualização **matplotlib** e **plotnine** permitem criar gráficos estáticos e animados para comunicar efetivamente os resultados de análises de dados.

## Parte 1 - Plotagem em Python

### Visualização com Matplotlib

O **matplotlib** é uma biblioteca abrangente para criação de visualizações estáticas, animadas e interativas em Python. Sua sintaxe versátil permite controle detalhado sobre todos os elementos do gráfico.

```python
# Exemplo de gráfico de linhas com matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=2, label='Seno(x)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Função Seno')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Visualização com Plotnine

O **plotnine** implementa uma interface baseada na gramática de gráficos do ggplot2 do R, oferecendo uma abordagem declarativa diferente para construção de visualizações.

```python
# Exemplo com plotnine (requer instalação: pip install plotnine)
from plotnine import ggplot, aes, geom_line, labs, theme_bw
import pandas as pd

df = pd.DataFrame({'x': x, 'y': y})

(ggplot(df, aes(x='x', y='y')) 
 + geom_line(color='blue', size=1)
 + labs(x='X', y='Y', title='Função Seno')
 + theme_bw()
)
```

### Diferenças de Sintaxe e Estilo

- **Matplotlib**: Abordagem imperativa com controle preciso sobre cada elemento
- **Plotnine**: Abordagem declarativa baseada em camadas (similar ao ggplot2)
- **Animação**: O matplotlib oferece módulos específicos (`FuncAnimation`) para criar GIFs e vídeos
- **Personalização**: Matplotlib permite ajustes mais granulares; plotnine oferece temas pré-definidos

## Parte 2 - Numba e Aceleração de Código

### O Método de Monte Carlo para π

O método de Monte Carlo estima π através de amostragem aleatória: geramos pontos aleatórios dentro de um quadrado unitário e calculamos a proporção que cai dentro do quarto de círculo unitário. A estimativa é dada por π ≈ 4 × (pontos dentro do círculo) / (total de pontos).

### Implementação em Python Puro

```python
def pi_monte_carlo_py(n: int, seed: Optional[int] = None) -> float:
    """
    Estima π por Monte Carlo em Python puro: amostra n pontos (x,y) ~ U(0,1)^2
    e estima π = 4 * P(x^2 + y^2 ≤ 1). Custo O(n).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n deve ser int positivo")
    import random
    if seed is not None:
        random.seed(seed)
    hits = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return 4.0 * (hits / n)
```

### Aceleração com Numba

```python
from numba import njit, prange

@njit(fastmath=True, cache=True)
def mc_hits_block(n: int, seed: int) -> int:
    """
    Núcleo NUMBA (serial): retorna o número de acertos no quarto de círculo.
    """
    np.random.seed(seed)
    hits = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return hits

@njit(fastmath=True, parallel=True, cache=True)
def mc_hits_block_par(n: int, seed: int) -> int:
    """
    Núcleo NUMBA (paralelo): usa prange para distribuir o laço.
    """
    np.random.seed(seed)
    hits = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return hits
```

### Benchmark e Resultados

### Benchmark Monte Carlo de π  
N = 5,000,000 • block_size = 500,000 • repeats = 3  
*Numba threads*: **16**

| Backend | Tempo (s) | Aceleração vs Python | π̂ | IC95% |
|---|---:|---:|---:|---:|
| py | 0.678 | 1.00× | 3.142657 | [3.141218, 3.144096] |
| numba | 0.097 | 6.96× | 3.140964 | [3.139524, 3.142404] |
| numba_par | 0.023 | 29.22× | 3.141301 | [3.139861, 3.142740] |

### Como o Numba Acelera o Código

O Numba funciona como um compilador JIT (Just-In-Time) que :
- **Analisa o bytecode Python** da função decorada com `@jit` ou `@njit`
- **Infere os tipos de dados** utilizados nos cálculos
- **Gera código de máquina otimizado** usando o compilador LLVM
- **Executa diretamente o código compilado** em chamadas subsequentes

O parâmetro `nopython=True` (ou o decorador `@njit`) garante que a função seja compilada sem dependências do interpretador Python, maximizando o desempenho . Já o `parallel=True` combinado com `prange` permite o paralelismo automático em múltiplos cores de CPU .

## Conclusão

Este projeto demonstrou efetivamente o poder das ferramentas modernas de Python para ciência de dados:

### Principais Aprendizados

1. **Numba para desempenho**: O Numba pode acelerar dramaticamente código numérico - em nosso caso, **29× mais rápido** que Python puro usando paralelismo. É particularmente eficaz para loops intensivos e operações com NumPy .

2. **Visualização flexível**: Tanto matplotlib quanto plotnine oferecem abordagens poderosas para visualização, atendendo a diferentes preferências de estilo e fluxo de trabalho.

3. **Integração eficiente**: As ferramentas podem ser facilmente combinadas - usando Numba para cálculos intensivos e matplotlib/plotnine para visualização dos resultados.

### Importância para Ciência de Dados

O Numba resolve uma das principais limitações do Python - a velocidade de execução - especialmente para :
- Simulações de Monte Carlo
- Processamento numérico intensivo
- Algoritmos com loops complexos
- Operações com arrays NumPy

Já as bibliotecas de visualização são essenciais para:
- Exploração e compreensão de dados
- Comunicação de resultados
- Validação de modelos e métodos
- Criação de relatórios e dashboards

Esta combinação de ferramentas permite que cientistas de dados aproveitem a simplicidade e ecosystem do Python sem sacrificar o desempenho computacional necessário para problemas complexos.

## Referências

### Tutoriais e Documentação Oficial

1. **Numba Documentation Oficial** - [https://numba.pydata.org/](https://numba.pydata.org/)  
   Guias completos, tutoriais e referência de API da biblioteca Numba .

2. **Hashtag Treinamentos - Numba Python**  
   Tutorial prático em português mostrando exemplos de aceleração com Numba, incluindo comparações de desempenho .

3. **Matplotlib Documentation** - [https://matplotlib.org/stable/tutorials/index.html](https://matplotlib.org/stable/tutorials/index.html)  
   Documentação oficial com exemplos, tutoriais e galeria de visualizações.

4. **Plotnine Documentation** - [https://plotnine.readthedocs.io/](https://plotnine.readthedocs.io/)  
   Guia de referência para a gramática de gráficos com plotnine.

5. **Python 3.13 Documentation** - [https://docs.python.org/3] (https://docs.python.org/3)
    Documentação e tutorial geral para Python (Parte 2 = NumPy e Parte 5 = Matplotlib)

### Artigos e Tutoriais Especializados

6. **Medium: Accelerated Python - Numba**  
   Explicação detalhada sobre o funcionamento do Numba e casos de uso apropriados .

7. **DataCamp: O que é Ciência de Dados**  
   Guia abrangente sobre o campo de ciência de dados e seu ecossistema de ferramentas .

8. **AWS: O que é Ciência de Dados**  
   Visão geral dos conceitos e processos em ciência de dados .

9. **Alura: O que é Data Science**  
   Artigo em português sobre fundamentos e aplicações da ciência de dados .

### APIS de Dados utilizados

11. **IBGE SIDRA** - [sidra.ibge.gov.br](sidra.ibge.gov.br)
    Sistema IBGE de Recuperação Automática -- Banco de dados do IBGE

#### Livro

12. **Python Data Science Handbook** - [https://jakevdp.github.io/PythonDataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook)
