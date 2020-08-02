# algoritmos_ML

Com o objetivo de estudar a aprender melhor sobre os algoritmos de aprendizado de máquina decidi recria-los do zero, porém adicionei um desafio que é usar apenas o numpy e o pandas para isso, não podendo nem mesmo usar a biblioteca "math" ou "statistics". Então nesse repositório estarão armazenados todos os algoritmos que eu reescrever e tiver sucesso.

## Naive Bayes Gaussiano

O primeiro e único algoritmo postado até o momento é o naive bayes gaussiano, desenvolvi uma função que recebe: um dataframe do pandas, o nome em string da variável resposta e um dataframe que possua as variáveis explicativas que precisam de uma previsão. 
Com esses três argumentos a minha função retorna dois dataframes. 

- O primeiro dataframe mostra todas as variáveis explicativas e qual seria a classe com maior probabilidade de acontecer;
- O segundo dataframe além de também mostrar todas as variáveis explicativas mostra qual é a probabilidade de acontecer cada classe;

Essa versão do Naive Bayes é interessante porque retorna os dataframes formatados e as probabilidades das classes, podendo ser utilizado para gerar insights da base dados em questão.

### Exemplo
No exemplo abaixo foi utilizado o clássico dataframe Iris junto com a função Naive Bayes em questão.

```python
# Importando bibliotecas
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carrecando Iris
iris = load_iris()
dados = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['species'])

# Substituindo os números das espécies pelo nome da especie
dados['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) 

# Separando em treino e teste
train, test = train_test_split(dados, test_size=0.2)

# Aplicando a função Naive Bayes
df_result,df_prob = naive_bayes(train,'species',test)

# Vendo o resultado
print(df_result)
print(df_prob)
```

O output do `print(df_result)` é um dataframe com todas as variáveis explicativas e a previsão da variável resposta com sua probabilidade de estar correta.
![Image of result](https://github.com/MrJunato/algoritmos_ML/blob/master/df_result.png)

O output do `print(df_prob)` é um dataframe também com todas as variáveis explicativas, porém ele mostra também a probabilidade de ocorrer cada uma das classes.
![Image of prob](https://github.com/MrJunato/algoritmos_ML/blob/master/df_prob.png)
