# algoritmos_ML

Com o objetivo de estudar a aprender melhor sobre os algoritmos de aprendizado de máquina decidi recria-los do zero, porém adicionei um desafio que é usar apenas o numpy e o pandas para isso. Então nesse repositório estarão armazenados todos os algoritmos que eu reescrever e tiver sucesso.

## Naive Bayes Gaussiano

O primeiro e único algoritmo postado até o momento é o naive bayes gaussiano, desenvolvi uma função que recebe: um dataframe do pandas, o nome em string da variável resposta e um dataframe que possua as variáveis explicativas que precisam de uma previsão. 
Com esses três argumentos a minha função retorna dois dataframes. 

- O primeiro dataframe mostra todas as variáveis explicativas e qual seria a classe com maior probabilidade de acontecer;
- O segundo dataframe além de também mostrar todas as variáveis explicativas mostra qual é a probabilidade de acontecer cada classe;

Essa versão do Naive Bayes é interessante porque retorna os dataframes formatados e as probabilidades das classes, podendo ser utilizado para gerar insights da base dados em questão.
