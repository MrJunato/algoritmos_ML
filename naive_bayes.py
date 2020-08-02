# definir função do naive bayes
def naive_bayes(df,Y_name,df_previsao):
    import numpy as np
    import pandas as pd
    
    # definir número pi
    pi = 3.141592653589793 
    
    # criar vetor com cada classe
    X = df.loc[:, df.columns != Y_name]
    x_cols = X.columns
    Y = df[Y_name]
    classes = df[Y_name].unique()
    
    # definir função que calcula a raiz quadrada
    def raizq(valor):
        return valor**0.5

    # definir função exponencial natural
    def exp(valor):
        e = 2.718281828459045
        return e**valor      
    
    # definir função familía densidade de probabilidade (FDP)
    # essa função cria uma matriz com o FDP calculado para todos os vetores
    def FDP(df,Y_name,df_previsao):
        prev_cols = ['pred: '+col for col in df_previsao.columns]
        pred_prob_cols = prev_cols + list(df.columns)
        pred_prob = pd.DataFrame(columns = pred_prob_cols)
        classes = df[Y_name].unique()
        for classe in classes:
            x_classe = df[df[Y_name]==classe].loc[:, df.columns != Y_name]
            mx = x_classe.mean()
            vx = x_classe.var()
            linha = pd.DataFrame(columns = pred_prob_cols)
            linha[prev_cols] = df_previsao
            linha.loc[:,x_cols] = exp(-0.5*(((df_previsao-mx)**2)/vx))/(raizq(2*pi*vx))
            linha[Y_name] = classe
            pred_prob = pred_prob.append(linha, ignore_index=True)
        return pred_prob
        
    # calcular probabilidades para cada classe
    df_prob = FDP(df,Y_name,df_previsao)
    for classe in classes:
        df_prob.loc[df_prob[Y_name] == classe, 'prob_classe'] = len(df[df[Y_name]==classe])/len(df[Y_name])
        
    # criar coluna P(classe|X)
    df_prob['prob_classe_dadoX'] = df_prob[x_cols].prod(axis=1)
    df_prob.drop(x_cols, axis = 1, inplace = True) # remover as colunas usadas pela multiplicação
    
    # criar coluna de P(classe)*P(classe|X)
    df_prob['prob_classe_e_prob_classe_dadoX'] = df_prob.loc[:,['prob_classe_dadoX','prob_classe']].prod(axis=1)
    df_prob.drop(['prob_classe_dadoX','prob_classe'], axis = 1, inplace = True) # remover as colunas usadas pela multiplicação
    
    # criar key para identificar as colunas com o mesmo conjunto X
    df_prob['key'] =  (df_prob.iloc[:,0:-2].astype(str).apply(''.join, axis=1)).astype(str)
    
    # calcular em uma nova matriz a somatória de P(classe)*P(classe|X) para cada conjunto diferente de X
    prob_B = pd.DataFrame(df_prob.loc[:,['prob_classe_e_prob_classe_dadoX','key']].groupby(['key'],as_index=False).sum())
    prob_B.columns = ['key','prob_B']
    
    # trazer o calculo feito no passo anterior para a matriz principal
    df_prob = df_prob.merge(prob_B,on='key',how='left')
     
    # probabilidade bayesiana para cada conjunto de X e Y
    df_prob['prob'] = df_prob['prob_classe_e_prob_classe_dadoX']/df_prob['prob_B']
    df_prob.drop(['prob_classe_e_prob_classe_dadoX','key','prob_B'], axis = 1, inplace = True) # remover as colunas usadas pelo calculo acima
    
    # cria dataframe que mostra o resultado
    df_pred = df_prob.iloc[df_prob.groupby(df_prob.columns[:-2])[df_prob.columns[-1]].max()]
    df_pred = df_pred.loc[~df_pred.index.duplicated(keep='first')]
    
    # cria dataframe que mostra a probabilidade de todas as classes
    df_prob = pd.concat([df_prob.drop(df_prob.columns[-2:], axis=1), (df_prob.pivot_table(columns=df_prob.columns[-2], values=df_prob.columns[-1]).reset_index().rename_axis(None, axis=1)).iloc[:,1:]], axis=1)
    df_prob.dropna(inplace=True)
    
    return df_pred, df_prob
