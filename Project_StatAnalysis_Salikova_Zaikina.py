# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp

# %%
data = pd.read_csv('movie_bd_v5.csv')
data.sample(5)

# %%
#удаляем столбцы, не участвующие в рассмотрении
data.drop(['imdb_id', 'original_title', 'tagline','overview','production_companies','release_date'], inplace=True, axis=1)

# %%
#нформация  о содержимом dataframe
data.info()

# %%
"""
Пропусков нет
"""

# %%
# построение гистограмм с выводом описательной характеристики
quantitative_variables = ['budget', 'revenue', 'runtime', 'vote_average', 'release_year']
for variable in quantitative_variables:
    plt.hist(data[variable], 12, density=1, facecolor='c')
    plt.grid(True)
    plt.xlabel("Значения")
    plt.ylabel("Относительная частота")
    plt.title(f'Распределение по {variable}')
    plt.savefig(f'{variable}.png', bbox_inches='tight')
    plt.show()
    print(sp.describe(data[variable], ddof=1, bias=False))
    print(data[variable].describe())

# %%
# избавляемся от "|" в столбцах director, cast, genres
data2 = data
data2['director'] = data.director.apply(lambda x: str(x).split('|'))
data2['cast'] = data.cast.apply(lambda x: str(x).split('|'))
data2['genres'] = data.genres.apply(lambda x: str(x).split('|'))
data3 = data2.explode('director')
data4 = data3.explode('cast')
data5 = data4.explode('genres')
data5


# %%
# построение гистограммы распредления жанров
plt.hist(data5.genres, bins=19, facecolor='c', density=1)
plt.xlabel("Жанр")
plt.ylabel("Относительная частота")
plt.title("Распределние по genres")
plt.xticks(rotation='vertical')
plt.savefig('genres.png', bbox_inches='tight')


# %%
# гистограмма, показывающая, сколько фильмов сняли режиссеры в период с 2000 по 2015
data3 = data2.explode('director')
plt.hist(data3.director.value_counts(), facecolor='c', density=1)
plt.annotate('Выброс', xy=(12.5, 0.001), xytext=(12 ,0.1), arrowprops=dict(facecolor='black', arrowstyle='fancy'))
plt.xlabel("Количество фильмов, снятых режиссером")
plt.ylabel("Относительная частота")
plt.title("Распределение по director")
plt.xticks(rotation='vertical')
plt.savefig('film_director.png', bbox_inches='tight')

# %%
# гистограмма, показывающая в скольких фильмах снялись актеры в период с 2000 по 2015
plt.hist(data4.cast.value_counts(), facecolor='c', density=1)
plt.annotate('Выброс', xy=(25, 0.01), xytext=(25 ,0.05), arrowprops=dict(facecolor='black', arrowstyle='fancy'))
plt.xlabel("Количество фильмов, в которых снялся актер")
plt.ylabel("Относительная частота")
plt.title("Распределение по cast")
plt.xticks(rotation='vertical')
plt.savefig('film_cast.png', bbox_inches='tight')
plt.show()

# %%
# функция для построения диаграмм Бокса-Уискера 
def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='vote_average', 
                data=data5.loc[data5.loc[:, column].isin(data5.loc[:, column].value_counts().index[:10])],
               ax=ax, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"6"})
    plt.xticks(rotation=45)
    ax.set_title(f'Распределение vote average по {column} ')
    plt.savefig(f'boxplot_{column}_vote_average', bbox_inches='tight')

# %%
for col in ['director', 'cast']:
    get_boxplot(col)
    
    
    

# %%
# построение диаграммы рассеивания для переменных budget и vote_average
data.plot(x = 'vote_average',
          y = 'budget',
          kind = 'scatter',
          grid = True,
          title = 'Д.рассеивания для пары бюджет, рейтинг',
          color='c')
plt.savefig("scatter_budget_vote_average")

# %%
# построение диаграммы рассеивания для переменных budget и runtime
data.plot(x = 'budget',
          y = 'runtime',
          kind = 'scatter',
          grid = True,
          title = 'Д.рассеивания для пары длительность, бюджет',
          color='c')
plt.savefig("scatter_runtime_budget")

# %%
df = data4[data4.director.isin(data4.loc[:, "director"].value_counts().index[:12])].copy()
 
df2 = df[df.cast.isin(df.loc[:, "cast"].value_counts().index[:10])].copy()
 
df3 = pd.crosstab(df2.director, df2.cast)
df3.head(10)

# %%
# пользуемся критерием хи-квадрат для определния наличия статистической связи 
table = np.array([[1,0,0,0,2,0,1,1,0,0], 
         [0,4,0,0,0,0,2,0,0,0], 
         [1,0,0,0,2,0,1,1,0,0], 
         [0,0,0,0,0,0,1,0,5,0],
         [0,0,0,0,0,1,0,0,0,0], 
         [0,0,0,0,1,0,0,0,2,2],
         [3,0,0,0,0,0,0,3,0,0],
         [0,0,5,0,0,0,5,0,0,0],
         [0,0,0,0,0,0,0,0,0,3],
         [0,0,0,7,0,5,0,0,0,0]])
 
chi2, prob, df, expected = sp.chi2_contingency(table)
output = "test Statistics: {}\ndegrees of freedom: {}\np-value: {}\n"
print(output.format( chi2, df, prob))
print(expected)

# %%
# выычисление коэффициента Крамера
def cramers_stat(confusion_matrix):
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))

result = cramers_stat(table)
result

# %%
# построение диаграммы рассеивания для переменных budget и release_year
data.plot(x = 'budget',
          y = 'release_year',
          kind = 'scatter',
          grid = True,
          title = 'Д.рассеивания для пары бюджет, рейтинг',
          color='c')
plt.savefig("scatter_budget_release_year.png")

# %%
# вычисление коэффициентов корреляции Пирсона, Спирмена и тау Кендалла
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
print(pearsonr(data.budget, data.runtime))
print(spearmanr(data.budget, data.runtime))
print(kendalltau(data.budget, data.runtime))
print('\n')
print(pearsonr(data.budget, data.vote_average))
print(spearmanr(data.budget, data.vote_average))
print(kendalltau(data.budget, data.vote_average))

# %%
# поиск корреляци между пременными budget и release_year
np.corrcoef(data.budget, data.release_year)

# %%
