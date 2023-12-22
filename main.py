"""
Zadanie: Klasyfikacja danych medycznych

Autorzy:
 - Aneta Pietrzak
 - Kacper Filipiuk

"""

# Import bibliotek do analizy danych
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Biblioteki do modelowania
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Pliki w projekcie
from ROC_curve import ROC
from model_evaluation import ocen_model_klasyfikacji
from neural_network_information import info_o_sieci

# Wczytaj dane z pliku CSV
df = pd.read_csv('heart.csv')

# Wyświetl kilka pierwszych wierszy danych, aby zobaczyć, jak wyglądają
print(df.head())

# Sprawdź informacje o danych, takie jak typy zmiennych, brakujące wartości itp.
print(df.info())
print(df.describe())

# Unikalne wartości w kolumnach
for column in df.columns:
    uniq_value = df[column].unique()
    print(f'Unikalne wartości w kolumnie {column}:\n{uniq_value}\n')

# Ogólna liczba osób chorych i zdrowych.
# WNIOSEK: Grupy w miare rownoliczne
sick_people = df[df['HeartDisease'] == 1].shape[0]
healthy_people = df[df['HeartDisease'] == 0].shape[0]
print("Ogólna liczba osób chorych:", sick_people)
print("Ogólna liczba osób zdrowych:", healthy_people)
print()

# Zauwazmy, ze jest jedna obserwacja która ma wartosc 0 w kolumnie 'RestingBP'
# WNIOSEK: Zmienna 'RestingBP' nie moze przyjmowac wartosci 0. Usuniecie obserwacji nie powinno zaburzyć wyników
print((df['RestingBP'] == 0).sum())
print()
df.drop(df[df['RestingBP'] == 0].index, inplace=True)

# Eksploracja rozkładów zmiennych numerycznych
# Zaprezentowanie rozkładów zmiennych numerycznych poprzez wykresy histogramów
num_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df[num_vars].hist(bins=20, figsize=(15, 10))
plt.suptitle('Rozkłady zmiennych numerycznych')
plt.show()

# Wykresy pudełkowe dla zmiennych kategorycznych w zależności od HeartDisease (zmienna celu)
num_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for var in num_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='HeartDisease', y=var, data=df)
    plt.title(f'Wykres pudełkowy dla {var} w zależności od HeartDisease')
    plt.xticks([0, 1], ['No', 'Yes'])  # Zmiana etykiet osi X
    plt.show()

# Grupowanie danych według zmiennej HeartDisease i obliczenie podstawowych statystyk dla zmiennych numerycznych
statistic_in_groups = df.groupby('HeartDisease')[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].describe().T
print(statistic_in_groups)
print()

# Zauwazmy, ze w kolumnie cholesterol jest 171 obserwacji dla ktorych wartosc wynosi 0
# Jest to niecałe 20%, zatem nie jest to 'jednorazowy przypadek'
print("Ilosc obserwacji dla ktorych wartosc cholesterol wynosi:", (df['Cholesterol'] == 0).sum())
print()

# Zauwazmy, że zmienna Cholesterol ma niski wskaznik korelacji ze zmienna HeartDisease - zmienna celu.
# Spróbujmy zastapic ja mediana wartosci w tej kolumnie oraz zaznaczmy na wykresie jej rozklad
median_chol = df['Cholesterol'].median()
df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)
df['Cholesterol'].hist(bins=20, figsize=(15, 10))
plt.suptitle('Nowy rozkład zmiennej cholesterol')
plt.show()

# Wykresy słupkowe dla zmiennych kategorycznych w zależności od HeartDisease
cat_vars = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for var in cat_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=var, hue='HeartDisease', data=df)
    plt.ylabel('Liczba obserwacji')
    plt.title(f'Wykres słupkowy dla {var} w zależności od HeartDisease')
    plt.show()

# Kopia ramki do wykresow znormalizowanych
df_copy_for_cat = df.copy()

# Prezentacja danych w postaci wykresu słupkowego zestawionego znormalizowanego
############################################### Age ############################################

# Znormalizowany i zestawiony wkres dla zmiennej liczbowej - Age

df_copy_for_cat['age_cat'] = pd.cut(df_copy_for_cat['Age'], bins=[0, 40, 50, 60, 101],
                                    labels=['<=40', '41-50', '51-60', '61+'])
data = df_copy_for_cat.groupby(['age_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data_age = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
normalized_data_age.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla Age w zależności od HeartDisease')
plt.xlabel('Kategoria wiekowa')
plt.ylabel('Procent przypadków')
plt.show()

# WNIOSEK: Wiek ma wpływ na występowanie choroby serca. Im starsza osoba, tym większe ryzyko wystąpienia choroby serca.

########################################## RestingBP ##########################################

# Znormalizowany i zestawiony wkres dla zmiennej liczbowej - RestingBP

df_copy_for_cat['resting_cat'] = pd.cut(df_copy_for_cat['RestingBP'], bins=4)
data = df_copy_for_cat.groupby(['resting_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data_resting = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(12, 12))
normalized_data_resting.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla RestingBP w zależności od HeartDisease')
plt.xlabel('Kategoria ciśnienia krwi')
plt.ylabel('Procent przypadków')
plt.show()

########################################## Cholesterol ########################################

# Znormalizowany i zestawiony wkres dla zmiennej liczbowej - Cholesterol

df_copy_for_cat['cholesterol_cat'] = pd.cut(df_copy_for_cat['Cholesterol'], bins=4)
data = df_copy_for_cat.groupby(['cholesterol_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data_cholesterol = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(12, 12))
normalized_data_cholesterol.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla Cholesterol w zależności od HeartDisease')
plt.xlabel('Kategoria cholesterolu')
plt.ylabel('Procent przypadków')
plt.show()

########################################## MaxHR ##########################################

# Znormalizowany i zestawiony wkres dla zmiennej liczbowej - MaxHR

df_copy_for_cat['max_hr_cat'] = pd.cut(df_copy_for_cat['MaxHR'], bins=4)
data = df_copy_for_cat.groupby(['max_hr_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data_max_hr = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(12, 12))
normalized_data_max_hr.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla MaxHR w zależności od HeartDisease')
plt.xlabel('Kategoria maksualnego tętna')
plt.ylabel('Procent przypadków')
plt.show()

# WNIOSEK: Im wyższe maksymalne tętno, tym mniejsze prawdopodobieństwo wystąpienia choroby serca

########################################## Oldpeak ##########################################

# Znormalizowany i zestawiony wkres dla zmiennej liczbowej - Oldpeak

df_copy_for_cat['oldpeak_cat'] = pd.cut(df_copy_for_cat['Oldpeak'], bins=4)
data = df_copy_for_cat.groupby(['oldpeak_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data_oldpeak = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(12, 12))
normalized_data_oldpeak.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla Oldpeak w zależności od HeartDisease')
plt.xlabel('Kategoria spadku ST')
plt.ylabel('Procent przypadków')
plt.show()

###############################################################################################

print()

# Podzial chorzy / zdrowi
print(df['HeartDisease'].value_counts(normalize=True).sort_index())  # 0 - 45%, 1 - 55%

# Kopia ramki danych - na potrzeby prezentaji Macierzy Korelacji Pearsona
df_copy = df.copy()

# Zamiana zmiennych kategorycznych na odpowiednie kategorie
df_copy.Sex = df_copy.Sex.replace({'M': 0, 'F': 1}).astype(np.uint8)
df_copy.ChestPainType = df_copy.ChestPainType.replace({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}).astype(np.uint8)
df_copy.RestingECG = df_copy.RestingECG.replace({'Normal': 0, 'ST': 1, 'LVH': 2}).astype(np.uint8)
df_copy.ExerciseAngina = df_copy.ExerciseAngina.replace({'Y': 0, 'N': 1}).astype(np.uint8)
df_copy.ST_Slope = df_copy.ST_Slope.replace({'Up': 0, 'Flat': 1, 'Down': 2}).astype(np.uint8)

# Sprawdzamy wszystkie zmienne
print()
print(df_copy.info())

correlation1 = df_copy
correlation2 = df_copy.drop(columns=['ChestPainType', 'RestingECG', 'ST_Slope', 'ExerciseAngina'])

# Tworzenie i wypisanie macierzy korelacji Pearsona
# Zauwazmy, ze najwiekszy wspolczynnik korelacji mają zmienne: ST_Slope, ChestPainType, Oldpeak, Age, FastingBS.
corr_matrix_pearson = correlation1.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Macierz Korelacji Pearsona")
plt.show()

# Drugi wariant macierzy korelacji Pearsona
corr_matrix_pearson = correlation2.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Macierz Korelacji Pearsona")
plt.show()

############################################################
# ---------- MLP - Perceptron wielowarstwowy ---------------
############################################################

"""
Podzial zbioru na uczacy i testowy
Y - zmienna celu (HeartDisease)
X - jako predyktory na poczatku przyjmujemy pozostale kolumny
Na obecnym etapie do naszego modelu wybieramy wszystkie kolumny, poniewaz na podstawie wstepnej analizy
mozemy stwierdzic, ze wszystkie zmienne maja wplyw na zmienna celu.
"""

# Kodowanie OneHot - zrobienie zmiennych kategorycznych na numeryczne
df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
df = df.astype(int)

destination_variable = df['HeartDisease']  # Y
# data_without_destination_variable = df.drop(columns=['HeartDisease'])  # X - pierwsze przejscie algorytmu
data_without_destination_variable = df[['FastingBS', 'Sex_F', 'Sex_M', 'ChestPainType_ASY',
                                        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
                                        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
                                        'ST_Slope_Down', 'ST_Slope_Flat',
                                        'ST_Slope_Up']]  # X - Drugie przejscie algorytmu

# Próby różnych kombinacji zmiennych do wyboru X
# data_without_destination_variable = df[['ExerciseAngina', 'Sex', 'MaxHR']]  # X
# data_without_destination_variable = df[['ST_Slope', 'ChestPainType', 'Oldpeak']]  # X
# data_without_destination_variable = df[
#     ['ST_Slope', 'ChestPainType', 'Oldpeak', 'Age', 'FastingBS', 'Sex', 'MaxHR', 'ExerciseAngina', 'RestingBP',
#      'RestingECG']]  # X


# Podzial zbioru na uczacy i testowy
X_train, X_test, y_train, y_test = train_test_split(data_without_destination_variable, destination_variable,
                                                    test_size=0.3, random_state=294858, stratify=destination_variable)

# Sprawdzenie czy podzial jest zgodny
print("Sprawdzenie czy podzial jest zgodny: \n")
print("Wielkosc zbioru uczacego:", len(X_train))
print("Wielkosc zbioru testowego:", len(X_test))
print()

# Wypisanie procentowego podzialu klas w zbiorach
print("Procentowy podzial klasie uczącej:")
print(y_train.value_counts(normalize=True))
print()
print("Procentowy podzial klasie testowej:")
print(y_test.value_counts(normalize=True))

# Przygotowanie danych do modelu. Wybór kolumn w zależności od typu danych
sel_num = make_column_selector(dtype_include=['int64', 'float64'])
sel_cat = make_column_selector(dtype_include='object')

print()
print(df.info())
print()

# Sprawdzenie czy wszystkie zmienne są typu liczbowego
print(sel_num(data_without_destination_variable))
print(sel_cat(data_without_destination_variable))

# Tworzenie preprocesora do zmiany zmiennych kategorycznych na liczbowe i skalowania zmiennych liczbowych
preprocesor = ColumnTransformer(transformers=
                                [('num', MinMaxScaler(feature_range=(-1, 1)), sel_num),
                                 ('cat', OneHotEncoder(handle_unknown='ignore'), sel_cat)]
                                )

# Budowa potoku - modelu MLP
potok1 = Pipeline(steps=[('prep', preprocesor),
                         ('siec', MLPClassifier(random_state=294858))
                         ]
                  )

# Parametry do strojenia modelu MLP
potok1.fit(X_train, y_train)

# Ocena modelu - sprawdzenie trefnośći sieci na zbiorze uczącym i testowym
print()
print("Trafnosc sieci na zbiorze uczącym:")
print(potok1.score(X_train, y_train))
print()
print("Trafnosc sieci na zbiorze testowym:")
print(potok1.score(X_test, y_test))
print()

# Wykorzestanie metody 'predict' do przewidzenia klas dla zbioru uczącego i testowego
y_train_pred = potok1.predict(X_train)
y_test_pred = potok1.predict(X_test)

# Wypisanie macierzy pomyłek dla zbioru uczącego i testowego
print('Zbiór uczący:')
print(pd.DataFrame(confusion_matrix(y_train, y_train_pred)))
print()
print('Zbiór testowy:')
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred)))
print()

# Wypisanie raportu klasyfikacji dla zbioru uczącego i testowego
print('Wypisanie raportu klasyfikacji dla zbioru uczącego:')
ocen_model_klasyfikacji(y_train, y_train_pred)
print()
print('Wypisanie raportu klasyfikacji dla zbioru testowego:')
ocen_model_klasyfikacji(y_test, y_test_pred)
print()

print('Wypisanie informacji o sieci:')
# Metoda do wypisania informacji o sieci
info_o_sieci(potok1, 'siec')

# Wykorzystanie metody 'predict_proba' do przewidzenia prawdopodobieństwa przynależności do danej klasy
y_train_score = potok1.predict_proba(X_train)[:, 1]
y_test_score = potok1.predict_proba(X_test)[:, 1]

# Metoda do rysowania krzywej ROC wraz z obliczeniem AUC
ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)

# Entropia krzyżowa
liczba_epok = len(potok1.named_steps['siec'].loss_curve_)
plt.plot(range(1, liczba_epok + 1), potok1.named_steps['siec'].loss_curve_)
plt.title('Entropia krzyżowa - MLP')
plt.xlabel('Epoki')
plt.ylabel('Entropia krzyżowa')
plt.show()


# Metoda do rysowania wykresu ważności predyktorów
def waznosc_predyktorow_mlp(model, X, y):
    # Przeprowadzenie analizy ważności cech
    results = permutation_importance(model, X, y, scoring='accuracy')

    # Przekształcenie wyników na serię danych pandas
    waznosci = pd.Series(results.importances_mean, index=X.columns)

    # Sortowanie ważności cech
    waznosci.sort_values(inplace=True)

    # Rysowanie wykresu
    waznosci.iloc[-10:].plot(kind='barh', figsize=(10, 4))
    plt.title('Ważność predyktorów - MLP')
    plt.xlabel('Ważność')
    plt.ylabel('Predyktory')
    plt.show()


waznosc_predyktorow_mlp(potok1, X_train, y_train)

"""
Najważniejszym predyktorem okazał się zdecydowanie ‚ST_Slope'. Zaraz za nią są dość mocno istotne zmienne 'FastingBS'
oraz 'ExerciseAngina'. Na dalszym planie, ale nadal istotne są zmienne 'ChestPainType' oraz ‚ RestingECG'.

Zbudujemy teraz model oparty na innym zbiorze predyktorów. Po analizie wykresu ważności predyktorów oraz macierzy 
koraelacji Pearsona wybraliśmy za nowe predyktory dla modelu następujące zmienne zmienne:

- ST_Slope
- ChestPainType
- Sex
- FastingBS
- RestingECG

WNIOSEK PO ANALIZIE: Oba modele są dość dobre. Ogranicznenie liczby predyktorów nie wpłynęło znacząco na wyniki. Jedynie
na zbiorze uczącym trafność spadła o 0.05. Warto zauważyć, że w przypadku modelu z ograniczoną liczbą predyktorów, 
wartości czułości i specyficzności są bardziej zbliżone do siebie, co może świadczyć o mniejszej skłonności modelu do 
przewidywania jednej z klas. Wartości pozostałych miar są bardzo zbliżone do siebie. 
"""

############################################################
# ---------------------- Lasy losowe ----------------------
############################################################

print()
print('Lasy losowe')
print()

# Inicjalizacja modelu lasów losowych
forest1 = RandomForestClassifier(n_estimators=100, random_state=294858, oob_score=True)

# Trenowanie modelu
forest1.fit(X_train, y_train)

# Przewidywanie na danych testowych
print('Zbiór uczący:')
y_train_pred = forest1.predict(X_train)
print(pd.DataFrame(confusion_matrix(y_train, y_train_pred)))
print()

print('Zbiór testowy:')
y_test_pred = forest1.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred)))
print()

# Ocena modelu
print()
print('Zbiór uczący - forest1: ', accuracy_score(y_train, y_train_pred))
print('Zbiór testowy- forest1: ', accuracy_score(y_test, y_test_pred))
print()

# Sprawdzamy, jakie jest prawdopodobieństwo poprawnej klasyfikacji przy użyciu danych out-of-bag
print('Prawdopodobieństwo poprawnej klasyfikacji przy użyciu danych out-of-bag: ')
print(forest1.oob_score_)
print()
print('Wypisanie informacji o modelu:')
print(forest1.estimator_)
print()


# Sprawdzamy, rozmiar drzewa
def tree_size(drzewo):
    print('Liczba wszystkich węzłów: ', drzewo.tree_.node_count)
    print('Liczba liści: ', drzewo.get_n_leaves())
    print('Głębokość: ', drzewo.get_depth())
    print()


print('Parametry drzewa: ')
tree_size(forest1.estimators_[0])
print()

# Ograniamy maksymalną głębokość drzewa
hyperparameters = {'max_depth': range(5, 10), 'min_samples_split': [20, 50, 100]}

# Inicjalizacja modelu lasów losowych z ograniczoną głębokością drzewa
forest2 = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=294858, oob_score=True), hyperparameters,
                       n_jobs=-1)

# Trenowanie modelu w wersji z ograniczoną głębokością drzewa
forest2.fit(X_train, y_train)

print("Hipermarametry: ")
print(forest2.best_params_)
print()

# Optymalne parametry
print("Parametry po ograniczeniu: ")
tree_size(forest2.best_estimator_.estimators_[0])

# Przewidywanie na danych testowych
print('Trafnosc na zbiorze uczącym:')
y_train_pred = forest2.predict(X_train)
print(accuracy_score(y_train, y_train_pred))
print()

print('Trafnosc na zbiorze testowym:')
y_test_pred = forest2.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
print()

# Ocena modelu
print('Zbiór uczący - forest2:')
ocen_model_klasyfikacji(y_train, y_train_pred)
print()

print('Zbiór testowy - forest2:')
ocen_model_klasyfikacji(y_test, y_test_pred)
print()

# Rysowanie krzywej ROC
y_train_score = forest2.predict_proba(X_train)[:, 1]
y_test_score = forest2.predict_proba(X_test)[:, 1]

# Metoda do rysowania krzywej ROC wraz z obliczeniem AUC
ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)


def waznosc_predyktorow(drzewo):
    waznosci = pd.Series(drzewo.feature_importances_, index=X_train.columns)
    waznosci.sort_values(inplace=True)
    waznosci.iloc[-10:].plot(kind='barh', figsize=(6, 4))
    plt.title('Ważność predyktorów - lasy losowe (RF)')
    plt.xlabel('Ważność')
    plt.ylabel('Predyktory')
    plt.show()


waznosc_predyktorow(forest2.best_estimator_)


# Próby przycinania drzew
# Metoda wyznaczania lasow losowych
forest3 = RandomForestClassifier(ccp_alpha=0.001, random_state=294858, n_jobs=-1)
forest3.fit(X_train, y_train)

tree_size(forest3.estimators_[0])

y_train_pred = forest3.predict(X_train)
y_test_pred = forest3.predict(X_test)

print('Zbiór uczący - forest3: \n')
ocen_model_klasyfikacji(y_train, y_train_pred)
print('\nZbiór testowy - forest3: \n')
ocen_model_klasyfikacji(y_test, y_test_pred)

# Rysowanie krzywej ROC
y_train_score = forest3.predict_proba(X_train)[:, 1]
y_test_score = forest3.predict_proba(X_test)[:, 1]

# Metoda do rysowania krzywej ROC wraz z obliczeniem AUC
ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)
