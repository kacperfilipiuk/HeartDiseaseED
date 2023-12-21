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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
# Pliki w projekcie
from ROC_curve import ROC
from model_evaluation import ocen_model_klasyfikacji
from neural_network_information import info_o_sieci

########################################################################################################

# Informacje na początku zadania

# Zbiór danych i unajamey jako poprawny
# Zauważono, że w kolumnie Cholesterol występują wartości 0, które nie są poprawne (prawdopodobnie).
# Dlatego zastąpiono je medianą na potrzeby szkolenia modelu i kompletowania danych.

########################################################################################################

# Wczytaj dane z pliku CSV
df = pd.read_csv('heart.csv')

# Wyświetl kilka pierwszych wierszy danych, aby zobaczyć, jak wyglądają
print(df.head())

# Sprawdź informacje o danych, takie jak typy zmiennych, brakujące wartości itp.
print(df.info())

# Zamiana danych w kolumnie Cholesterol, tzn. wartosci 0 zastapione mediana
median_chol = df['Cholesterol'].median()
df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)

# Wykresy pudełkowe dla zmiennych numerycznych w zależności od HeartDisease
num_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for var in num_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='HeartDisease', y=var, data=df)
    plt.title(f'Wykres pudełkowy dla {var} w zależności od HeartDisease')
    plt.xticks([0, 1], ['No', 'Yes'])  # Zmiana etykiet osi X
    plt.show()

# Wykresy słupkowe dla zmiennych kategorycznych w zależności od HeartDisease
cat_vars = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for var in cat_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=var, hue='HeartDisease', data=df)
    plt.title(f'Wykres słupkowy dla {var} w zależności od HeartDisease')
    plt.show()

# Liczba mężczyzn z chorobą serca
number_of_men_with_heart_disease = df[(df['Sex'] == 'F') & (df['HeartDisease'] == 1)].shape[0]
print("Liczba mezczyzn z chorobami serca: ")
print(number_of_men_with_heart_disease)

# Prezentacja danych w postaci wykresu słupkowego zestawionego znormalizowany
# Wykresy słupkowe dla zmiennych kategorycznych w zależności od Age

df['age_cat'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 101],
                       labels=['<=40', '41-50', '51-60', '61+'])
data = df.groupby(['age_cat', 'HeartDisease'], observed=True).size().unstack()

# Normalizacja danych
normalized_data = data.div(data.sum(axis=1), axis=0)

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
normalized_data.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Znormalizowany wykres słupkowy zestawiony dla Age w zależności od HeartDisease')
plt.xlabel('Kategoria wiekowa')
plt.ylabel('Procent przypadków')
plt.show()

# WNIOSEK: Wiek ma wpływ na występowanie choroby serca. Im starsza osoba, tym większe ryzyko wystąpienia choroby serca.
# (wniosek został stworzony na podstawie wykresu słupkowego zestawionego znormalizowanego dla Age w zależności od HeartDisease)

print(df['HeartDisease'].value_counts(normalize=True).sort_index())  # 0 - 45%, 1 - 55%

# Kopia ramki danych
df_copy = df.copy()

# Zamian zmiennych kategorycznych na liczbowe
df_copy.Sex = df_copy.Sex.replace({'M': 0, 'F': 1}).astype(np.uint8)
df_copy.ChestPainType = df_copy.ChestPainType.replace({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}).astype(np.uint8)
df_copy.RestingECG = df_copy.RestingECG.replace({'Normal': 0, 'ST': 1, 'LVH': 2}).astype(np.uint8)
df_copy.ExerciseAngina = df_copy.ExerciseAngina.replace({'Y': 0, 'N': 1}).astype(np.uint8)
df_copy.ST_Slope = df_copy.ST_Slope.replace({'Up': 0, 'Flat': 1, 'Down': 2}).astype(np.uint8)

# Sprawdzamy czy wszystkie zmienne są typu liczbowego
print()
print(df_copy.info())
print(df_copy.describe())

correlation = df_copy.drop(columns=['age_cat'])

# knn = KNeighborsClassifier(3)
# knn.fit(X_train, y_train)
# print("-------------------")
# # Jezeli wynik na zbiorze testowym jest rowny 1 to znaczy, ze model jest zbyt dobry i prawdopodobnie jest overfitted
# # Jezeli wynik na zbiorze testowym jest mniejszy niż na zbiorze treningowym to znaczy, ze model jest zbyt słąby, miał
# # za mało danych i prawdopodobnie jest underfitted
# print(knn.score(X_train, y_train))
# print(knn.score(X_test, y_test))
# print("-------------------")

# Tworzenie i wypisanie macierzy korelacji Pearsona
corr_matrix_pearson = correlation.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Macierz Korelacji Pearsona")
plt.show()

############################################################
# ---------- MLP - Perceptron wielowarstwowy ---------------
############################################################

# Podzial zbioru na uczacy i testowy
destination_variable = df['HeartDisease']  # Y
data_without_destination_variable = df.drop(columns=['HeartDisease', 'age_cat'])  # X - wszystkie
# data_without_destination_variable = df[['ST_Slope', 'ChestPainType', 'Oldpeak', 'Age', 'Sex', 'MaxHR']]  # X - negatywne i pozytywne
# data_without_destination_variable = df[['ExerciseAngina', 'Sex', 'MaxHR']]  # X - najbardziej negatywne
# data_without_destination_variable = df[['ST_Slope', 'ChestPainType', 'Oldpeak']]  # X - najbardziej pozytywne | Miejsce III
# data_without_destination_variable = df[
#     ['ST_Slope', 'ChestPainType', 'Oldpeak', 'Age', 'FastingBS', 'Sex', 'MaxHR', 'ExerciseAngina', 'RestingBP',
#      'RestingECG']]  # X - pozytywne i negatywne, któtych wartości korelacji są znacząco wyższe

# Podzial zbioru na uczacy i testowy
X_train, X_test, y_train, y_test = train_test_split(data_without_destination_variable, destination_variable,
                                                    test_size=0.3, random_state=294858, stratify=destination_variable)

# Sprawdzenie czy podzial jest zgodny
print(len(X_train), len(X_test))
print()

# Wypisanie procentowego podzialu klas w zbiorach
print("Procentowy podzial klasie uczącej: \n")
print(y_train.value_counts(normalize=True))
print()
print("Procentowy podzial klasie testowej: \n")
print(y_test.value_counts(normalize=True))

# Przygotowanie danych do modelu. Wybór kolumn w zależności od typu danych
sel_num = make_column_selector(dtype_include=['int64', 'float64'])
sel_cat = make_column_selector(dtype_include='object')

print(df.info())

# Sprawdzenie czy wszystkie zmienne są typu liczbowego
print(sel_num(data_without_destination_variable))
print(sel_cat(data_without_destination_variable))

# Tworzenie preprocesora do zmiany zmiennych kategorycznych na liczbowe i skalowania zmiennych liczbowych
preprocesor = ColumnTransformer(transformers=
                                [('num', MinMaxScaler(feature_range=(-1, 1)), sel_num),
                                 ('cat', OneHotEncoder(handle_unknown='ignore'), sel_cat)]
                                )

# Sprawdzenie jak wyglądają dane wykorzystywane do modelu
tmp = preprocesor.fit_transform(X_train)
print()
print("Przetworzone dane: ")
print(pd.DataFrame(tmp, columns=preprocesor.get_feature_names_out()))

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
print("Trafnosc sieci na zbiorze uczącym:")
print(potok1.score(X_test, y_test))
print()

# Wykorzestanie metody 'predict' do przewidzenia klas dla zbioru uczącego i testowego
y_train_pred = potok1.predict(X_train)
y_test_pred = potok1.predict(X_test)

# Wypisanie macierzy pomyłek dla zbioru uczącego i testowego
print('Zbiór uczący\n')
print(pd.DataFrame(confusion_matrix(y_train, y_train_pred)))
print()
print('Zbiór testowy\n')
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred)))
print()

# Wypisanie raportu klasyfikacji dla zbioru uczącego i testowego
print('Zbiór uczący')
ocen_model_klasyfikacji(y_train, y_train_pred)
print()
print('Zbiór testowy')
ocen_model_klasyfikacji(y_test, y_test_pred)

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
plt.xlabel('Epoki')
plt.ylabel('Entropia krzyżowa')
plt.show()

############################################################
# ---------------------- Lasy losowe ----------------------
############################################################

print()
print('Lasy losowe')
print()
# Zakodowanie zmiennych kategorycznych
for column in df.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# Podzielenie danych na cechy i etykietę

data_without_destination_variable = df.drop(columns=['HeartDisease', 'age_cat'])  # X

destination_variable = df['HeartDisease']  # Dane są odpowiednie przygotowane do modelu - nie trzeba ich już przetwarzać

# Podział na zbiór uczący i testowy

X_train, X_test, y_train, y_test = train_test_split(data_without_destination_variable, destination_variable,
                                                    test_size=0.3, random_state=294858, stratify=destination_variable)
# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicjalizacja modelu lasów losowych
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=294858, oob_score=True)

# Trenowanie modelu
rf_classifier.fit(X_train_scaled, y_train)

# Przewidywanie na danych testowych
print('Zbiór uczący: \n')
y_train_pred = rf_classifier.predict(X_train_scaled)
print(pd.DataFrame(confusion_matrix(y_train, y_train_pred)))

print('Zbiór testowy: \n')
y_test_pred = rf_classifier.predict(X_test_scaled)
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred)))

# Ocena modelu
print('Zbiór uczący: ', accuracy_score(y_train, y_train_pred))
print('Zbiór testowy: ', accuracy_score(y_test, y_test_pred))

print(rf_classifier.oob_score_)
print(rf_classifier.estimator_)


# Sprawdzamy, rozmiar drzewa
def tree_size(drzewo):
    print('Liczba wszystkich węzłów: ', drzewo.tree_.node_count)
    print('Liczba liści: ', drzewo.get_n_leaves())
    print('Głębokość: ', drzewo.get_depth())


print(tree_size(rf_classifier.estimators_[0]))

# Ograniamy maksymalną głębokość drzewa
hyperparameters = {'max_depth': range(5, 10), 'min_samples_split': [20, 50, 100]}
forest2 = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=294858, oob_score=True), hyperparameters,
                       n_jobs=-1)
forest2.fit(X_train, y_train)
print(forest2.best_params_)
# Optymalne parametry: {'max_depth': 6, 'min_samples_split': 20}

# Przewidywanie na danych testowych
print('Zbiór uczący: \n')
y_train_pred = forest2.predict(X_train_scaled)
print(accuracy_score(y_train, y_train_pred))

print('Zbiór testowy: \n')
y_test_pred = forest2.predict(X_test_scaled)
print(accuracy_score(y_test, y_test_pred))

print('Zbiór uczący: \n')
ocen_model_klasyfikacji(y_train, y_train_pred)
print('\nZbiór testowy: \n')
ocen_model_klasyfikacji(y_test, y_test_pred)

# Rysowanie krzywej ROC
y_train_score = forest2.predict_proba(X_train_scaled)[:, 1]
y_test_score = forest2.predict_proba(X_test_scaled)[:, 1]

# Metoda do rysowania krzywej ROC wraz z obliczeniem AUC
ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)


# Jest problem z przeniesieniem metody do zewnetrznego modulu

def waznosc_predyktorow(drzewo):
    waznosci = pd.Series(drzewo.feature_importances_, index=X_train.columns)
    waznosci.sort_values(inplace=True)
    waznosci.iloc[-10:].plot(kind='barh', figsize=(6, 4))
    plt.xlabel('Ważność predyktorów')
    plt.ylabel('Predyktory')
    plt.show()


waznosc_predyktorow(forest2.best_estimator_)

# Metoda wyznaczania lasow losowych - przycinanie drzew
forest3 = RandomForestClassifier(ccp_alpha=0.001, random_state=294858, n_jobs=-1)
forest3.fit(X_train, y_train)

print(tree_size(forest3.estimators_[0]))

y_train_pred = forest3.predict(X_train_scaled)
y_test_pred = forest3.predict(X_test_scaled)

print('Zbiór uczący: \n')
ocen_model_klasyfikacji(y_train, y_train_pred)
print('\nZbiór testowy: \n')
ocen_model_klasyfikacji(y_test, y_test_pred)

# Rysowanie krzywej ROC
y_train_score = forest3.predict_proba(X_train_scaled)[:, 1]
y_test_score = forest3.predict_proba(X_test_scaled)[:, 1]

# Metoda do rysowania krzywej ROC wraz z obliczeniem AUC
ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)
