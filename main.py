import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Wczytaj dane z pliku CSV
df = pd.read_csv('heart.csv')

# Wyświetl kilka pierwszych wierszy danych, aby zobaczyć, jak wyglądają
df.head()

# Sprawdź informacje o danych, takie jak typy zmiennych, brakujące wartości itp.
df.info()

# Zamiana danych w kolumnie Cholesterol, tzn. wartosci 0 zastapione mediana
median_chol = df['Cholesterol'].median()
mean_chol = df['Cholesterol'].mean()
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

# Liczba mezczyzn z choroba serca

# Liczba mężczyzn z chorobą serca
number_of_men_with_heart_disease = df[(df['Sex'] == 'F') & (df['HeartDisease'] == 1)].shape[0]
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

rozklad = df['HeartDisease'].value_counts(normalize=True).sort_index()
print(rozklad)  # 0 - 45%, 1 - 55%

#
destination_variable = df['HeartDisease']

data_without_destination_variable = df.drop(columns=['HeartDisease', 'age_cat'])

X_encoded = pd.get_dummies(data_without_destination_variable, drop_first=True)
# Podzial zbioru na uczacy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_encoded, destination_variable,
                                                    test_size=0.3, random_state=294858)

print(len(X_train), len(X_test))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

sel_num = make_column_selector(dtype_include=['int64', 'float64'])
sel_cat = make_column_selector(dtype_include='object')

print(sel_num(data_without_destination_variable))
print(sel_cat(data_without_destination_variable))

preprocesor = ColumnTransformer(transformers=
                                [('num', MinMaxScaler(feature_range=(-1, 1)), sel_num),
                                 ('cat', OneHotEncoder(handle_unknown='ignore'), sel_cat)]
                                )

# Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Uczenie modelu MLP
mlp = MLPClassifier(random_state=294858)
mlp.fit(X_train_scaled, y_train)

# Generowanie wykresów Partial Dependence
features_to_plot = [3,4,5]  # Indeksy dla trzech pierwszych cech po kodowaniu
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(mlp, X_train_scaled, features_to_plot, feature_names=X_train.columns, ax=ax, grid_resolution=20)
plt.show()

# print(pd.DataFrame(tmp, columns=preprocesor.get_feature_names_out()))

# Budowa modelu
potok1 = Pipeline(steps=[('prep', preprocesor),
                         ('siec', MLPClassifier(random_state=294858))
                         ]
                  )

potok1.fit(X_train, y_train)

# Ocena modelu
print(potok1.score(X_train, y_train))
print(potok1.score(X_test, y_test))

y_train_pred = potok1.predict(X_train)
y_test_pred = potok1.predict(X_test)
print('Zbiór uczący\n')
print(pd.DataFrame(confusion_matrix(y_train, y_train_pred)))
print('\n')
print('Zbiór testowy\n')
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred)))


def ocen_model_klasyfikacji(y_true, y_pred, digits=2):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    overall_error_rate = 1 - accuracy
    sensitivity = tp / (fn + tp)
    fnr = fn / (fn + tp)
    specificity = tn / (tn + fp)
    fpr = fp / (tn + fp)
    precision = tp / (fp + tp)
    f1 = (2 * sensitivity * precision) / (sensitivity + precision)
    print('Trafność: ', round(accuracy, digits))
    print('Całkowity współczynnik błędu', round(overall_error_rate, digits))
    print('Czułość: ', round(sensitivity, digits))
    print('Wskaźnik fałszywie negatywnych: ', round(fnr, digits))
    print('Specyficzność: ', round(specificity, digits))
    print('Wskaźnik fałszywie pozytywnych: ', round(fpr, digits))
    print('Precyzja: ', round(precision, digits))
    print('Wynik F1: ', round(f1, digits))


print('Zbiór uczący')
ocen_model_klasyfikacji(y_train, y_train_pred)
print()
print('Zbiór testowy')
ocen_model_klasyfikacji(y_test, y_test_pred)


def info_o_sieci(potok, krok):
    print('Liczba warstw: ', potok.named_steps[krok].n_layers_)
    print('Liczba neuronów w warstwie wejściowej: ', potok.named_steps[krok].n_features_in_)
    print('Liczba neuronów w warstwach ukrytych: ', potok.named_steps[krok].hidden_layer_sizes)
    print('Funkcja aktywacji w warstwach ukrytych : ', potok.named_steps[krok].activation)
    print('Liczba neuronów w warstwie wyjściowej: ', potok.named_steps[krok].n_outputs_)
    print('Funkcja aktywacji w warstwie wyjściowej : ', potok.named_steps[krok].out_activation_)


info_o_sieci(potok1, 'siec')


def ROC(y_true, y_score, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specyficzność')
    plt.ylabel('Czułość')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.show()


y_train_score = potok1.predict_proba(X_train)[:, 1]
y_test_score = potok1.predict_proba(X_test)[:, 1]

ROC(y_train, y_train_score, 1)
ROC(y_test, y_test_score, 1)

# Entropia krzyżowa
liczba_epok = len(potok1.named_steps['siec'].loss_curve_)
plt.plot(range(1,liczba_epok+1), potok1.named_steps['siec'].loss_curve_)
plt.xlabel('Epoki')
plt.ylabel('Entropia krzyżowa')
plt.show()
