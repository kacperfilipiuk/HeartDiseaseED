 # Projekt Prognozowania Chorób Serca

## Założenia projektu
Projekt był pomysłem na otrzymanie oceny cząstkowej z przedmiotu Eksploracja Danych. Przedstawiony poniższy opis zotał sformatowany na potrzeby przedstawienia problemu użytkownikowi, sposobu dedukcji i dojścia do rozwiązania problemu.

## Opis Danych
Plik `heart.csv` zawiera dane pacjentów kardiologicznych. Celem projektu jest budowa modelu przewidującego wystąpienie choroby serca u pacjenta. Zbiór danych zawiera 918 rekordów.

## Specyfikacja Zmiennych
Zmienną celu jest `HeartDisease`, która przyjmuje jedną z dwóch wartości: brak (0) lub obecność (1) choroby serca. Opis predyktorów jest dostępny w pliku `opis_heart.docx`.

## Instrukcje do Zadania

### Wczytywanie Danych
1. Wczytaj dane z pliku `heart.csv`.
2. Wyspecyfikuj zmienne, ustawiając odpowiednie poziomy pomiaru.

### Przygotowanie Danych
1. Użyj średniej arytmetycznej z numerów indeksów osób tworzących daną grupę jako ziarna generatora liczb losowych.
2. Zaokrąglij wynik w dół do najbliższej liczby całkowitej.
3. Podziel dane losowo na zbiory uczący i testowy.

### Analiza Eksploracyjna Danych (EDA)
1. Przeprowadź eksploracyjną analizę danych.
2. Zdecyduj, które zmienne będą używane jako predyktory w modelu.
3. Wyjaśnij wybór predyktorów i ewentualne rezygnacje.

### Budowa Modeli
1. Zbuduj model klasyfikacyjny za pomocą metody MLP.
2. Zbuduj drugi model klasyfikacyjny, wybierając jedną z metod: CART, C4.5/C5.0, las losowy, Extra Trees, XGBoost.
3. Dobierz odpowiednie hiperparametry dla obu modeli.

### Ocena Modeli
1. Oceń jakość modeli na zbiorach uczącym i testowym, obliczając współczynniki takie jak trafność, czułość, swoistość i współczynnik F1.
2. Narysuj krzywe ROC i oblicz pole pod nimi.
3. Zanalizuj wyniki i omów ewentualne problemy.
4. W przypadku przeuczenia, zbuduj poprawiony model.
5. Porównaj modele i zdecyduj, który jest lepszy.
