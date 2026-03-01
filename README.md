# Random Forest From Scratch
*(Implementacja algorytmów Drzew Decyzyjnych oraz Lasów Losowych w czystym Pythonie)*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/Library-NumPy-lightblue.svg)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)
![Subject](https://img.shields.io/badge/Subject-Machine%20Learning-green.svg)

## 📌 O projekcie
Projekt ten obejmuje własną implementację dwóch klasycznych algorytmów uczenia maszynowego w zadaniach klasyfikacji: **Drzewa Decyzyjnego (Decision Tree)** oraz **Lasu Losowego (Random Forest)**. 

Głównym założeniem było stworzenie modeli od podstaw, bez wykorzystania gotowych metod trenowania z bibliotek wysokopoziomowych (takich jak scikit-learn). Zaimplementowane algorytmy poddano testom na rzeczywistych zbiorach danych, a ich wyniki skonfrontowano z rozwiązaniami dostępnymi w bibliotece scikit-learn.

## ⚙️ Kluczowe Funkcjonalności
* **Decision Tree:** Autorska implementacja algorytmu budowy drzewa z obsługą klasyfikacji binarnej i wieloklasowej.
* **Reduced Error Pruning (REP):** Mechanizm przycinania drzewa zapobiegający przeuczeniu modelu. Pozwala na drastyczną redukcję złożoności (nawet o 75% węzłów), poprawiając czytelność i interpretowalność drzewa.
* **Random Forest:** Las losowy wykorzystujący techniki **Bagging** (Bootstrap Aggregating) oraz losowanie podprzestrzeni cech w celu redukcji wariancji predykcji.

## 📊 Eksperymenty i Wyniki

Testy przeprowadzono na zbiorach danych UCI: **Breast Cancer Wisconsin**  oraz **Wine Quality**.

**Wnioski w skrócie:**
* **Przewaga Lasu Losowego:** Algorytm Random Forest konsekwentnie osiąga wyższą stabilność i dokładność na badanych zbiorach (średnio o 4-8 punktów procentowych lepiej niż pojedyncze drzewo).
* **Wpływ przycinania (Pruning):** Metoda REP świetnie sprawdza się w upraszczaniu modelu, jednak w przypadku zbioru Wine Quality, agresywne przycinanie może prowadzić do utraty cennych informacji i spadku dokładności.
* **Zgodność z Scikit-Learn:** Implementacja autorska osiąga zbliżone metryki jakości z rozwiązaniami z biblioteki scikit-learn, ustępując im głównie pod względem optymalizacji czasu obliczeń.


> 📄 **Pełny raport:** Szczegółowy opis eksperymentów oraz pogłębione wnioski znajdują się w pliku:  [**docs/UMA_Dokumentacja_Koncowa.pdf**](docs/UMA_Dokumentacja_Koncowa_Dominik_Bijoch_Michał_Dobrowolski_25Z.pdf).

## 📂 Struktura Repozytorium

### Kody źródłowe
* **`RandomForest.py`** – Implementacja klasy lasu losowego.
* **`DecisionTree.py`** – Implementacja klasy pojedynczego drzewa oraz REP.
* **`preprocessing.py`** – Funkcje pomocnicze do ładowania danych, podziału na zbiory (train/test split) i dyskretyzacji.
* **`compare_models.py`** – Skrypt porównujący skuteczność własnego Drzewa vs własnego Lasu.
* **`compare_with_sklearn.py`** – Benchmark: Autorska implementacja vs sklearn.
* **`tune_hyperparameters.py`** – Skrypt do szukania optymalnych parametrów (Grid Search).

### Pozostałe
* **`docs/`** – Dokumentacja projektowa i sprawozdanie PDF.
* **`requirements.txt`** – wymagane biblioteki.

## 📄 Licencja
Projekt jest udostępniony na licencji MIT. Szczegóły w pliku [LICENSE](LICENSE).

## 👥 Autorzy
* **Dominik Bijoch**
* **Michał Dobrowolski**

Projekt zrealizowany w ramach przedmiotu "Uczenie Maszynowe (UMA)" na Politechnice Warszawskiej (Semestr 25Z).
