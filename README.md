[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bgQosLZL)
# SSN. Lab. 7. Samoorganizacja za pomocą reguły Hebba

Zapoznaj się z zawartością notatnika Jupyter umieszczonego w repozytorium  i wykonaj zawarte w nim ćwiczenia.


Notatnik: [07_hebb.ipynb](https://github.com/IS-UMK/ssn_lab_07/blob/master/07_hebb.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IS-UMK/ssn_lab_07/blob/master/07_hebb.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IS-UMK/ssn_lab_07/master?filepath=07_hebb.ipynb)

---

## Zadanie 7: Ogólny algorytm Hebbowski (GHA)


Zaimplementuj sieć jednowarstwową realizującą rozkład na czynniki główne za pomocą ogólnego algorytmu hebbowsiego (GHA). W implementacji możesz wykorzytsać szablon klasy ``GHA`` umieszczony w pliku [gha.py](gha.py).

**Parametry początkowe** (ustawiane w argumentach konstruktora):
* $N$ liczba epok
* $k$ liczba neuronów (komponentów)
* $\eta$ wspólczynnik uczenia

**Algorytm uczenia GHA** (realizowany w metodzie `fit(X)`)  

1. Zainicjuj macierz wag $\mathbf{W}$ 
2. Powtarzaj $N$ razy:
3. <ul>Dla każdego przypadku $\mathbf{x}$ ze zbioru uczącego wykonaj</ul>
4. <ul><ul> wyznacz aktywacje <br> $\mathbf{y} = \mathbf{W}\mathbf{x}$ </ul></ul>
5. <ul><ul>zaktualizuj wagi <br> $\Delta \mathbf{W}=\eta \left(\mathbf{y} \cdot \mathbf{x}^T-\mathrm{LT}\left[\mathbf{y} \cdot \mathbf{y}^T\right] \mathbf{W}\right)$ </ul></ul>
gdzie operacja $\mathbf{a}\cdot \mathbf{b}^T$ oznacza iloczyn zewnętrzny, a funkcja $\mathrm{LT}[\mathbf{A}]$ zeruje wszystkie elementy macierzy $\mathbf{A}$ znajdujące się powyżej diagonali

Przydatne funkcje:
* iloczyn zewnętrzny można wyznaczyć za pomocą funkcji [numpy.outer(a,b)](https://numpy.org/doc/stable/reference/generated/numpy.outer.html),
* funkcja [numpy.tril(x)](https://numpy.org/doc/stable/reference/generated/numpy.tril.html) zeruje wartości powyżej diagonali macierzy.

Wykorzystaj algorytm GHA do wyznaczenia $k$ czynników głównych dla danych MNIST. W celu skrócenia czasu uczenia wybież do zbioru treningowego 1000 losowych przypadków ze zbioru cyfr.  
* zbuduj model uzywając $k$ neuronwów. Liczbę $k$ możesz dobrać dowolnie ale staraj się wybrać jak najmniejszą liczbę neuronów, która pozwoli później na odtworzenie danych z ,,dostatecznie wysoką jakością'' 
* przetransformuj dane treningowe do $k$ wymiarowej przestrzeni
* zrekonstruuj zbiór treningowy przez transformację odwrotną i zaprezentuj rekonstrukcję kilku przykładowych cyfr w postaci obrazów 28x28
* przedstaw wizualizację wektorów własnych cyfr (wagi $\mathbf{W}$ w postaci obrazów 28x28)
* przedstaw wykres rozrzutu prezentujący rozkład danych MNIST uzyskany po rzutowaniu dla dwóch pierwszych składowych głównych. Wyróżnij klasy różnymi kolorami. 

Dane MNIST zawierają 70K obrazków cyfr o wymiarach 28x28 (784 pikseli) i można je zaimportować w następujący sposób
```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', data_home='./dane/', parser='auto')
X = mnist.data.values
target = mnist.target.values
```

Rozwiązanie w postaci notatnika Jupyter (``.ipynb``) lub skrypt w języku Python (``.py``) umieść w repozytorium GitHub.

## Materiały:
* [Generalized Hebbian algorithm](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm), Wikipedia




