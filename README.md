# Uncertainty Sampling

Spis treści:
1. Etap4 - folder zawierający stan implementacji na etap 4
  - wyniki0 - folder zawierający wyniki dla wstępnego eksperymentu (sprawdzającego funckjonalność metody)
  - metoda.py - plik zawierający podstawową metodę, na której będą wykonywane eksperymenty (przy zmianie różnych parametrów)
2. Etap5 - folder zawierający stan implementacji na etap 5
  - Eksperyment(1, 2, 3) - foldery zawierające pliki do danego numeru eksperymentu
  - eksp(1, 2, 3).py - plik metoda.py skonfigurowany pod dany numer eksperymentu
  - bac(1, 2, ...).npy/.csv - plik wyniku eksperymentu w postaci wartości balanced accuracy score o numerze odpowiadającym próbie w danym eksperymencie 
  - us(1, 2, ...).npy/.csv - plik wyniku eksperymentu w postaci wartości uncertainty score o numerze odpowiadającym próbie w danym eksperymencie 
