# Résolution de Sudoku

Ce projet implémente un programme de résolution de Sudoku en Python, utilisant différents algorithmes de recherche. Ces algorithmes explorent les possibilités pour remplir une grille de Sudoku tout en respectant ses règles.

## Contenu

Le programme propose quatre algorithmes de recherche :

1. **DFS (Depth-First Search)** : Explore les nœuds les plus profonds en premier. Approche exhaustive mais lente.
2. **BFS (Breadth-First Search)** : Explore les nœuds en largeur. Plus adapté aux petits problèmes.
3. **A*** : Combine le coût du chemin parcouru (​g(n)​) et une estimation du coût restant (​h(n)​) pour guider la recherche de manière efficace.
4. **Greedy Best-First Search (GBFS)** : Basé uniquement sur l'heuristique pour accélérer l'exploration des solutions potentielles.

## Utilisation

### Prérequis

- Python 3.x installé sur votre machine.
- Les bibliothèques nécessaires : 'copy', 'heapq', 'time'.

### Lancer le programme

Pour exécuter le programme, ouvrez le fichier principal et choisissez l'algorithme souhaité dans la section `if __name__ == "__main__"`.

Exemple :

```python
if __name__ == "__main__":
    # Choisissez l'algorithme ici
    solve_sudoku("A*")
```

Les options disponibles pour la fonction `solve_sudoku()` sont :

- "DFS"
- "BFS"
- "A*"
- "Greedy"

Modifiez cette ligne pour exécuter l'algorithme souhaité.

### Exemple de configuration

La configuration d'une grille de Sudoku initiale se fait dans le code, en remplissant une matrice 9x9. Voici un exemple :

```python
grid = [
    [2, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 7, 0, 9, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 9, 0],
    [9, 0, 0, 0, 5, 0, 0, 0, 2],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 2, 0],
    [4, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 5, 0, 2, 0, 0, 0, 0, 7]
]
```

### Exécution

Une fois l'algorithme choisi et la grille initiale configurée, exécutez le script avec la commande :

```bash
python sudoku.py
```

Le programme affichera la solution trouvée et les statistiques de performance, comme le temps d'exécution et le nombre de nœuds explorés.

## Notes

- **Heuristique pour A*** : L'heuristique utilisée dans A* est basée sur le nombre de cases encore à remplir et leur faisabilité. Elle est essentielle pour guider efficacement la recherche.
- **Adaptation possible** : Vous pouvez facilement ajouter vos propres grilles pour tester différentes configurations.

## Références

- Wikipédia : [Sudoku](https://fr.wikipedia.org/wiki/Sudoku).
- Rapport complet dans le fichier `rapport.pdf`