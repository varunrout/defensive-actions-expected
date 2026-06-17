import nbformat as nbf
from pathlib import Path
nb06 = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell("# 06 - Team Defensive Profile Clustering"))
cells.append(nbf.v4.new_code_cell("print('Notebook 06')"))
nb06.cells = cells
nb07 = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell("# 07 - Player Defensive Archetypes"))
cells.append(nbf.v4.new_code_cell("print('Notebook 07')"))
nb07.cells = cells
with open("notebooks/06_team_defensive_clustering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb06, f)
with open("notebooks/07_player_defensive_archetypes.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb07, f)
print("All 7 notebooks created!")
