import nbformat
from nbclient import NotebookClient
import sys 
from collections import defaultdict

def convert_notebook_to_script(notebook_path: str, script_path: str):
    # Load the notebook
    notebook = nbformat.read(notebook_path, as_version=4)
    tagged_cells = defaultdict(list)

    # Extract code cells and group by tags
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue

        tags = cell.metadata.get('tags', [])
        if not tags:
            continue
        for tag in tags:
            tagged_cells[tag].append(cell.source)

    with open(script_path, 'w') as f:
        for tag, cells in tagged_cells.items():
            if tag == "setup":
                for cell_source in cells:
                    f.write(f"{cell_source}\n\n")
                continue
            f.write(f"async def {tag}():\n")
            for cell_source in cells:
                indented_source = '\n'.join(['    ' + line for line in cell_source.splitlines()])
                f.write(f"{indented_source}\n\n")
            f.write("\n")
        with open("template.txt", "r") as tem:
            f.write(tem.read())



if __name__ == "__main__":
    convert_notebook_to_script("dataset_pipeline.ipynb", "script_notebook.py")