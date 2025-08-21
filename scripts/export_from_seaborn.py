import argparse
import sys
from pathlib import Path

try:
    import seaborn as sns  # type: ignore
except Exception as e:
    print(
        "Seaborn no está instalado en este intérprete. Instálalo e intenta nuevamente:\n"
        "- conda: conda install seaborn  (o conda env create -f environment.yml; conda activate add-env)\n"
        "- pip  : pip install seaborn\n"
        f"Detalle: {e}"
    )
    sys.exit(1)

def export_from_seaborn(dataset_name: str, destination_path: str = "../datasets/") -> None:
    """
    Exporta un dataset desde seaborn a un archivo CSV.
    """
    # Cargar dataset
    try:
        df = sns.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error al cargar el dataset '{dataset_name}': {e}")
        sys.exit(1)

    # Asegurar directorio destino
    dest_dir = Path(destination_path)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Exportar CSV
    out_path = dest_dir / f"{dataset_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset '{dataset_name}' exportado a {out_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta un dataset de Seaborn a CSV.")
    parser.add_argument("dataset_name", help="Nombre del dataset en seaborn (por ejemplo: titanic, iris, penguins)")
    parser.add_argument("--destination_path", default="../datasets/", help="Ruta destino (por defecto: ../datasets/)")

    args = parser.parse_args()
    export_from_seaborn(args.dataset_name, args.destination_path)