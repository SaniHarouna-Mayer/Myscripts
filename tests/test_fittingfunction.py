from myscripts.fittingfunction import *
from myscripts.fittingclass import GenConfig, ConConfig
from tempfile import TemporaryDirectory
import os
import pandas as pd


def test_fit_ni():
    ni = GenConfig(name="Ni",
                   stru_file="Ni.cif",
                   ncpu=4)

    config = ConConfig(name="crystal",
                       data_id=0,
                       data_file="Ni.gr",
                       fit_range=(1.5, 10., 0.01),
                       qparams=(0.04, 0.02),
                       eq="Ni",
                       phases=ni)

    recipe = make(config)

    sgconstrain(recipe, "Ni")
    free_and_fit(recipe, 'all')

    with TemporaryDirectory(dir="./") as temp_folder:
        csv_db = os.path.join(temp_folder, f"csv.csv")
        fgr_db = os.path.join(temp_folder, f"fgr.csv")
        cif_db = os.path.join(temp_folder, f"cif.csv")
        save = gen_save_all(temp_folder, csv_db, fgr_db, cif_db)
        save(recipe)

        print(pd.read_csv(csv_db).to_string())
        print(pd.read_csv(fgr_db).to_string())
        print(pd.read_csv(cif_db).to_string())

    return


if __name__ == "__main__":
    test_fit_ni()
