from myscripts.fittingfunction import *
from myscripts.fittingclass import *
from myscripts.tools import summarize
from tempfile import TemporaryDirectory
import os
import pandas as pd


def test_fit_ni():
    ni = GenConfig(name="Ni",
                   stru_file="../myscripts/data_files/Ni.cif",
                   ncpu=4)
    f = FunConfig("f", F.sphericalCF, argnames=["r", "psize_Ni"])

    config = ConConfig(name="crystal",
                       data_id=0,
                       data_file="../myscripts/data_files/Ni.gr",
                       fit_range=(1.5, 10., 0.01),
                       qparams=(0.04, 0.02),
                       eq="f * Ni",
                       phases=[ni],
                       functions=[f])

    recipe = make(config)
    cfconstrain(recipe, "psize_Ni", dv={"psize_Ni": 1000})
    sgconstrain(recipe, "Ni")
    free_and_fit(recipe, 'all')

    with TemporaryDirectory(dir="/") as temp_folder:
        csv_db = os.path.join(temp_folder, f"csv.csv")
        fgr_db = os.path.join(temp_folder, f"fgr.csv")
        cif_db = os.path.join(temp_folder, f"cif.csv")
        save = gen_save_all(temp_folder, csv_db, fgr_db, cif_db)
        save(recipe)

        csv_df = pd.read_csv(csv_db)
        fgr_df = pd.read_csv(fgr_db)
        cif_df = pd.read_csv(cif_db)

        print(csv_df.to_string())
        print(fgr_df.to_string())
        print(cif_df.to_string())

        df = pd.merge(csv_df, fgr_df, how="left", on="recipe_id")
        summarize(df, printout=True)

    return


if __name__ == "__main__":
    test_fit_ni()
