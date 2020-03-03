"""An example: fit fcc-nickel using myscripts."""
from myscripts.fittingfunction import *
from myscripts.fittingclass import *
from myscripts.tools import summarize
from tempfile import TemporaryDirectory
import os
import pandas as pd


def fit_ni():
    """
    Fit the fcc nickel data in the 'myscripts/data_files/Ni.gr' using the structure in 'myscripts/data_files/Ni.cif'
    as the model. Save results and their mete data in a temporary directory.
    """
    # configuration for the generator of PDF
    ni = GenConfig(name="Ni",
                   stru_file="../myscripts/data_files/Ni.cif",
                   stru_type="crystal",
                   ncpu=4)
    # configuration of the characteristic function
    f = FunConfig("f", F.sphericalCF, argnames=["r", "psize_Ni"])
    # configuration of the FitContribution
    config = ConConfig(
        name="crystal",  # name for the FitContribution
        data_id=0,  # this is not important, just for identification for the data
        data_file="../myscripts/data_files/Ni.gr",  # path to the data_file
        fit_range=(1.5, 10., 0.01),  # (rmin, rmax, rstep)
        qparams=(0.04, 0.02),  # (qdamp, qbroad)
        eq=f"{f.name} * {ni.name}",
        # names must be the name of configuration, otherwise it will be treated as a variable
        phases=[ni],  # a list of the generator configuration
        functions=[f]  # a list of the characteristic function configuration
    )
    # make FitRecipe
    recipe = make(config)
    # Constrain the characteristic function and add variables to the recipe
    cfconstrain(recipe, "psize_Ni", dv={"psize_Ni": 1000})
    # Constrain the PDF generator by the space group (read automatically from the cif file) and add variables
    sgconstrain(recipe, "Ni")
    # free all the parameters and fit them
    free_and_fit(recipe, 'all')
    # save the results in a temporary folder
    with TemporaryDirectory(dir="../tests") as temp_folder:
        csv_db = os.path.join(temp_folder, f"csv.csv")
        fgr_db = os.path.join(temp_folder, f"fgr.csv")
        cif_db = os.path.join(temp_folder, f"cif.csv")
        # Create three empty csv files as our data base collection and generate a 'save' function linked to the database
        save = gen_save_all(temp_folder, csv_db, fgr_db, cif_db)
        # save the recipe
        save(recipe)
        # let's see what is saved in the database
        csv_df = pd.read_csv(csv_db)
        fgr_df = pd.read_csv(fgr_db)
        cif_df = pd.read_csv(cif_db)
        print(csv_df.to_string())
        print(fgr_df.to_string())
        print(cif_df.to_string())
        # let's merge the information together and summarize the fitting results from this fitting
        df = csv_df.merge(fgr_df, how="left", on="recipe_id").merge(cif_df, how="left", on="con_id")
        summarize(df, printout=True)
    return


if __name__ == "__main__":
    fit_ni()
