from myscripts.fittingfunction import make, fit, save_all, constrainAsSpaceGroup, updated
from myscripts.fittingclass import GenConfig, ConConfig, MyRecipe
from diffpy.srfit.pdf import PDFGenerator
from tempfile import TemporaryDirectory
import os
import pandas as pd


def constrain(recipe: MyRecipe):
    con = recipe.crystal
    gen: PDFGenerator = con.Ni

    recipe.addVar(gen.scale)
    recipe.addVar(gen.delta2)

    sgpars = constrainAsSpaceGroup(gen.phase, 225)
    for par in sgpars.latpars:
        recipe.addVar(par, tag="lat")
    for par in sgpars.adppars:
        recipe.addVar(par, value=0.006, tag="adp")
    return


def refine(recipe: MyRecipe):
    recipe.fix("all")
    recipe.free("scale", "lat")
    fit(recipe)
    recipe.free("adp")
    fit(recipe)
    recipe.free("delta2")
    fit(recipe)
    return


def test_fit_ni():
    ni = GenConfig(name="Ni",
                   stru_file="Ni.cif",
                   ncpu=4)
    config = ConConfig(name="crystal",
                       data_file="Ni.gr",
                       fit_range=(1.5, 30.0001, 0.01),
                       qparams=(0.04, 0.02),
                       eq="Ni",
                       phases=ni)

    recipe = make(config)

    csv_df = pd.DataFrame()
    fgr_df = pd.DataFrame()
    cif_df = pd.DataFrame()

    constrain(recipe)
    refine(recipe)

    with TemporaryDirectory(dir="./") as temp_folder:
        uid = save_all(recipe, temp_folder, "Ni", info={"sample": "Ni", "model": "crystal"})

        csv_file = os.path.join(temp_folder, f"Ni_{uid}.csv")
        fgr_file = os.path.join(temp_folder, f"Ni_{uid}_{config.name}.fgr")
        cif_file = os.path.join(temp_folder, f"Ni_{uid}_{config.name}_{config.phases[0].name}.cif")

        assert os.path.isfile(csv_file)
        assert os.path.isfile(fgr_file)
        assert os.path.isfile(cif_file)

    csv_df, fgr_df, cif_df = updated(recipe, csv_df, fgr_df, cif_df)
    assert not csv_df.empty
    assert not fgr_df.empty
    assert not cif_df.empty
    return


if __name__ == "__main__":
    test_fit_ni()
