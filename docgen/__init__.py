# Deliberately empty -- do not re-export anything here.
#
# training/ imports docgen.prompt and docgen.astutils, and the Colab notebook
# installs only requirements.txt (torch, transformers -- no typer, no libcst).
# A re-export would drag cli/inserter into every `python -m training.train` run
# and break it with ModuleNotFoundError on a machine that never needed the CLI.
