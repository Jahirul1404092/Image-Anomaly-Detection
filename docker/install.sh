#!/bin/bash

prod_yes=1

if [[ $INSTALL_GROUPS == *"dev"* ]]; then
    echo "Installing dependencies of 'dev' group"
    poetry install --only dev --no-interaction --no-ansi
    prod_yes=0
fi

if [[ $INSTALL_GROUPS = *"test"* ]]; then
    echo "Installing dependencies of 'test' group"
    poetry install --only test --no-interaction --no-ansi
    prod_yes=0
fi

if [[ $INSTALL_GROUPS = *"docs"* ]]; then
    echo "Installing dependencies of 'docs' group"
    poetry install --only docs --no-interaction --no-ansi
    prod_yes=0
fi

if [[ $INSTALL_GROUPS = *"api"* ]]; then
    echo "Installing dependencies of 'api' group"
    poetry install --only api --no-interaction --no-ansi
    prod_yes=0
fi

if [[ $INSTALL_GROUPS = *"api"* || $INSTALL_GROUPS = *"test"* ]]; then
    echo "Removing non-oss libraries that are not needed"
    source $VENV_PATH/bin/activate && pip uninstall -y ansi2html
fi

if [[ $INSTALL_TYPE == "editable" ]]; then
    echo "Installing hamacho in editable mode"
    poetry install --only-root
elif [[ $INSTALL_TYPE == "lib" ]]; then
    echo "Installing hamacho as a Library"
    poetry build
    source $VENV_PATH/bin/activate && pip install --no-dependencies dist/hamacho-*.whl
    rm -rf dist
fi

if [[ $prod_yes -eq "1" ]]; then
    echo "Production environment has been built"
fi
