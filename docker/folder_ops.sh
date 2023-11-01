#!/bin/bash

if [[ $COPY_GROUPS == *"api"* ]]; then
    echo "Adding 'api' folder"
    cp -r ./${TEMP_FOLDER}/api ./${CODE_FOLDER}/
    cp -r ./${TEMP_FOLDER}/obfuscated ./${CODE_FOLDER}/
fi

if [[ $COPY_GROUPS == *"test"* ]]; then
    echo "Adding 'tests' folder"
    cp -r ./${TEMP_FOLDER}/tests ./${CODE_FOLDER}/
fi

if [[ $INSTALL_TYPE == "lib" ]]; then
    rm -rf ${CODE_FOLDER}/hamacho/
fi
