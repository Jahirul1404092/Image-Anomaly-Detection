#!/bin/bash

# options
opt_arr=("prod" "dev" "api" "test" "api-test" "prod-lib" "prod-api-obfuscated")

function show_usage {
    echo "Build the desired docker image from the options below:"
    for i in ${!opt_arr[@]}; do
        echo "$i) ${opt_arr[$i]}"
    done
    echo "==========================="
    echo "Example: ./build_image.sh"
    echo "Selection: 1"
    echo "==========================="
}

function get_image_tag () {
    echo "hamacho:v${VERSION}-${opt_arr[$1]}"
}

show_usage

VERSION=$(grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
echo "Detected version from 'pyproject.toml' file is $VERSION"
echo "Select image type."
read -p 'Selection: ' input

# do some validation
if [[ -z "$input" ]]; then
    echo "Invalid selection. Nothing is selected. Try again."
    exit 1
elif [[ -z ${opt_arr[$input]} ]]; then
    echo "Invalid selection. Try again."
    exit 1
elif [[ -z "${input##*[!0-9]*}" ]]; then
    echo "Invalid selection. Not a positive integer. Try again."
    exit 1
fi

img_tag="$(get_image_tag $input)"

echo "The <image>:<tag> will be $img_tag"
echo "If a different <image>:<tag> is required, please input it below. Otherwise hit Enter."
echo "==========================="
echo "Example:"
echo "<image>:<tag> => abd:v1.0"
echo "==========================="
read -p '<image>:<tag> => ' custom_img_tag

if [[ -z "$custom_img_tag" ]]; then
    echo "Continuing with image tag '$img_tag'"
elif [[ $custom_img_tag != *":"* ]]; then
    echo "Invalid image tag. There must be a ':' between <image> and <tag>."
    exit 1
else
    img_tag=$custom_img_tag
    echo "Image tag set '$img_tag'"
fi

case $input in
    "0")
    echo "Building Production Image with '$img_tag' tag"
    docker build --no-cache -t $img_tag --build-arg INSTALL_GROUPS="prod" -f docker/Dockerfile .
    ;;

    "1")
    echo "Building Development Image with '$img_tag' tag"
    echo "In this development image all the required dependencies of api, test, docs, dev groups will be installed."
    docker build -t $img_tag --build-arg INSTALL_GROUPS="api,test,docs,dev" -f docker/Dockerfile .
    ;;

    "2")
    echo "Building Image with api code and dependency included with '$img_tag' tag"
    docker build -t $img_tag --build-arg INSTALL_GROUPS="api" -f docker/Dockerfile .
    ;;

    "3")
    echo "Building Image with test code and dependency included with '$img_tag' tag"
    docker build -t $img_tag --build-arg INSTALL_GROUPS="test" -f docker/Dockerfile .
    ;;

    "4")
    echo "Building Image with api, test code and dependency included with '$img_tag' tag"
    docker build -t $img_tag --build-arg INSTALL_GROUPS="api,test" -f docker/Dockerfile .
    ;;

    "5")
    echo "Building Production Image as a Lib with '$img_tag' tag"
    docker build --no-cache -t $img_tag --build-arg INSTALL_GROUPS="prod" --build-arg INSTALL_TYPE="lib" -f docker/Dockerfile .
    ;;

    "6")
    echo "Building Obfuscated Production Image with '$img_tag' tag"
    docker build --no-cache -t $img_tag --build-arg INSTALL_GROUPS="api" -f docker/Dockerfile.obfuscate .
    ;;

    *)
    echo "Invalid Selection! Please check usage and try again."
    exit 1
    ;;
esac
