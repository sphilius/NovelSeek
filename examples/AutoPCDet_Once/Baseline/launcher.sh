source activate pcdet

cd tools

# Check if $1 exists, if not create the directory
if [ -z "$1" ]; then
    echo "Error: Output directory not specified"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "Creating output directory: $1"
    mkdir -p "$1"
fi

bash scripts/dist_train.sh 2 --cfg_file ./cfgs/once_models/centerpoint.yaml --out_dir $1 --extra_tag $1 
cd ../
cp -r tools/$1/* ./