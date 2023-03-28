cwd=`pwd`

pushd ../tools

sh run_all_models.sh $1 $2 $cwd/model_configs.json

popd