thisScriptPath=$(dirname $0)
execPath=$thisScriptPath/../../../build/examples/test_dispatch_combine_ops
echo $execPath
# ------------------------------------------------------------------------------------------------ #
#                                          Inra-Node Test                                          #
# ------------------------------------------------------------------------------------------------ #
worldSizeList=(2 4 8)
hiddenStateSizeList=(7168)
scaleDimList=(8 32)
tokenNumList=(1 128)
expertPerRankList=(8 256)
expertPerTokenList=(8)
warpPerBlockList=(4)
blockNumList=(8)
dataTypeList=("fp8" "bf16")
scaleTypeList=("fp8" "fp32")

for worldSize in "${worldSizeList[@]}"; do
for hiddenStateSize in "${hiddenStateSizeList[@]}"; do
for scaleDim in "${scaleDimList[@]}"; do
for tokenNum in "${tokenNumList[@]}"; do
for expertPerRank in "${expertPerRankList[@]}"; do
for expertPerToken in "${expertPerTokenList[@]}"; do
for warpPerBlock in "${warpPerBlockList[@]}"; do
for blockNum in "${blockNumList[@]}"; do
for dataType in "${dataTypeList[@]}"; do
for scaleType in "${scaleTypeList[@]}"; do

cmd="mpirun -np $worldSize --allow-run-as-root $execPath --cmd test --data_type $dataType --hdim=$hiddenStateSize \
--scale_dim=$scaleDim --max_tokens=$tokenNum --expert_per_rank=$expertPerRank --expert_per_token=$expertPerToken \
--warp_per_blk=$warpPerBlock --block_num=$blockNum --scale_type=$scaleType --max_token_type_size=4 --num=3"
echo "$cmd"
eval "$cmd"
if [ $? -ne 0 ]; then
    echo "Command failed: $cmd"
    exit 1
fi

done # scaleType
done # dataType
done # blockNum
done # warpPerBlock
done # expertPerToken
done # expertPerRank
done # tokenNum
done # scaleDim
done # hiddenStateSize
done # worldSize
