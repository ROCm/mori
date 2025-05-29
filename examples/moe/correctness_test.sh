thisScriptPath=$(dirname $0)
execPath=$thisScriptPath/../../build/examples/test_dispatch_combine
echo $execPath
# ------------------------------------------------------------------------------------------------ #
#                                          Inra-Node Test                                          #
# ------------------------------------------------------------------------------------------------ #
worldSizeList=(2 4 8)
hiddenStateSizeList=(7168)
tokenNumList=(1 128)
expertPerRankList=(8 256)
expertPerTokenList=(8)
warpPerBlockList=(4)
blockNumList=(8)
dataTypeList=("fp8" "bf16")

for worldSize in "${worldSizeList[@]}"; do
for hiddenStateSize in "${hiddenStateSizeList[@]}"; do
for tokenNum in "${tokenNumList[@]}"; do
for expertPerRank in "${expertPerRankList[@]}"; do
for expertPerToken in "${expertPerTokenList[@]}"; do
for warpPerBlock in "${warpPerBlockList[@]}"; do
for blockNum in "${blockNumList[@]}"; do
for dataType in "${dataTypeList[@]}"; do

mpirun -np $worldSize --allow-run-as-root $execPath --cmd test --data_type $dataType --hdim=$hiddenStateSize \
        --max_tokens=$tokenNum --expert_per_rank=$expertPerRank --expert_per_token=$expertPerToken \
        --warp_per_blk=$warpPerBlock --block_num=$blockNum --num=3

done # dataTypes
done # blockNum
done # warpPerBlock
done # expertPerToken
done # expertPerRank
done # tokenNum
done # hiddenStateSize
done # worldSize