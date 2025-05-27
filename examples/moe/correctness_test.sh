thisScriptPath=$(dirname $0)
execPath=$thisScriptPath/../../build/examples/test_dispatch_combine
echo $execPath
# ------------------------------------------------------------------------------------------------ #
#                                          Inra-Node Test                                          #
# ------------------------------------------------------------------------------------------------ #
worldSizeList=(2 4 8)
hiddenStateSizeList=(4096 6144)
tokenNumList=(32 128 512 1024)
expertPerRankList=(4 8 16)
expertPerTokenList=(4 8)
warpPerBlockList=(4)
blockNumList=(4 8 16)

for worldSize in "${worldSizeList[@]}"; do
    for hiddenStateSize in "${hiddenStateSizeList[@]}"; do
        for tokenNum in "${tokenNumList[@]}"; do
            for expertPerRank in "${expertPerRankList[@]}"; do
                for expertPerToken in "${expertPerTokenList[@]}"; do
                    for warpPerBlock in "${warpPerBlockList[@]}"; do
                        for blockNum in "${blockNumList[@]}"; do
                            mpirun -np $worldSize --allow-run-as-root $execPath $hiddenStateSize \
                              $tokenNum $expertPerRank $expertPerToken $warpPerBlock $blockNum 3
                        done
                    done
                done
            done
        done
    done
done