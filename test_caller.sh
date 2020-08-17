#!/bin/bash
pipenv shell


BATCH_SIZE=8
CSV_BATCH_SIZE=1000
K=200



declare -a test_datasets=("war_stories_dataset" "war_stories_deduplicated" "NSU_stories_dataset")
declare -a other_datasets=("tiny_dataset" "tiny_failing_dataset" "war_stories_deduplicated_no_fail")
declare -a valid_model_names=("bert-base-uncased" "bert-base-cased" "bert-base-multilingual-cased" "roberta-base-openai-detector")
declare -a german_dataset_model_names=("bert-base-german-dbmdz-uncased" "bert-base-german-dbmdz-cased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")
declare -a valid_SentenceBert_model_names=("bert-base-nli-stsb-mean-tokens" "roberta-base-nli-stsb-mean-tokens" "distiluse-base-multilingual-cased")
declare -a german_dataset_SentenceBert_models_names=("distiluse-base-multilingual-cased")
declare -a pooling_modes=("CLS" "mean")


array_contains () { 
    local array="$1[@]"
    local seeking=$2
    local in=1
    for element in "${!array}"; do
        if [[ $element == "$seeking" ]]; then
            in=0
            break
        fi
    done
    return $in
}

# $1 is the dataset $2 is the pooling method $3  is the model name These three get concatenated as the name for the test case
function execute_test() {
name="$1_$2_$3"
if [ $2 = "CLS" ]
then
    BATCH_SIZE=1
else
    BATCH_SIZE=8
fi
python index_generator.py start $BATCH_SIZE $name $1 $2 $3
python experiments.py $name $CSV_BATCH_SIZE $K
}
case $# in
    1)
        dataset=$1
        if [ $dataset = "all" ] 
        then
            for test_dataset in "${test_datasets[@]}"
            do
                if ! [ "$test_dataset" = "NSU_stories_dataset" ]
                then
                    for pooling_mode in "${pooling_modes[@]}"
                    do
                        for model_name in "${valid_model_names[@]}"
                        do
                            execute_test $test_dataset $pooling_mode $model_name
                        done
                    done
                    for model_name in "${valid_SentenceBert_model_names[@]}"
                    do
                        execute_test $test_dataset "SentenceBert" $model_name
                    done
                else
                    for pooling_mode in "${pooling_modes[@]}"
                    do
                        for model_name in "${german_dataset_model_names[@]}"
                        do
                            execute_test $test_dataset $pooling_modep $model_name
                        done
                    done
                    for model_name in "${german_dataset_SentenceBert_models_names[@]}"
                    do
                        execute_test $test_dataset "SentenceBert" $model_name
                    done
                fi
            done
        elif array_contains test_datasets $dataset || array_contains other_datasets $dataset
        then
            if ! [ "$dataset" = "NSU_stories_dataset" ]
            then
                for pooling_mode in "${pooling_modes[@]}"
                do
                    for model_name in "${valid_model_names[@]}"
                    do
                        execute_test $dataset $pooling_mode $model_name
                    done
                done
                for model_name in "${valid_SentenceBert_model_names[@]}"
                do
                    execute_test $dataset "SentenceBert" $model_name
                done
            else
                for pooling_mode in "${pooling_modes[@]}"
                do
                    for model_name in "${german_dataset_model_names[@]}"
                    do
                        execute_test $dataset $pooling_mode $model_name
                    done
                done
                for model_name in "${german_dataset_SentenceBert_models_names[@]}"
                do
                    execute_test $dataset "SentenceBert" $model_name
                done
            fi
        else
            echo "no valid one argument command $1"
            exit 1
        fi
        ;;
    2)
        echo "two variables are not valid. You've given $1 and $2, either give only a dataset, or dataset, model and pooling"
        exit 1
        ;;
    3)
        dataset=$1
        model_name=$2
        pooling_mode=$3

        if ! array_contains test_datasets $dataset && ! array_contains other_datasets $dataset 
        then
            echo "unknown dataset $dataset, datasets are ${other_datasets[@]} ${test_datasets[@]} "
        fi
        if  ! array_contains pooling_modes $pooling_mode  && ! [ "$pooling_mode" = "SentenceBert" ]
        then
            echo "unknown pooling mode $pooling_mode, pooling modes are ${pooling_modes[@]} and SentenceBert"
        else
            if [ "$pooling_mode" = "SentenceBert" ]
            then
                if ! [ "$dataset" = "NSU_stories_dataset" ]
                then
                    if array_contains valid_SentenceBert_model_names $model_name
                    then
                        execute_test $dataset $pooling_mode $model_name
                    else
                        echo "$model_name is not a valid SentenceBert model"
                    fi
                else
                    if array_contains german_dataset_SentenceBert_models_names $model_name
                    then
                        execute_test $dataset $pooling_mode $model_name
                    else
                        echo "$model_name is not a valid SentenceBert model for German data sets"
                    fi
                fi
            else
                if ! [ "$dataset" = "NSU_stories_dataset" ]
                then
                    if array_contains valid_model_names $model_name
                    then
                        execute_test $dataset $pooling_mode $model_name
                    else
                        echo "$model_name is not a valid model for pooling mode $pooling_mode"
                    fi
                else
                    if array_contains german_dataset_model_names $model_name
                    then
                        execute_test $dataset $pooling_mode $model_name
                    else
                        echo "$model_name is not a valid model for pooling mode $pooling_mode for German datasets"
                    fi
                fi


            fi
        fi
        ;;
    *)
        echo="not a valid number of input arguments"
        ;;
esac

# example  calls for reference
# python index_generator.py start 8 tiny_dataset_test tiny_failing_dataset bert-base-cased CLS
# python experiments.py tiny_dataset_test 1000 200
