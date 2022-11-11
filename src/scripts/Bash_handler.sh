#!/bin/bash
# NER software handler

if [ $# -gt 0 ]
    then
    MODE="$1"
        if [ ${MODE} == 'TRAIN' ]
        then
            shift # past argument
            if [ $# -gt 1 ]
                then
                while [[ $# -gt 1 ]]; do
                    case $1 in
                        -m|--model)
                        MODEL="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -id|--inputdir)
                        INPUTDIR="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -u|--upsampleflag)
                        UFLAG="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -cu|--cuda)
                        CUDA="$2"
                        shift # past argument
                        shift # past value
                        ;;

                    esac
                done
                    if [ -n "${UFLAG}" ] && [ -n "${CUDA}" ]; then
                        python Train_model.py -m ${MODEL} -id "${INPUTDIR}" -u "${UFLAG}" -cu "${CUDA}"

                    elif [[ -n "${UFLAG}" ]]; then
                        python Train_model.py -m ${MODEL} -id "${INPUTDIR}" -u "${UFLAG}" 

                    elif [[ -n "${CUDA}" ]]; then
                        python Train_model.py -m ${MODEL} -id "${INPUTDIR}" -cu "${CUDA}"

                    else
                        python Train_model.py -m ${MODEL} -id "${INPUTDIR}"
                    fi

            else
                echo Not arguments the script requires at least input directory
            fi


        elif [ $1 == 'USE' ]
        then
        shift # past argument
        if [ $# -gt 1 ]
            then
            while [[ $# -gt 1 ]]; do
                case $1 in
                    -m|--model)
                    MODEL="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -id|--inputdir)
                    INPUTDIR="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -od|--outputdir)
                    OUTPUTDIR="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -cu|--cuda)
                    CUDA="$2"
                    shift # past argument
                    shift # past value
                    ;;

                esac

            done
            if [ -n "${OUTPUTDIR}" ] && [ -n "${CUDA}" ]; then
                python Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -od "${OUTPUTDIR}" -cu "${CUDA}"

            elif [[ -n "${OUTPUTDIR}" ]]; then
                python Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -od "${OUTPUTDIR}" 

            elif [[ -n "${CUDA}" ]]; then
                python Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -cu "${CUDA}"

            else
                python Tagged_document.py -m ${MODEL} -id "${INPUTDIR}"
            fi
        

        else
            echo Not arguments the script requires at least model and input file
        fi

    else
        echo invalid option, USE for use a model, TRAIN for train a new model
    fi

fi