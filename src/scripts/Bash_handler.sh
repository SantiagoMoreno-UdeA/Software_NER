#!/bin/bash
# NER software handler

if [-n $1]
then
    if [$1 = 'TRAIN']
    then
       shift # past argument
       if [$# -gt 1]
       then
       while [[ $# -gt 0 ]]; do
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
           python Train_model.py -m ${MODEL} -id ${INPUTDIR} -u ${UFLAG} -cu ${CUDA}

    else
        echo Not arguments requires at least input directory
    fi
    

    elif [$1 = 'USE']
    then
       shift # past argument
       if [$# -gt 2]
       then
       while [[ $# -gt 0 ]]; do
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
           python Tagged_document.py -m ${MODEL} -id ${INPUTDIR} -od ${OUTPUTDIR} -cu ${CUDA}

       else
           echo Not arguments requires at least model and input file
       fi

    else
	echo invalid option, USE for use a model, TRAIN for train a new model
    fi

fi