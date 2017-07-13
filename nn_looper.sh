#!/bin/bash

LAYERS="1 3 5 7"
WEIGHTS="20 30 60 90"
LAMBDAS="10 15 20 40"
PERCENTS="20 30 50 100"

for layer in $LAYERS
do 
  for weight in $WEIGHTS
  do 
    for lambda in $LAMBDAS
    do
      for percent in $PERCENTS
      do
        filename="layers_"$layer"_weights_"$weight"_lambda_"$lambda"_percent_"$percent".png"
        python experiment_46.py $layer $weight $lambda $percent $filename
      done
    done
  done
done

