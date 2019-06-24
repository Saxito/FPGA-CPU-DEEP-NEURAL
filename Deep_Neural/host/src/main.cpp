#include "rnn.h"
#include "lstm.h"
#include "calcul.h"
#include <stdio.h>
#include <cstring>


const char* file_name_learn = "/home/guillaume/Documents/stage/Multiplication_Maxtrix/Deep_Neural/data/KDDTrain+.txt";
const char* file_name_test = "/home/guillaume/Documents/stage/Multiplication_Maxtrix/Deep_Neural/data/KDDTest+.txt";

int main(int argc, char** argv)
{ 
  if(argc != 1){
    if(!strcmp(argv[1], "-lstm")){
      lstm_train(file_name_learn);
    }
    else if(!strcmp(argv[1], "-rnn")){
        learn_KDD(file_name_learn);
        test_KDD(file_name_test);
    }else{
      printf("Choissisez un moyen de calcul : -lstm pour choisir lstm -rnn pour le mode normal \n");
    }
  }else{
      printf("Choissisez un moyen de calcul : -lstm pour choisir lstm -rnn pour le mode normal \n");
  } 

  return 0;
}