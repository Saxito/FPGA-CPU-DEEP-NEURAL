#include "rnn.h"
#include "lstm.h"
#include <stdio.h>
#include <cstring>


const char* file_name_learn = "./data/KDDTest+.txt";
const char* file_name_test = "./data/KDDTest+.txt";

int main(int argc, char** argv)
{ 
	if(argc != 1){
		if(!strcmp(argv[1], "-lstm")){
			lstm_train(file_name_learn);
			lstm_test(file_name_test);
		}else if(!strcmp(argv[1], "-rnn")){
			learn_KDD(file_name_learn, file_name_test);
			//test_KDD(file_name_test);
		}else{
			printf("Choissisez un moyen de calcul : -lstm pour choisir lstm -rnn pour le mode normal \n");
		}
	}else{
		printf("Choissisez un moyen de calcul : -lstm pour choisir lstm -rnn pour le mode normal \n");
	} 

	return 0;
}