//Our own created .h files
#include "dataHandling.h"


/*
*	External libraries that will be used.When a ned library is add,
*	append a comment to the library explaining why it is needed.
*/
#include <stdio.h>					//Standard C library


//Declare global variable type [structure of data to be interpreted]

/*
*	NOTE: Function types can/will change when code begins to be formed.
*	ex: void -> vData where vData is the structure that is a series of
*		variables that the vehicle uses to executes commands.
*	May need to distinguish between V2V and V2C data handling
*/

void sendData()
{
	/*
	*	Will convert the selected data to a data packer and send it.
	*/
}


void receiveData()
{
	/*
	*	Will receive a data packet and then execute parseData() to 
	*	convert it.
	*/
}


void parseData()
{
	/*
	*	Input: dataPack
	*	Output: 
	*	Function will parse data into structure that will be used by
	*	the program to execute commands.
	*	Convert the received data into a structure to be defined in dataStruct.h.
	*/
}


void compressData()
{
	/*
	*	Input: dataToSend
	*	Output:
	*	Function will convert the data structure used by the vehicle to the appropriate
	*	format.
	*/
}