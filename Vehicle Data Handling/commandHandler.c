//Our own created .h files
#include "commandHandler.h"


/*
*	External libraries that will be used.When a ned library is add,
*	append a comment to the library explaining why it is needed.
*/
#include <stdio.h>					//Standard C library


/*
*	This file will be the brains of the vehicle.
*	It will utilize functions to collect data, interpret data, and issue
*	commands as it sees nessecary.
*
*	In essence there will be a while loop that will continously executes commands;
*	the loop will poll for data, interpret the data, issue commands (or do nothing if
*	nothing has changed), and send data.
*
*	There may be an issue with receiving data, as with the loop design, a vehicle will
*	receive data with some latency, so design must either be tweaked or it must be
*	optimized.
*/