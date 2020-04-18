#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

/*
*	Define important variables that will not be changed.
*	These variables will be used throughout the program.
*/
#define	TRUE	0			//This is simply an example define, can be removed later
#define coord	2			//Size of position array; position[0] is x, position[1] is y

/*
*	Define data structures.
*	If states are required , utilize enum.
*/

/*
*	An example of states that a car could be in.
*	This may be thrown out entirely, just an idea.
*/
typedef enum
{
	OFF,
	IDLE,
	MOVING,
	BRAKING,
	SOS
} carStatus;


/*
*	Data structure of the vehicle data handling that will be used throughout
*	the program.
*	Data received must be converted to this format.
*	More variables to be added.
*/
typedef struct
{
	carStatus status;
	int speed;
	int position[coord];
} vData;
#endif