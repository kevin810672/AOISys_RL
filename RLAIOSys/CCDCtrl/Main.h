#include "uEye.h"
#include "rs232.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <stdio.h>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;


/***************************************   Parameters  ***************************************/
struct CAM_PARAMETER
{
	HIDS hCam;          // 0 for the next available Camera. 1-254 to access a specific camera by ID
	SENSORINFO sInfo;
	UINT cbSizeOfParam;
	char* pcImageMemory;
	int nMemoryId;
	int nRet;
	int nColorMode;
	int nBitsPerPixel;
	int gamma;
	INT gain;
	int fileName;
	double expTime, min, max, intervall;
	string colorMode;
	string type;
	bool init;
	CAM_PARAMETER()     //initialize the camera parameters;
	{
		hCam = 0;
		cbSizeOfParam = 8; 
		fileName = 0;
		type = '0';
		expTime = 1;
		init= true;
		gamma = 160;
		gain = 100;
	}

};

struct SERIAL_PARAMETER
{
	/*	Baud Rate:19200
	Byte Size:8 bits
	Parity:None
	Stop Bit:1 stop bit	*/
	
	fstream ACT, TER;
	int port_n,
		bdrate;
	char mode[4];
	string command;
	string actionPath, statePath;
	int count;
	bool init0, init1;
	SERIAL_PARAMETER()     // initialize the serailport parameter
	{
		init0 = true;
		init1 = true;
		count = 0;
		int port_i = -1;
		port_n = 3; // COM X
		port_n = port_i + port_n;   
		bdrate = 19200;
		char MODE[] = {'8', 'N', '1', 0};
		actionPath = "../Communication/action.txt";
		statePath = "../Communication/state.txt";
		strcpy(mode, MODE);
	}
};

/********************************************************************************************/


/***************************************   Functions  ***************************************/
bool camInit();
void imgCapture();
bool portInit();
void portClose();
bool setParameters();
int state();
void camClose();
string recTime();
void state_update(string);
/********************************************************************************************/