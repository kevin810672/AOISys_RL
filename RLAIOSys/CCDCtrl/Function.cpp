#include "Main.h"

CAM_PARAMETER camData;
CAM_PARAMETER *camDataPtr = &camData;
SERIAL_PARAMETER portData;
SERIAL_PARAMETER *portDataPtr = &portData;



bool camInit()
{

    /* starts the driver and establishes the connection to the camera */
    camDataPtr->nRet = is_InitCamera(&camDataPtr->hCam, NULL);
	is_SetExternalTrigger(camDataPtr->hCam, IS_SET_TRIGGER_SOFTWARE);
    if (camDataPtr->nRet != IS_SUCCESS)
    {
        cout  << "Camera is not connected !" << endl;
		return false;
    }
    else
    {
		//is_SetExternalTrigger(camDataPtr->hCam, IS_SET_TRIGGER_SOFTWARE);
        cout << "¡iCamera initialisation is successful¡j ";
    }
	

	/* query the camera information from the ini file */
	is_GetSensorInfo(camDataPtr->hCam, &(camDataPtr->sInfo));
	
    if (camDataPtr->sInfo.nColorMode == IS_COLORMODE_BAYER)
    {
        // for color camera models use RGB24 mode
        camDataPtr->nColorMode = IS_CM_BGR8_PACKED;
		camDataPtr->colorMode = "RGB24 mode";
        camDataPtr->nBitsPerPixel = 24;
    }
    else if (camDataPtr->sInfo.nColorMode == IS_COLORMODE_CBYCRY)
    {
        // for CBYCRY camera models use RGB32 mode
        camDataPtr->nColorMode = IS_CM_BGRA8_PACKED;
		camDataPtr->colorMode = "RGB32 mode";
        camDataPtr->nBitsPerPixel = 32;
    }
    else
    {
        // for monochrome camera models use Y8 mode
        camDataPtr->nColorMode = IS_CM_MONO8;
		camDataPtr->colorMode = "MONO8 mode";
        camDataPtr->nBitsPerPixel = 8;
    }

	is_SetColorMode(camDataPtr->hCam, camDataPtr->nColorMode);
	is_Gamma(camDataPtr->hCam, IS_GAMMA_CMD_SET, (void*) &camDataPtr->gamma, sizeof(camDataPtr->gamma));
	is_SetHWGainFactor(camDataPtr->hCam, IS_SET_MASTER_GAIN_FACTOR, camDataPtr->gain);
	is_Exposure(camDataPtr->hCam, IS_EXPOSURE_CMD_SET_EXPOSURE, &camDataPtr->expTime, camDataPtr->cbSizeOfParam);

	is_Exposure(camDataPtr->hCam, IS_EXPOSURE_CMD_GET_EXPOSURE, &camDataPtr->expTime, camDataPtr->cbSizeOfParam);
	is_SetHWGainFactor(camDataPtr->hCam, IS_GET_MASTER_GAIN_FACTOR, camDataPtr->gain);
	is_Gamma(camDataPtr->hCam, IS_GAMMA_CMD_GET, (void*) &camDataPtr->gamma, sizeof(camDataPtr->gamma));
	cout << recTime() << endl << endl;
	cout << "¡iCamera basic specification¡j" << endl << endl;
	cout << " Name:          " << camDataPtr->sInfo.strSensorName << endl;
	cout << " Exposure time: " << fixed << setprecision(2) << camDataPtr->expTime << " ms" << endl;
	cout << " Gain Value:    " << fixed << setprecision(2) << camDataPtr->gain << endl;
	cout << " Gamma Value:   " << fixed << setprecision(2) << camDataPtr->gamma << endl;
	
	/* allocates an image memory for an image, activates it and sets the way in which the images will be displayed on the screen */
	int nMemoryId;
	is_AllocImageMem(camDataPtr->hCam, camDataPtr->sInfo.nMaxWidth, camDataPtr->sInfo.nMaxHeight, camDataPtr->nBitsPerPixel, &(camDataPtr->pcImageMemory), &nMemoryId);
	is_SetImageMem(camDataPtr->hCam, camDataPtr->pcImageMemory, nMemoryId);

	return true;
}



void imgCapture()
{	
	if (!camDataPtr->init)
	{
		double e;
		is_Exposure(camDataPtr->hCam, IS_EXPOSURE_CMD_SET_EXPOSURE, &camDataPtr->expTime, camDataPtr->cbSizeOfParam);
		is_Exposure(camDataPtr->hCam, IS_EXPOSURE_CMD_GET_EXPOSURE, &e, camDataPtr->cbSizeOfParam);
		cout << "¡iTraining¡j"<< recTime() << endl;
		cout << " CameraName:    " << camDataPtr->sInfo.strSensorName << endl;
		cout << " Exposure time: " << fixed << setprecision(2) << camDataPtr->expTime << " ms" << endl;
		cout << " Gain Value:    " << fixed << setprecision(2) << camDataPtr->gain << endl;
		cout << " Gamma Value:   " << fixed << setprecision(2) << camDataPtr->gamma << endl;
	}
	is_FreezeVideo(camDataPtr->hCam, IS_WAIT);
	void *pMemVoid;
	is_GetImageMem(camDataPtr->hCam, &pMemVoid);
	// create tempImg 
	IplImage* tmpImg;
	tmpImg = cvCreateImageHeader(cvSize(camDataPtr->sInfo.nMaxWidth, camDataPtr->sInfo.nMaxHeight), IPL_DEPTH_8U, 1);
	// assign image address to IplImage data pointer
	tmpImg->imageData = (char*) pMemVoid;
	Mat img = Mat(tmpImg, false);
	/* calculate the graylevel of image */
	float Sum = 0;
	for (int i=0; i<img.rows; i++)
	{
		uchar *data = img.ptr<uchar>(i);
		for (int j=0; j<img.cols; j++)
		{
			Sum += data[j];
		}
	}
	float v = Sum/(img.cols*img.rows);
	cout<< "channel"<< camDataPtr->type <<"_meanGrayValue: " << fixed << setprecision(2) << v << endl;
	cout << "----------------------------------------------------------------------" << endl << endl;
	imwrite("../Communication/pic/img.jpg", img);
	if (camDataPtr->init){camDataPtr->init=false;}

}

void camClose()
{
	/* Releases an image memory that was allocated */
	is_FreeImageMem(camDataPtr->hCam, camDataPtr->pcImageMemory, camDataPtr->nMemoryId);
	is_ExitCamera(camDataPtr->hCam);
	cout << "Close the cammera completely!" << endl;
	/* clear the state file */ 
	portDataPtr->TER.open(portDataPtr->statePath, ios::in | ios::trunc);
	portDataPtr->TER.close();
}

string recTime()
{
	time_t now = time(0);
	tm *ltm = localtime(&now);
	string t = to_string(1900 + ltm->tm_year) + "_" + to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday) + " "+ 
		       to_string(ltm->tm_hour) + ":" + to_string(ltm->tm_min) + ":" + to_string(ltm->tm_sec);
	return t;
}

bool portInit()
{
	if(RS232_OpenComport(portDataPtr->port_n, portDataPtr->bdrate, portDataPtr->mode))
	{
		printf("Can not open COM%d !\n", portDataPtr->port_n+1);
		return false;
	}
	printf("COM%d connected successfully!\n", portDataPtr->port_n+1);
	cout << "----------------------------------------------------------------------" << endl << endl;
	return true;
}

void portClose()
{
	portDataPtr->ACT.close();
	printf("Close the action.txt !\n");
	RS232_CloseComport(portDataPtr->port_n);
	printf("Close the COM%d !\n", portDataPtr->port_n+1);
}



bool setParameters()
{
	string line;
	int counter = 0;

	portDataPtr->ACT.open(portDataPtr->actionPath, ios::in | ios::out);
	if (portDataPtr->ACT.is_open() && portDataPtr->ACT.peek()!=EOF)
	{
		while (getline(portDataPtr->ACT, line))
		{
			switch (counter)
			{
			case 0:
				camDataPtr->expTime = atof(line.c_str());
				counter ++;
				break;
			case 1:
				camDataPtr->type = line.c_str();
				break;
			}	
		}
		portDataPtr->ACT.close();
		/* clean up the actoin.txt */
		portDataPtr->ACT.open(portDataPtr->actionPath, ios::out | ios::trunc);
		portDataPtr->ACT.close();
		return true;
	}
	else
	{
		if (!portDataPtr->ACT.is_open())
		{
			cout << "Action.txt is not opened successfully!" <<endl;
		}
		else
		{
			//cout << "Action.txt is empty!" <<endl;
		}
		portDataPtr->ACT.close();
		return false;
	}
}

int state()
{
	if (portDataPtr->init0)
	{
		/* Initialize the state */ 
		state_update("c");
		state_update("2");
	}
	string line;
	portDataPtr->TER.open(portDataPtr->statePath, ios::in | ios::out);
	if (portDataPtr->TER.is_open())
	{
		getline(portDataPtr->TER, line);
		if (line == "0")
		{
			portDataPtr->TER.close();
			cout << "**********************************************************************" << endl;
			cout << "*              Python State: Disconnect to the Python                *" << endl;
			cout << "**********************************************************************" << endl;
			return 0;
		}
		else if (line == "1")
		{
			portDataPtr->TER.close();
			if (portDataPtr->init1)
			{
				cout << "**********************************************************************" << endl;
				cout << "*          Python State: Connect to the Python                       *" << endl;
				cout << "**********************************************************************" << endl << endl;
				portDataPtr->init1 = false;
			}
			return 1;
		}
		else if (line == "2")
		{
			portDataPtr->TER.close();
			if (portDataPtr->init0)
			{
				cout << "**********************************************************************" << endl;
				cout << "*          Python State: Wait for connection to Python...            *" << endl;
				cout << "**********************************************************************" << endl;
				portDataPtr->init0 = false;
			}
			return 2;
		}
		else
		{
			// others state //
			portDataPtr->TER.close();
			return 3;
		}
	}
	else
	{
		portDataPtr->TER.close();
		return 0;
	}
}

void state_update(string mode)
{
	if (mode == "c")
	{
		portDataPtr->TER.open(portDataPtr->statePath, ios::in | ios::trunc);
		portDataPtr->TER.close();
	}
	else
	{
		portDataPtr->TER.open(portDataPtr->statePath, ios::in | ios::out);
		portDataPtr->TER << mode;
		portDataPtr->TER.close();
	}
};