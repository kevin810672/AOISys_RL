#include "Main.h"


int main()
{
	bool check = false;
	while (state() > 0)
	{
		if (state()==1)
		{
			if (setParameters())
			{
				if (!check)
				{
					check = true;
					camInit();
				}
				imgCapture();
			}
		}
	}
	camClose();
	cout << "Press Enter to leave the program" << endl;
	cin.get();
    return 0;
}