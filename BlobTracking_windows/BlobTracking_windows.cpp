
#include <iostream>
#include <string>

#include "App.h"



int main()
{
	App * app = new App("t1.MP4");
	//App * app = new App(0);
	
	app->RunTracking(2.2);

	return 0;
}
