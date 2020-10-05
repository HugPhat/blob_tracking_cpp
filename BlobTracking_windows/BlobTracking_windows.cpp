
#include <iostream>
#include <string>

#include "App.h"



int main()
{
	App * app = new App("t1.MP4");

	app->RunTracking(3);

	return 0;
}
