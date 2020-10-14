
#include <iostream>
#include <string>
#include <filesystem>

#include "App.h"



int main(int argc, const char **argv)
{
	std::string testFile = "test.MP4";
	
	cv::CommandLineParser parser(argc, argv,
		"{file|test.MP4|}"
		"{scale|2.0|}"
		"{cam|-1|}"
	);
	testFile = parser.get<string>("file");

	int cam = parser.get<int>("cam");
	double scale = parser.get<double>("scale");
	
	if (cam != -1) {
		App * app = new App(cam);
		app->RunTracking(scale);
	}
	else {
		App * app = new App(testFile);
		app->RunTracking(scale);
	}
	//App * app = new App(0);


	return 0;
}
