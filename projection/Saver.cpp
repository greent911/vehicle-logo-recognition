#include "Saver.h"
#include <opencv2/highgui/highgui_c.h>


Saver::Saver(void):m_path("")
{
}

Saver::Saver( std::string path ):m_path(path)
{

}


Saver::~Saver(void)
{
	
}

void Saver::saveImage( IplImage* iamge,std::string name )
{
	assert(m_path != "");
	std::string fullpath = m_path + name;
	cvSaveImage(fullpath.c_str(), iamge,0);
}
