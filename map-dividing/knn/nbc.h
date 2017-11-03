#ifndef NBC
#define NBC
#include <cstring>

int get_color(char *c)
{
	if(strcmp(c, "Red")==0)
		return 0;
	else if(strcmp(c, "Yellow")==0)
		return 1;
	else if(strcmp(c, "White")==0)
		return 2;
	else return -1;
}

int get_type(char *t)
{
	if(strcmp(t, "Sports")==0)
		return 0;
	else if(strcmp(t, "SUV")==0)
		return 1;
	else if(strcmp(t, "Luxury")==0)
		return 2;
	else return -1;
}

int get_origin(char *o)
{
	if(strcmp(o, "USA")==0)
		return 0;
	else if(strcmp(o, "JP")==0)
		return 1;
	else if(strcmp(o, "GM")==0)
		return 2;
	else return -1;
}

int get_transmission(char *tr)
{
	if(strcmp(tr, "Manual")==0)
		return 0;
	else if(strcmp(tr, "Auto")==0)
		return 1;
	else if(strcmp(tr, "Combine")==0)
		return 2;
	else return -1;
}

int get_stolen(char *s)
{
	if(strcmp(s, "Yes")==0)
		return 0;
	else if(strcmp(s, "No")==0)
		return 1;
	else return -1;
}
#endif
