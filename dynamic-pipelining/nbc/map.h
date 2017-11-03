#ifndef MAP
#define MAP

#include "lib\util_device.h"
#include "lib\roc_f.h"
#include "lib\rog_f.h"
#include "lib\rol_f.h"
#include "lib\kvbuffer.h"
#include "lib\kvbuffer_f.h"

bool eq(__global char *str1, __constant char *str2)
{
        int i = 0;
        char tmp1 = str1[0];
        char tmp2 = str2[0];
        while(tmp1==tmp2)
        {
                if(tmp1=='\0')
                        return true;
                i++;
                tmp1 = str1[i];
                tmp2 = str2[i];
        }
        return false;
}

int get_color_d(__global char *c)
{
        if(eq(c, "Red"))
                return 0;
        else if(eq(c, "Yellow"))
                return 1;
        else if(eq(c, "White"))
                return 2;
        else return -1;
}

int get_type_d(__global char *t)
{
        if(eq(t, "Sports"))
                return 0;
        else if(eq(t, "SUV"))
                return 1;
        else if(eq(t, "Luxury"))
                return 2;
        else return -1;
}

int get_origin_d(__global char *o)
{
        if(eq(o, "USA"))
                return 0;
        else if(eq(o, "JP"))
                return 1;
        else if(eq(o, "GM"))
                return 2;
        else return -1;
}

int get_transmission_d(__global char *tr)
{
        if(eq(tr, "Manual"))
                return 0;
        else if(eq(tr, "Auto"))
                return 1;
        else if(eq(tr, "Combine"))
                return 2;
        else return -1;
}

int get_stolen_d(__global char *s)
{
        if(eq(s, "Yes"))
                return 0;
        else if(eq(s, "No"))
                return 1;
        else return -1;
}

int get_age_d(__global char *a)
{
        if(eq(a, "1"))
                return 1;
        else if(eq(a, "2"))
                return 2;
        else if(eq(a, "3"))
                return 3;
        else if(eq(a, "4"))
                return 4;
        else if(eq(a, "5"))
                return 5;
        else return -1;
}

bool map_cpu(__global Reduction_Object_C *object, __global void *global_data, __global void *offset)
{
	int c_i = ((__global char *)global_data)[0];
		int t_i = ((__global char *)global_data)[1];
		int o_i = ((__global char *)global_data)[2];
		int tr_i = ((__global char *)global_data)[3];
		int s_i = ((__global char *)global_data)[4];

		unsigned int ofst = *(__global unsigned int *)offset;
		__global char *c, *t, *o, *tr, *s, *a;
		int count = 0;
		__global char *p = (__global char *)global_data + ofst;
		c = p;
		for(int i = 0; count<6; i++)
		{
			if(((__global char *)global_data)[ofst+i]=='\0')
			{
				count++;
				if(count==1)
				t = p + i + 1;
				else if(count==2)
				o = p + i + 1;
				else if(count==3)
				tr = p + i + 1;
				else if(count==4)
				s = p + i + 1;
				else if(count==5)
				a = p + i + 1;
			}
		}

        int c_d = get_color_d(c);
        int t_d = get_type_d(t);
        int o_d = get_origin_d(o);
        int tr_d = get_transmission_d(tr);
        int s_d = get_stolen_d(s);
		int age = get_age_d(a);

		//printf("age: %c\n", *a);

		age--;

		int keys[6];
		for(int i = 0; i<6; i++)
		keys[i] = -1;

		keys[0] = age+25;

		if(c_d==c_i)
			keys[1] = age*5;
		if(t_d==t_i)
			keys[2] = age*5+1;
		if(o_d==o_i)
			keys[3] = age*5+2;
		if(tr_d==tr_i)
			keys[4] = age*5+3;
		if(s_d==s_i)
			keys[5] = age*5+4;

		int j = 0; //records the index
		int val = 1;
		bool result;
		for(; j<6; j++)
		{
			if(keys[j]!=-1)
			{
				result = cinsert_from_private(object, &keys[j], sizeof(int), &val, sizeof(int));
			}

			//the object is full
			if(!result)
				break;
         }

		if(result)
			return true;
		else
			return false;
}

bool map_local(__local Reduction_Object_S *object, __global void *global_data, 
	__global void *offset)
{
	int c_i = ((__global char *)global_data)[0];
		int t_i = ((__global char *)global_data)[1];
		int o_i = ((__global char *)global_data)[2];
		int tr_i = ((__global char *)global_data)[3];
		int s_i = ((__global char *)global_data)[4];

		unsigned int ofst = *(__global unsigned int *)offset;
		__global char *c, *t, *o, *tr, *s, *a;
		int count = 0;
		__global char *p = (__global char *)global_data + ofst;
		c = p;
		for(int i = 0; count<6; i++)
		{
			if(((__global char *)global_data)[ofst+i]=='\0')
			{
				count++;
				if(count==1)
				t = p + i + 1;
				else if(count==2)
				o = p + i + 1;
				else if(count==3)
				tr = p + i + 1;
				else if(count==4)
				s = p + i + 1;
				else if(count==5)
				a = p + i + 1;
			}
		}

        int c_d = get_color_d(c);
        int t_d = get_type_d(t);
        int o_d = get_origin_d(o);
        int tr_d = get_transmission_d(tr);
        int s_d = get_stolen_d(s);
		int age = get_age_d(a);

		//printf("age: %c\n", *a);

		age--;

		int keys[6];
		for(int i = 0; i<6; i++)
		keys[i] = -1;

		keys[0] = age+25;

		if(c_d==c_i)
			keys[1] = age*5;
		if(t_d==t_i)
			keys[2] = age*5+1;
		if(o_d==o_i)
			keys[3] = age*5+2;
		if(tr_d==tr_i)
			keys[4] = age*5+3;
		if(s_d==s_i)
			keys[5] = age*5+4;

		int j = 0; //records the index
		int val = 1;
		bool result;
		for(; j<6; j++)
		{
			if(keys[j]!=-1)
			{
				result = linsert(object, &keys[j], sizeof(int), &val, sizeof(int));
			}

			//the object is full
			if(!result)
				break;
         }

		if(result)
			return true;
		else
			return false;
}

bool map_buffer(__global Kvbuffer *buffer, __global void *global_data, 
	__global void *offset, __local unsigned int *index_offset, __local unsigned int *pool_offset)
{
	int c_i = ((__global char *)global_data)[0];
		int t_i = ((__global char *)global_data)[1];
		int o_i = ((__global char *)global_data)[2];
		int tr_i = ((__global char *)global_data)[3];
		int s_i = ((__global char *)global_data)[4];

		unsigned int ofst = *(__global unsigned int *)offset;
		__global char *c, *t, *o, *tr, *s, *a;
		int count = 0;
		__global char *p = (__global char *)global_data + ofst;
		c = p;
		for(int i = 0; count<6; i++)
		{
			if(((__global char *)global_data)[ofst+i]=='\0')
			{
				count++;
				if(count==1)
				t = p + i + 1;
				else if(count==2)
				o = p + i + 1;
				else if(count==3)
				tr = p + i + 1;
				else if(count==4)
				s = p + i + 1;
				else if(count==5)
				a = p + i + 1;
			}
		}

        int c_d = get_color_d(c);
        int t_d = get_type_d(t);
        int o_d = get_origin_d(o);
        int tr_d = get_transmission_d(tr);
        int s_d = get_stolen_d(s);
		int age = get_age_d(a);

		//printf("age: %c\n", *a);

		age--;

		int keys[6];
		for(int i = 0; i<6; i++)
		keys[i] = -1;

		keys[0] = age+25;

		if(c_d==c_i)
			keys[1] = age*5;
		if(t_d==t_i)
			keys[2] = age*5+1;
		if(o_d==o_i)
			keys[3] = age*5+2;
		if(tr_d==tr_i)
			keys[4] = age*5+3;
		if(s_d==s_i)
			keys[5] = age*5+4;

		int j = 0; //records the index
		int val = 1;
		int result = 0;

		unsigned int locations[6];
		for(int i = 0; i<6; i++)
		locations[i] = -1;

		for(; j<6; j++)
		{
			if(keys[j]!=-1)
			{
				result = kinsert_g(buffer, &keys[j], sizeof(int), &val, sizeof(int), index_offset, pool_offset);
			}

			//the object is full
			if(result == -1)
			{
				/*roll back*/
				for(int k = 0; k < 6; k++)
				{
					if(locations[k]!=-1)
						kset_empty(buffer, locations[k]);
				}

				break;
			}

			else
			{
				locations[j] = result;
			}
         }

		if(result!=-1)
			return true;
		else
			return false;
}

#endif