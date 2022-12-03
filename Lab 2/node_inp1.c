#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

/* Please add your event here */
int events[1] = {PAPI_L2_DCM}; /*PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_DM*/
long long values[1];
int eventset;
int nEvents, retval;
char eventLabel[PAPI_MAX_STR_LEN];

static unsigned t1, t2, t3;

typedef struct s1 {
	int a;
	int b;
	int c;
	int d;
	int e;
} s1;

static s1 * a;

const int arraysize = 16*(1 << 10); 

// counters to track each node profile
int n0, n1, n2, n3, n4, n5, n6;
static inline void func(int cond1, int cond2, int cond3)
{
	int i;
	
	if(cond1 || cond3){
		n0++;	// node 0
		for(i=0; i<arraysize; i++){
			a[i].a += a[i].e;
		}
		n1++;	// node 1
		if(cond2){
			n3++; // node 3
			for(i=0; i<arraysize; i++){
				a[i].c++;
			}
			n4++;	// node 4
		}
	} else {
		for(i=0; i<arraysize; i++){
			a[i].b = a[i].d++;
		}
		n2++;	// node 2

	}

	if(cond2 && cond3){
		n5++;	// node 5
		for(i=0; i<arraysize; i++){
			a[i].c = a[i].d++;
		}
		n6++;	// node 6
	}
	
}

/* Input set 1 */
void input1()
{
	t1 = 0.5 * RAND_MAX;
	t2 = 0.5 * RAND_MAX;
	t3 = 0.5 * RAND_MAX;
}
/* End input set 1*/

/* Input set 2 */
void input2()
{
	t1 = 0.2 * RAND_MAX;
	t2 = 0.6 * RAND_MAX;
	t3 = 0.8 * RAND_MAX;
}
/* End input set 2*/

int main()
{
	int i;

	a = (s1 *) malloc(sizeof(s1) * arraysize);

	if(PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT)){
		printf("Can't initiate PAPI library!\n");
		exit(-1);
	}

	eventset = PAPI_NULL;
	if(PAPI_create_eventset(&eventset) != PAPI_OK){
		printf("Can't create eventset!\n");
		exit(-3);
	}
	nEvents = sizeof(values)/sizeof(values[0]);
	for(i = 0; i < nEvents; i++){
		if((retval = PAPI_add_event(eventset, events[i])) != PAPI_OK)	
		{	
			printf("\n\t   Error : PAPI failed to add event %d\n", i);	
			printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); 
		}
	}


	unsigned cond1, cond2, cond3;
	
	srandom(0);

	input1();
	//input2();

	if ((retval = PAPI_start(eventset)) != PAPI_OK) {
		fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(retval));
		exit(1);
	}


	for(i=0; i<10000; i++){
		cond1 = (random() > t1);
		cond2 = (random() > t2);
		cond3 = (random() > t3);

		func(cond1, cond2, cond3);
	}

	if ((retval = PAPI_stop(eventset, values)) != PAPI_OK) {
		fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
		exit(1);
	}


	/* Print out your profiling results here */
	for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
	printf("\n");


	/* Cleanup */
	free(a);

	if((retval = PAPI_cleanup_eventset(eventset)) != PAPI_OK)	
	{	
		printf("\n\t   Error : PAPI failed to clean the events from created Eventset");	
		printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
		return(-1);  
	}
	if((retval = PAPI_destroy_eventset(&eventset)) != PAPI_OK)
	{	
		printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
		printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
		return(-1); 
	}
	PAPI_shutdown(); 
}

