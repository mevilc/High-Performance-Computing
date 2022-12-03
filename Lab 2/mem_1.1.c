#include "papi.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned len = 16*(1 << 20);

typedef struct Mem {
	struct mem1 {
		int fa;
		char fb;
		char fd;
	} mem1;

	struct mem2 {
		int fc;
		int fe;
	} mem2;
} Mem;

 
void func(Mem * a)
{
	int i;

	for(i=0; i<len; i++){
		a[i].mem1.fa = a[i].mem1.fb+a[i].mem1.fd;
	}
		
	for(i=0; i<len; i++){
		a[i].mem2.fc = a[i].mem2.fe*2;
	}

}

/* Please add your events here */
int events[1] = {PAPI_L2_DCM}; /*PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_DM*/
int eventnum = 1;

int main()
{
	long long values[1];
	int eventset;
	Mem * a;

	if(PAPI_VER_CURRENT != PAPI_library_init(PAPI_VER_CURRENT)){
		printf("Can't initiate PAPI library!\n");
		exit(-1);
	}

	eventset = PAPI_NULL;
	if(PAPI_create_eventset(&eventset) != PAPI_OK){
		printf("Can't create eventset!\n");
		exit(-3);
	}
	if(PAPI_OK != PAPI_add_events(eventset, events, eventnum)){
		printf("Can't add events!\n");
		exit(-4);
	}

	a = (Mem *) malloc(len*sizeof(Mem));
	PAPI_start(eventset);
	func(a);
	PAPI_stop(eventset, values);
	free(a);

	/*Print out PAPI reading*/
	char event_name[PAPI_MAX_STR_LEN];
	if (PAPI_event_code_to_name( events[0], event_name ) == PAPI_OK)
		printf("%s: %lld\n", event_name, values[0]);
	
	return EXIT_SUCCESS;
}
