#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned len = (8 << 20);

void func(int * a)
{
	int i, j;

	int m, n;

	m = 1000;
	n = len / m;
	
	for(j=0; j<m; j++){		
		for(i=0; i<n; i++){
			a[i*m+j]++;
		}
	}
}

int main()
{
	int * a;
	int i;

	int nEvents, retval;
    int EventSet = PAPI_NULL;
    int events[] = {PAPI_L2_DCM}; //, PAPI_L2_TCM};
    long_long values[] = {0, 0};
    char eventLabel[PAPI_MAX_STR_LEN];

    if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
    {
        printf("\n\t  Error : PAPI Library initialization error! \n");
        return(-1);
    }

    if((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
    {   printf("\n\t  Error : PAPI failed to create the Eventset\n");
        printf("\n\t  Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }

    nEvents = sizeof(events)/sizeof(events[0]);
    for(i = 0; i < nEvents; i++){
        if((retval = PAPI_add_event(EventSet, events[i])) != PAPI_OK)
        {
            printf("\n\t   Error : PAPI failed to add event %d\n", i);
            printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        }
    }


	a = (int *) malloc(len*sizeof(int));
	PAPI_start(EventSet);
	func(a);
	PAPI_stop(EventSet, values);
	free(a);

	/*Print out PAPI reading*/
    for(i = 0; i < nEvents; i++){
        PAPI_event_code_to_name(events[i], eventLabel);
        printf("%s:\t%lld\t", eventLabel, values[i]);
    }
    printf("\n");
	

	/* Cleanup and shutdown PAPI */
	if((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)
    {
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }
    if((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
    {
        printf("\n\t   Error : PAPI failed to clean the events from created Eventset");
        printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
        return(-1);
    }
    PAPI_shutdown();

}
