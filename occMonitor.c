#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int max_sm_id(int * sm_id, int warps)
{
    int max = 0;
    for (int w = 0; w < warps; w ++){
        if (sm_id[w] > max) max = sm_id[w];
    }
    return max+1;
}

int main(int argc, char *argv[])
{
    /*
     * Read in result and total #warps
     */
    if (argc < 3) {
        printf("Please specify a file name and its line numbers -1\n");
        exit(0);
    }
    int warps = atoi(argv[2]);
    FILE * fptr = fopen(argv[1], "r");
    /*
     * Read in the file and fill in
     *   sm_id
     *   tStart
     *   tEnd
     * of each warp
     */
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    if (fptr == NULL)
        exit(EXIT_FAILURE);
    long int *tStart = (long int*)malloc(sizeof(long int)*warps);
    long int *tEnd = (long int*)malloc(sizeof(long int)*warps);
    int *sm_id = (int*)malloc(sizeof(int)*warps);
    /* skip the header line */
    getline(&line, &len, fptr);
    /* Start to read each line and get the numbers */
    char *token;
    int count = 0;
    while ((read =getline(&line, &len, fptr)) != -1 ){
        fflush(stdout);
        token = strtok(line, " \t");
        token = strtok(NULL, " \t");
        sm_id[count] = atoi(token);
        token = strtok(NULL, " \t");
        tStart[count] = atol(token);
        token = strtok(NULL, " \t");
        tEnd[count] = atol(token);
        count ++;
    }
    fclose(fptr);
    /*
     * Step 1: obtain the max and min cycles on each SM
     */
    int numSM = max_sm_id(sm_id, warps);
    long int *minCycle, *maxCycle;
    minCycle=(long int*)malloc(sizeof(long int)*numSM);
    maxCycle=(long int*)malloc(sizeof(long int)*numSM);
    int *asgW = (int*)malloc(sizeof(int)*numSM);
    for (int sm = 0; sm < numSM; sm++){
        minCycle[sm] = LONG_MAX; // max of long int
        maxCycle[sm] = LONG_MIN; // min of long int
        asgW[sm] = 0;
    }
    for (int w = 0; w < warps; w ++){
        int sm = sm_id[w];
        asgW[sm]++;
        if (tStart[w] < minCycle[sm])
            minCycle[sm] = tStart[w];
        if (tEnd[w] > maxCycle[sm])
            maxCycle[sm] = tEnd[w];
    }
    /*
     * Step 2: Compute the occupancy on each SM
     */
    int **occ = (int**)malloc(sizeof(int*)*numSM);
    for (int sm = 0; sm < numSM; sm ++){
        size_t occSize = sizeof(int)*(maxCycle[sm]-minCycle[sm]+1);
        occ[sm] = (int*)malloc(occSize);
        for (int i = 0; i < maxCycle[sm]-minCycle[sm]+1; i ++)
            occ[sm][i] = 0;
    }
    for (int w = 0; w < warps; w ++){
        int sm = sm_id[w];
        occ[sm][tStart[w]-minCycle[sm]] ++;
        occ[sm][tEnd[w]-minCycle[sm]] --;
    }
    /*
     * Print a brief summary of each SM
     */
    printf("Occupancy Summary:\n");
    printf("smid  max\tave      \tlatency \ttotal\n");
    for (int sm = 0; sm < numSM; sm ++){
        int state = 0;
        int max = 0;
        long long int sum = 0;

        long int len = maxCycle[sm]-minCycle[sm]+1;
        for (int i = 0; i < len; i ++){
            state += occ[sm][i];
            occ[sm][i] = state;
            if (state > max) max = state;
            sum += state;
        }
        printf("%2d    %d\t%f\t%ld\t%d\n", sm, max, (float)sum/(float)len, len-1, asgW[sm]);
    }
    /*
     * If a third parameter is provided, write the occ of the required SM
     * into a file for plot
     * E.g. To write the occupancy of sm 0 to occ_smN.txt, do
     *   ./a.out result_1024_32_128.txt 89600 N
     * To plot the figure, do
     *   ./plot N
     */
    if (argc == 4){
        int sm = atoi(argv[3]);
        if (sm >= numSM){
            printf("SM id too big\n");
            exit(EXIT_FAILURE);
        }
        char filename[32];
        sprintf(filename, "occ_sm%d.txt", sm);
        printf("writing occupancy of sm %d to %s ...\n", sm, filename);
        FILE *fp = fopen(filename, "w");
        long int len = maxCycle[sm]-minCycle[sm]+1;
        for (int i = 0; i < len; i ++) {
            fprintf(fp, "%d  %d\n", i, occ[sm][i]);
        }
        fclose(fp);
    }

    /*
     * Write the arrays into the file and compare it with the input file
     * to see if we read in the data correctly.
     */
#if 0
    fptr = fopen("output.txt", "w");
    fprintf(fptr, "warpid\tsmid\tstart\tend\n");
    for (int w = 0; w < warps; w ++){
        fprintf(fptr, "%5d\t%2d\t%ld\t%ld\n", w, sm_id[w], tStart[w], tEnd[w]);
    }
    fclose (fptr);
#endif

    /* clean up */
    free (tStart);
    free (tEnd);
    free (sm_id);
    free (line);
    free (minCycle);
    free (maxCycle);
    free (asgW);
    for (int sm = 0; sm < numSM; sm ++)
        free (occ[sm]);
    free (occ);
    return EXIT_SUCCESS;
}
