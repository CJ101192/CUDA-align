#include <stdio.h>
int measureSimilarity(int *a, int *b, int len1, int len2)
{
    if(len1 == 0 || len2 ==0)
        return 0;
    else if(a[0]==b[0])
        return (1 + measureSimilarity(&a[1], &b[1], len1-1, len2-1));
    else if(a[0]>b[0])
        return (measureSimilarity(&a[1], b, len1-1, len2));
    else
        return (measureSimilarity(a, &b[1], len1, len2-1));
}
int measurePeriodicSimilarity(int *a, int *b, int period, int len1, int len2)
{
    if(len1 == 0 || len2 ==0)
        return 0;
    else
        return measureSimilarity(a, b, period, period) + measurePeriodicSimilarity(&a[period], &b[period], period, len1-period, len2-period);
}
/*
int main()
{
    int a[20] = {30, 25, 22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 15, 12};
    int b[20] = {22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 12, 12, 12, 12};
    int c = measureSimilarity(a,b, 20, 20);
    printf("Similarity  : %d\n", c);
    

    int d[40] = {30, 25, 22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 15, 12, 30, 25, 22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 15, 12};
    int e[40] = {22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 12, 12, 12, 12, 22, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17, 15, 15, 12, 12, 12, 12};
    int g = measurePeriodicSimilarity(d,e, 20, 40, 40);
    printf("Similarity  : %d\n", g);
}
*/
