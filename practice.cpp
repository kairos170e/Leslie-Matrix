#include<iostream>
using namespace std;
const int row = 3;
const int num = 4;

int     years = 4;

float   X[num]                      = {};
float   breeding_rate[num]          = {};
float   survival_rate[num]          = {};
float   LeslieMatrix[num][num]      = {};
float   result[num][1]              = {};
float   normalized_result[num][1]   = {};

void multiplying();
int main()
{
    // input raw data...
    for (int x = 0; x < num; x++)
        cin >> X[x];
    for (int y = 0; y < num; y++)
        cin >> breeding_rate[y];
    for (int z = 0; z < num; z++)
        cin >> survival_rate[z];

    for (int j = 0; j < num; j++)
        LeslieMatrix[0][j] = breeding_rate[j];
    for (int k = 1; k < num; k++)
        LeslieMatrix[k][k - 1] = survival_rate[k - 1];

    for (int l = 0; l < num; l++)
        result[l][0] = X[l];

    // testing...
    //printf("Leslie Matrix\n");
    //for (int l = 0; l < num; l++)
    //{
    //    for (int m = 0; m < num; m++)
    //    {
    //        printf("%f ,", LeslieMatrix[l][m]);
    //    }
    //    printf("\n");
    //}

    for (int m = 0; m < years; m++)
    {
        multiplying();

        cout << "Year " << m + 1 << " Population: ";
        float total = 0;
        for (int c = 0; c < num; c++)
            total += result[c][0];
        for (int c = 0; c < num; c++)
        {
            printf("%7.2f ", result[c][0]);
            normalized_result[c][0] = result[c][0] / total * 100;
        }

        cout << endl;
        cout << "Distribution:      ";
        for (int c = 0; c < num; c++)
            printf("%6.2f%% ", normalized_result[c][0]);
        cout << endl;
    }

    return 0;
}

void multiplying()
{
    for (int a = 0; a < num; a++)
    {
        result[a][0] = 0.0;
        for (int b = 0; b < num; b++)
        {
            result[a][0] += LeslieMatrix[a][b] * X[b];
        }
        X[a] = result[a][0];
    }
}