/**************************************************************
 This code generates a set of states distributed according to 
 1-dimensional ising probability distribution using metropolis


**************************************************************/
 
#include"iostream"
#include"cmath"
#include"fstream"
#include"cstdlib"

using namespace std;

double E(int*, int);

int main(){

 ofstream output;
 output.open("training.data");
 
 // initialization
 int nspins = 6;
 double T = 1;
 int npoints = 10001;
 int *sigma1 = new int[nspins];
 int *sigma2 = new int[nspins];

 for(int i=0;i < nspins; i++)
    sigma1[i] = -1;

 double a;
 int rand_int;
 
 // metropolis algorithm
 for(int n = 0; n<npoints; n++){

   // compute energy for the actual state
   a = E(sigma1, nspins);

   // choose one spin randomly and flip it
   rand_int = rand() % nspins;
   for(int i=0;i < nspins; i++)
       sigma2[i] =  sigma1[i];
   if(sigma1[rand_int] == -1) sigma2[rand_int] = 1;
   else                       sigma2[rand_int] = -1;

   // compute the energy for the new spin and the number c
   int b = E(sigma2, nspins);
   double c = exp(-double(b-a)/T);

   // compute a random number r between 0 and 1
   double r = (double)rand()/RAND_MAX;

   // if r<c keep the new state, if r>c keep the old one
   if(r<c){

    for(int i=0; i<nspins-1; i++)
         output << (1+sigma2[i])/2 << ",";

    output << (1+sigma2[nspins-1])/2 << endl;
    output << endl;

    for(int i=0; i<nspins; i++)
         sigma1[i]= sigma2[i];
 
   }else{

     for(int i=0; i<nspins-1; i++)
          output << (1+sigma1[i])/2<< ",";

     output << (1+sigma1[nspins-1])/2<< endl;
     output << endl;
   };

 };

 return 0;
}


double E(int* a, int n)
{
  double H = 0;
  for(int i=0; i<n-1; i++)
  H += -(*(a+i)* *(a+i+1));
  return H;
}

