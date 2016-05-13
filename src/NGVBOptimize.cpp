#include "NGVBOptimize.hpp"
#include <vector>


#include <boost/math/special_functions/digamma.hpp>

#include "boost/random/normal_distribution.hpp"
#include "boost/random/mersenne_twister.hpp"

#define SWAPD(x,y) {tmpD=x;x=y;y=tmpD;}
#define ZERO_LIMIT 1e-12

NGVBOptimize::NGVBOptimize(int transcript_size) {
	M = transcript_size;
}
NGVBOptimize::NGVBOptimize(std::vector<Transcript>& transcripts,
			std::vector<std::pair<const TranscriptGroup, TGValue> >& eqVec,
			Eigen::VectorXd effLens,
			std::vector<tbb::atomic<double> > _alphas,
			int seed) {
	converged = false;
	error = false;
	iteration = 0;

	M = transcripts.size();
	N = eqVec.size();
	totalReads = 0;

    for (auto& kv : eqVec) {
        uint64_t count = kv.second.count;
        // for each transcript in this class
        const TranscriptGroup& tgroup = kv.first;
        if (tgroup.valid) {
            const std::vector<uint32_t>& txps = tgroup.txps;
            std::vector<double> auxs; // = kv.second.weights;
            double normalizer = 0;
            for(int i=0;i<txps.size();i++) {
            	normalizer+=1/effLens(txps[i]);
            }
            for(int i=0;i<txps.size();i++) {
            	auxs.push_back((1/effLens(txps[i]))/normalizer);
            }
            txpGroupLabels.push_back(txps);
            txpGroupWeights.push_back(auxs);
            txpGroupCounts.push_back(count);
            totalReads += count;
        }
    }
/*    std::cout<<M<<"\n";
    std::cout<<txpGroupLabels.size()<<"\n";
    for(int i=0;i<txpGroupLabels.size(); i++){
    	for(int j=0;j<txpGroupLabels[i].size();j++){
    		std::cout<<i<<" "<<j<<" "<<txpGroupLabels[i][j]<<" "<<txpGroupWeights[i][j]<<"\n";
    	}
    	std::cout<<"txpGroupCounts "<<txpGroupCounts[i]<<"\n";
    }
    std::cout<<txpGroupWeights.size()<<"\n";
    std::cout<<txpGroupCounts.size()<<"\n";
*/
 	digA_pH =  new double[M];
	phiHat = new double[M];
	alpha = new double[M];
	
	for(size_t i=0;i<M;i++) 
		alpha[i]= 1;//_alphas[i];

	T = 0;
	rowStart =  new size_t[N];
	rowStart[0] = 0;
	for(size_t eqID=0;eqID<N;eqID++) {		
		size_t groupSize = txpGroupLabels[eqID].size();
		T += groupSize; 		
		if(eqID<N-1)
			rowStart[eqID+1] = rowStart[eqID]+groupSize;
		const std::vector<double> auxs = txpGroupWeights[eqID];
		const std::vector<uint32_t> txps = txpGroupLabels[eqID];
		for(size_t i=0;i<groupSize;i++) {
			beta.push_back(log(auxs[i]));
			beta_col.push_back(txps[i]);	
			classID.push_back(eqID);
		}
	}

	boost::random::mt11213b rng_mt;
	rng_mt.seed(seed);
	boost::random::normal_distribution<long double> normalD;
	phi_sm = new double[T];
	phi = new double[T];
	for(size_t i=0;i<T;i++) 
		phi_sm[i] = normalD(rng_mt);	
	unpack(phi_sm,NULL); 
 

	double alphaS=0,gAlphaS=0;
	for(size_t i=0;i<M;i++){
		alphaS+=alpha[i];
		gAlphaS+=lgamma(alpha[i]);
	}
	boundConstant = lgamma(alphaS) - gAlphaS - lgamma(alphaS+totalReads);//N); //DaNGERRR


	gradPhi=natGrad=gradGamma=searchDir=tmpD=phiOld=NULL;
	gradPhi = new double[T];
	phiOld = NULL;
	natGrad = new double[T];
	gradGamma = new double[T];
	searchDir = new double[T];
	boundOld=getBound();
	squareNormOld=1;
	valBeta=0;
}



void NGVBOptimize::negGradient(double res[]){
   size_t i;
   for(i=0;i<M;i++){
   		//std::cout << alpha[i]+phiHat[i] << " alpha[i]+phiHat[i] \n";
    	digA_pH[i]=boost::math::digamma(alpha[i]+phiHat[i]);
	}
	// beta is logged now
	for(i=0;i<T;i++)res[i]= -(beta[i] - phi_sm[i] - 1.0 + digA_pH[beta_col[i]]);//*txpGroupCounts[classID[i]];
}

double NGVBOptimize::getBound(){
   // the lower bound on the model likelihood
   double A=0,B=0,C=0;
   size_t i;
   for(i=0;i<T;i++){
      // beta is logged now.
      A += phi[i] * beta[i];
      // PyDif use nansum instead of ZERO_LIMIT (nansum sums all elements treating NaN as zero
      if(phi[i]>ZERO_LIMIT){
         B += phi[i] * phi_sm[i];
      }
   }
   for(i=0;i<M;i++){
      C += lgamma(alpha[i]+phiHat[i]);
   }
   return (A+B+C+boundConstant);
}



double NGVBOptimize::logSumExpVal(double* val, size_t st, size_t en) const{
	if(st<0)st = 0;
	if((en == -1) || (en > T)) en = T;
	if(st >= en) return 0;
	size_t i;
	double sumE = 0, m = val[st];
	for(i = st; i < en; i++)
		if(val[i] > m) m = val[i];
	for(i = st; i < en; i++)
	 	sumE += exp(val[i] - m);
	return  m + log(sumE);
}
void NGVBOptimize::sumCols(double* val, double* res) const{
	memset(res,0,M*sizeof(double));
	for(size_t i=0;i<T;i++) {
		res[beta_col[i]] += val[i];//*txpGroupCounts[classID[i]]; 
	}
}

void NGVBOptimize::sumColsWeighed(double* val, double* res) const{
	memset(res,0,M*sizeof(double));
	for(size_t i=0;i<T;i++) {
		res[beta_col[i]] += val[i]*txpGroupCounts[classID[i]]; 
	}
}


void NGVBOptimize::softmaxInplace(double* val,double* res) {
	double logRowSum = 0;
	long i,r;
//	memset(res,0,T*sizeof(double));
//	res = new double[T];
	for(size_t eqID=0;eqID<N;eqID++){
		logRowSum = logSumExpVal(val, rowStart[eqID],rowStart[eqID+1]);
		for(i=rowStart[eqID];i<rowStart[eqID+1];i++){
			val[i] = val[i] - logRowSum ;//+ log(txpGroupCounts[eqID]);
			res[i] = exp( val[i] )*txpGroupCounts[eqID];
 		}
	}
}


void NGVBOptimize::unpack(double* vals,double* adds){
   if(adds==NULL){
      if(vals!=phi_sm)
      	memcpy(phi_sm,vals,T*sizeof(double));
   }else{
      for(size_t i=0;i<T;i++) 
      	phi_sm[i] = vals[i]+adds[i];
   }
   softmaxInplace(phi_sm,phi); //softmax  phi_sm into phi; and set phi_sm to log(phi)

   sumCols(phi,phiHat); // sumCols of phi into phiHat
}

void NGVBOptimize::optimizationStep(){
	negGradient(gradPhi);
	squareNorm=0;
	valBeta = 0;
	valBetaDiv = 0;
	for(size_t eqID=0;eqID<N;eqID++) {
		
		phiGradPhiSum_r = 0;
		for(size_t i = rowStart[eqID]; i < rowStart[eqID+1]; i++) 
			phiGradPhiSum_r += phi[i] * gradPhi[i];

		for(size_t i = rowStart[eqID]; i < rowStart[eqID+1]; i++){
			natGrad_i = (gradPhi[i] - phiGradPhiSum_r);
			gradGamma_i = natGrad_i * phi[i];
			squareNorm += natGrad_i * gradGamma_i;
			natGrad[i] = natGrad_i;		
		}
	}
	//Method: OPTT_FR
	if (iteration % (N*M)==0)
		valBeta = 0;
	else {
		valBeta = squareNorm / squareNormOld;	
	}

	if(valBeta>0){
		usedSteepest = false;
		for(i=0;i<T;i++)
			searchDir[i]= (-natGrad[i] + valBeta*searchDir[i]);//*txpGroupCounts[classID[i]];
	}else{
		usedSteepest = true;
		for(i=0;i<T;i++)
			searchDir[i]= -natGrad[i];//*txpGroupCounts[classID[i]];
	}


	//try conjugate step
	SWAPD(gradPhi,phiOld);
	memcpy(phiOld,phi_sm,T*sizeof(double));
	unpack(phiOld,searchDir);

	bound = getBound();
	iteration++;
	// make sure there is an increase in L, else revert to steepest
	if((bound<boundOld) && (valBeta>0)){
		usedSteepest = true;
		for(size_t i=0;i<T;i++)
			searchDir[i]= -natGrad[i];
		unpack(phiOld,searchDir);
		bound = getBound();
		// this should not be increased: iteration++;
	}
	if(bound<boundOld) { // If bound decreased even after using steepest, step back and quit.
		unpack(phiOld, NULL);
	}
	SWAPD(gradPhi,phiOld);

}

bool NGVBOptimize::checkConvergance(double ftol, double gtol){
	if(abs(bound-boundOld)<=ftol){
		std::cout<<"\nEnd: converged (ftol)\n";
		converged = true;
	}
	//std::cout<<"squareNorm " <<squareNorm<<std::endl;
	if(squareNorm<=gtol){
		std::cout<<"\nEnd: converged (gtol)\n";
		converged = true;
	}

	squareNormOld=squareNorm;
	boundOld=bound;	
	return converged;
}
double NGVBOptimize::getConvVal(){
	return bound-boundOld;
}

double* NGVBOptimize::getAlphas(){
   double *alphas = new double[M];
   //sumColsWeighed(phi,phiHat);
   for(long i=0;i<M;i++) { alphas[i] = alpha[i] + phiHat[i];}
   	std::cout<<"sss"<<totalReads<<"\n";
   return alphas;
}

