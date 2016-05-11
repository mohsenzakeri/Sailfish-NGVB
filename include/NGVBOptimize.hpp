#ifndef __NGVBOptimize_HPP__
#define __NGVBOptimize_HPP__

#include <boost/math/special_functions/digamma.hpp>
#include <vector>
#include <cstdint>
#include "Transcript.hpp"
#include "TranscriptGroup.hpp"
#include "EquivalenceClassBuilder.hpp"
#include "Eigen/Dense"

//#include <stdint.h>	

class NGVBOptimize {
	private:
		double * alpha; // prior over expression
		size_t T,M,N;
		double * phiHat;
		double * digA_pH;
		double * phi_sm;
		double * phi;
		double boundConstant;
		size_t* rowStart;
	

		bool converged;
		bool error;

		std::vector<double> beta;
		std::vector<uint32_t> beta_col;
		std::vector<uint32_t> classID;

		std::vector<std::vector<uint32_t> > txpGroupLabels;
		std::vector<std::vector<double> > txpGroupWeights;
		std::vector<uint64_t> txpGroupCounts;

		bool usedSteepest;
		long iteration,i,r;
		double boundOld,bound,squareNorm,squareNormOld,valBeta,valBetaDiv,natGrad_i,gradGamma_i,phiGradPhiSum_r;
		double *gradPhi,*natGrad,*gradGamma,*searchDir,*tmpD,*phiOld;

	public:
		NGVBOptimize(int transcript_size);
		NGVBOptimize(std::vector<Transcript>& transcripts,
					std::vector<std::pair<const TranscriptGroup, TGValue> >& eqVec,
					Eigen::VectorXd effLens,
					std::vector<tbb::atomic<double> > _alphas,
					int seed);
		
		void negGradient(double* res);
		double getBound();
		void unpack(double* vals,double* adds);
		void softmaxInplace(double* val,double* res);
		void sumCols(double* val, double* res) const;
		double logSumExpVal(double* val, size_t st, size_t en) const;

		void optimizationStep();
		bool checkConvergance(double ftol, double gtol);
		double* getAlphas();

		double getConvVal();


};

#endif